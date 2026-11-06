"""Paired keep-only return training; one arm resides on the GPU at a time."""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import torch

from common import WORK_ROOT, write_json
from model import ARMS, ConditionModel, precision, width
from data import load_rows, load_pair


def pack(rows, indices, device, cache=None):
    pairs = [(load_pair(rows[i]['pair_id']) if cache is None else cache[rows[i]['pair_id']])
             for i in indices]
    # Pair-major ordering keeps the two counterfactual variants adjacent.
    return {
        'boundary': torch.stack([p['boundary'] for p in pairs]).repeat_interleave(2, 0).to(device).float(),
        'cue': torch.stack([p['cue'] for p in pairs]).flatten(0, 1).to(device).float(),
        'target': torch.stack([p['target'] for p in pairs]).flatten(0, 1).to(device).float(),
        'mask': torch.stack([p['mask'] for p in pairs]).flatten(0, 1).to(device).float(),
    }


def noise_batch(target, generator, drop_generator):
    """Each pair shares its noise, interpolation time, and dropout draw."""
    device = target.device
    pairs = len(target) // 2
    low = torch.rand(pairs, device=device, generator=generator) < .5
    t = torch.rand(pairs, device=device, generator=generator)
    t = torch.where(low, .2*t, t).repeat_interleave(2)
    noise = torch.randn((pairs, *target.shape[1:]), device=device,
                        generator=generator).repeat_interleave(2, 0)
    dropped = (torch.rand(pairs, device=device, generator=drop_generator) < .1).repeat_interleave(2)
    w = t[:, None, None, None, None]
    return (1-w)*noise+w*target, target-noise, t, dropped


def mask_loss(output, target, mask):
    # Channel-average first; normalize each region over all T,H,W per sample.
    error = (output.float()-target.float()).square().mean(1, keepdim=True)
    axes = (1, 2, 3, 4)
    foreground = (error*mask).sum(axes)/mask.sum(axes)
    background = (error*(1-mask)).sum(axes)/(1-mask).sum(axes)
    loss = (.5*foreground+.5*background).mean()
    return loss, {'weighted': loss.detach(), 'unweighted': error.mean().detach(),
                  'fg': foreground.mean().detach(), 'bg': background.mean().detach()}


def train_arm(arm, args, rows, run, config, cache=None):
    final_path = run / f'final_{arm}.pt'
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location='cpu', weights_only=False)
        if args.resume and checkpoint['step'] == args.steps:
            print(f'{arm}: already complete at step {args.steps}', flush=True)
            return
        raise FileExistsError(f'{final_path} exists; use another --run-dir')
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
    model = ConditionModel().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    data_rng = random.Random(args.seed+100)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed+200)
    drop_rng = torch.Generator(device=device).manual_seed(args.seed+400)
    start, seconds, peak_allocated, peak_reserved = 0, 0., 0, 0
    last_path = run/'last.pt'
    if args.resume and last_path.exists():
        previous = torch.load(last_path, map_location='cpu', weights_only=False)
        if previous['arm'] == arm:
            resumed_config = {k: v for k, v in previous['config'].items() if k != 'parameters'}
            if resumed_config != config or previous['step'] > args.steps:
                raise ValueError('Resume configuration mismatch')
            model.load_state_dict(previous['model'])
            optimizer.load_state_dict(previous['optimizer'])
            start = previous['step']
            data_rng.setstate(previous['rng']['data'])
            noise_rng.set_state(previous['rng']['noise'])
            drop_rng.set_state(previous['rng']['drop'])
            torch.set_rng_state(previous['rng']['torch'])
            if device.type == 'cuda':
                torch.cuda.set_rng_state_all(previous['rng']['cuda'])
            seconds = previous['resource']['training_seconds']
            peak_allocated = previous['resource']['peak_allocated_bytes']
            peak_reserved = previous['resource']['peak_reserved_bytes']
        del previous
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    config = dict(config, parameters=sum(p.numel() for p in model.parameters()))
    print(f'{arm}: training {start + 1}..{args.steps}, 8 scene pairs / 16 targets per step', flush=True)
    recent = []
    for step in range(start+1, args.steps+1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        batch = pack(rows, data_rng.sample(range(len(rows)), 8), device, cache)
        noisy, velocity, t, dropped = noise_batch(batch['target'], noise_rng, drop_rng)
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            state = (model.observe(batch['cue']) if arm == 'C' else
                     torch.zeros((len(noisy), 16, 256), dtype=torch.float16, device=device))
            output = model.predict(batch['boundary'], noisy, t, state, phase=1, dropped=dropped)
        loss, metrics = mask_loss(output, velocity, batch['mask'])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f'{arm}: nonfinite loss/gradient at {step}')
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        seconds += time.perf_counter()-began
        recent.append({key: value.item() for key, value in metrics.items()})
        del batch, noisy, velocity, state, output, loss, metrics, norm, t, dropped
        resource = {'training_seconds': seconds, 'peak_allocated_bytes': peak_allocated,
                    'peak_reserved_bytes': peak_reserved}
        if step % 50 == 0 or step == args.steps:
            record = {'arm': arm, 'seed': args.seed, 'step': step,
                      **{k: sum(r[k] for r in recent)/len(recent) for k in recent[0]},
                      **resource}
            with (run/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            recent = []
        if step in (2000, 4000) or step == args.steps:
            checkpoint = {'arm': arm, 'step': step, 'config': config, 'model': model.state_dict()}
            name = f'final_{arm}.pt' if step == args.steps else f'step{step:05d}_{arm}.pt'
            torch.save(checkpoint, run/name)
        if step % 200 == 0 or step == args.steps:
            checkpoint = {'arm': arm, 'step': step, 'config': config, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'resource': resource,
                          'rng': {'data': data_rng.getstate(), 'noise': noise_rng.get_state(),
                                  'drop': drop_rng.get_state(), 'torch': torch.get_rng_state(),
                                  'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
            torch.save(checkpoint, run/'last.tmp.pt')
            (run/'last.tmp.pt').replace(last_path)
    write_json(run/f'resource_{arm}.json', dict(resource, arm=arm, step=args.steps,
               timing_scope='synchronized training steps including CPU batch packing and transfer; excludes preload and checkpoint I/O',
               memory_scope='only this model, optimizer and training batch resident on CUDA'))
    del model, optimizer, checkpoint
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=WORK_ROOT/'runs/seed42')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    width.setup()
    rows = list({r['pair_id']: r for r in load_rows('train')}.values())
    if len(rows) != 1000:
        raise ValueError(f'Expected 1000 training scene pairs, got {len(rows)}')
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run/'config.json').exists() and not args.resume:
        raise FileExistsError('Run exists; use --resume or a fresh --run-dir')
    config = {'seed': args.seed, 'steps': args.steps, 'arms': args.arms, 'scene_pairs': 1000,
              'batch_pairs': 8, 'batch_targets': 16, 'lr': 2e-4, 'weight_decay': .01,
              'gradient_clip': 1., 'dropout': .1, 'phase': 1, 'state': 'FP16 16x256',
              'precision': 'BF16 CUDA autocast, FP32 loss; CPU FP32 debug',
              'condition': 'N zero state; C once U(0,E(cue)); shared boundary',
              'loss': 'per-sample 0.5 foreground mean + 0.5 background mean after channel average',
              't': 'equal mixture of U[0,.2] and U[0,1]; t=0 pure noise',
              'pairing': 'same init, scene pairs, noise, t and dropout across arms; variants share noise/t/dropout'}
    write_json(run/'config.json', config)
    print(f'Preloading {len(rows)} training scene pairs to CPU', flush=True)
    cache = {row['pair_id']: load_pair(row['pair_id']) for row in rows}
    cache_bytes = sum(value.numel()*value.element_size()
                      for pair in cache.values() for value in pair.values()
                      if torch.is_tensor(value))
    print(f'CPU training cache ready: {cache_bytes / 2**30:.3f} GiB', flush=True)
    for arm in args.arms:
        train_arm(arm, args, rows, run, config, cache)


if __name__ == '__main__':
    main()
