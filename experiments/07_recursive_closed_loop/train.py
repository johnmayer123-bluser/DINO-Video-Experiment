"""Teacher-forced Q7 training with full BPTT and a separate blank objective."""
import argparse
import gc
import json
from pathlib import Path
import random
import time

import torch

from common import WORK_ROOT, write_json
from model import ARMS, RecurrentModel, precision, width
from data import load_rows, load_pair


def pack(rows, indices, device, cache=None):
    pairs = [(load_pair(rows[i]['pair_id']) if cache is None else cache[rows[i]['pair_id']])
             for i in indices]
    return {
        'boundary': torch.stack([p['boundary'] for p in pairs]).repeat_interleave(2, 0).to(device).float(),
        'cue': torch.stack([p['cue'] for p in pairs]).flatten(0, 1).to(device).float(),
        'target': torch.stack([p['target'] for p in pairs]).flatten(0, 1).to(device).float(),
        'mask': torch.stack([p['mask'] for p in pairs]).flatten(0, 1).to(device).float(),
        'hidden': torch.stack([p['hidden'] for p in pairs]).repeat_interleave(2, 0).to(device).float(),
    }


def noise_batch(target, generator, drop_generator):
    """Adjacent counterfactual variants share t, noise and dropout."""
    pairs = len(target) // 2
    low = torch.rand(pairs, device=target.device, generator=generator) < .5
    t = torch.rand(pairs, device=target.device, generator=generator)
    t = torch.where(low, .2*t, t).repeat_interleave(2)
    noise = torch.randn((pairs, *target.shape[1:]), device=target.device,
                        generator=generator).repeat_interleave(2, 0)
    dropped = (torch.rand(pairs, device=target.device,
                          generator=drop_generator) < .1).repeat_interleave(2)
    w = t[:, None, None, None, None]
    return (1-w)*noise+w*target, target-noise, t, dropped


def mask_loss(output, target, mask):
    error = (output.float()-target.float()).square().mean(1, keepdim=True)
    axes = (1, 2, 3, 4)
    foreground = (error*mask).sum(axes)/mask.sum(axes)
    background = (error*(1-mask)).sum(axes)/(1-mask).sum(axes)
    loss = (.5*foreground+.5*background).mean()
    return loss, {'return': loss.detach(), 'return_unweighted': error.mean().detach(),
                  'return_fg': foreground.mean().detach(), 'return_bg': background.mean().detach()}


def temporal_conditions(model, batch, delay, blank_index):
    """Predict blank j from s_j, then observe true blanks through s_d.

    One delay/index draw is shared by the batch. No state is detached,
    including when selected for the auxiliary blank objective.
    """
    state = model.observe(batch['cue'])
    aux = None
    for depth in range(delay):
        if depth == blank_index:
            aux = state[:8]
        state = model.observe(batch['hidden'], state)
    blank_boundary = (batch['cue'][:8, :, -1:] if blank_index == 0
                      else batch['hidden'][:8, :, -1:])
    return state, aux, blank_boundary


def train_arm(arm, args, rows, run, config, cache=None):
    final_path = run / f'final_{arm}.pt'
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location='cpu', weights_only=False)
        if args.resume and checkpoint['step'] == args.steps and checkpoint['config'] == config:
            print(f'{arm}: already complete at step {args.steps}', flush=True)
            return
        raise FileExistsError(f'{final_path} exists with incompatible run or without --resume')
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
    model = RecurrentModel(arm).to(device).train()
    if sum(p.numel() for p in model.parameters()) != 14633616:
        raise ValueError('Q6 parameter count changed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    rng = {'data': random.Random(args.seed+100), 'delay': random.Random(642),
           'noise': torch.Generator(device=device).manual_seed(args.seed+200),
           'drop': torch.Generator(device=device).manual_seed(args.seed+400),
           'blank_noise': torch.Generator(device=device).manual_seed(args.seed+201),
           'blank_drop': torch.Generator(device=device).manual_seed(args.seed+401)}
    start, seconds, peak_allocated, peak_reserved = 0, 0., 0, 0
    last_path = run/'last.pt'
    if args.resume and last_path.exists():
        previous = torch.load(last_path, map_location='cpu', weights_only=False)
        if previous['arm'] == arm:
            if previous['config'] != config or previous['step'] > args.steps:
                raise ValueError('Resume configuration mismatch')
            model.load_state_dict(previous['model'])
            optimizer.load_state_dict(previous['optimizer'])
            start = previous['step']
            for key, generator in rng.items():
                if isinstance(generator, random.Random):
                    generator.setstate(previous['rng'][key])
                else:
                    generator.set_state(previous['rng'][key])
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
    print(f'{arm}: training {start+1}..{args.steps}; 16 return + 8 blank targets', flush=True)
    recent = []
    for step in range(start+1, args.steps+1):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        began = time.perf_counter()
        batch = pack(rows, rng['data'].sample(range(len(rows)), 8), device, cache)
        delay = rng['delay'].randint(1, 4)
        blank_index = rng['delay'].randrange(delay)
        noisy, velocity, t, dropped = noise_batch(batch['target'], rng['noise'], rng['drop'])
        bn, bv, bt, bd = noise_batch(batch['hidden'][:8], rng['blank_noise'], rng['blank_drop'])
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            state, aux, blank_boundary = temporal_conditions(model, batch, delay, blank_index)
            output = model.predict(torch.cat((batch['boundary'], blank_boundary)),
                                   torch.cat((noisy, bn)), torch.cat((t, bt)),
                                   torch.cat((state, aux)),
                                   phase=torch.cat((torch.ones(16, device=device, dtype=torch.long),
                                                    torch.zeros(8, device=device, dtype=torch.long))),
                                   dropped=torch.cat((dropped, bd)))
        return_loss, metrics = mask_loss(output[:16], velocity, batch['mask'])
        blank_loss = (output[16:].float()-bv.float()).square().mean()
        loss = return_loss + .1*blank_loss
        metrics.update(total=loss.detach(), blank=blank_loss.detach())
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
        del batch, noisy, velocity, t, dropped, bn, bv, bt, bd, state, aux, blank_boundary
        del output, loss, return_loss, blank_loss, metrics, norm
        resource = {'training_seconds': seconds, 'peak_allocated_bytes': peak_allocated,
                    'peak_reserved_bytes': peak_reserved}
        if step % 50 == 0 or step == args.steps:
            record = {'arm': arm, 'seed': args.seed, 'step': step,
                      **{k: sum(r[k] for r in recent)/len(recent) for k in recent[0]}, **resource}
            with (run/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            recent = []
        if step in (2000, 4000) or step == args.steps:
            checkpoint = {'arm': arm, 'step': step, 'config': config, 'model': model.state_dict()}
            name = f'final_{arm}.pt' if step == args.steps else f'step{step:05d}_{arm}.pt'
            torch.save(checkpoint, run/name)
        if step % 200 == 0 or step == args.steps:
            rng_state = {key: (generator.getstate() if isinstance(generator, random.Random)
                               else generator.get_state()) for key, generator in rng.items()}
            rng_state.update(torch=torch.get_rng_state(),
                             cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            checkpoint = {'arm': arm, 'step': step, 'config': config, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'resource': resource, 'rng': rng_state}
            torch.save(checkpoint, run/'last.tmp.pt')
            (run/'last.tmp.pt').replace(last_path)
    write_json(run/f'resource_{arm}.json', dict(resource, arm=arm, step=args.steps,
               timing_scope='synchronized training steps including CPU packing/transfer; excludes preload/checkpoint I/O',
               memory_scope='only this model, optimizer and batch resident on CUDA'))
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
    if args.steps < 1 or len(set(args.arms)) != len(args.arms):
        parser.error('steps must be positive and arms unique')
    width.setup()
    rows = list({r['pair_id']: r for r in load_rows('train')}.values())
    if len(rows) != 1000:
        raise ValueError(f'Expected 1000 training scene pairs, got {len(rows)}')
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    config = {'seed': args.seed, 'steps': args.steps, 'arms': args.arms, 'scene_pairs': 1000,
              'batch_pairs': 8, 'return_targets': 16, 'blank_targets': 8,
              'parameters': 14633616, 'lr': 2e-4, 'weight_decay': .01,
              'gradient_clip': 1., 'dropout': .1, 'state': 'FP16 16x256',
              'precision': 'BF16 CUDA autocast, FP32 loss; CPU FP32 debug',
              'condition': 'F fixed first cue; L U(0,E(latest)); R U(old,E(latest))',
              'delay': 'per-batch uniform integer 1..4, independent RNG seed642; j uniform 0..d-1',
              'unroll': 'full BPTT; teacher-forced true blank; predict blank j before observing it',
              'loss': 'return mean(.5 fg+.5 bg) + .1 blank MSE; separate normalization',
              't': 'equal mixture U[0,.2] and U[0,1]; t=0 pure noise',
              'pairing': 'same init/batch/d/j/noise/t/drop across arms; variants share noise/t/drop; blank RNG independent'}
    config_path = run/'config.json'
    if config_path.exists():
        if not args.resume:
            raise FileExistsError('Run exists; use --resume or a fresh --run-dir')
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('Run configuration mismatch')
    write_json(config_path, config)
    print(f'Preloading {len(rows)} training scene pairs to CPU', flush=True)
    cache = {row['pair_id']: load_pair(row['pair_id']) for row in rows}
    cache_bytes = sum(value.numel()*value.element_size()
                      for pair in cache.values() for value in pair.values() if torch.is_tensor(value))
    print(f'CPU training cache ready: {cache_bytes/2**30:.3f} GiB', flush=True)
    for arm in args.arms:
        train_arm(arm, args, rows, run, config, cache)


if __name__ == '__main__':
    main()
