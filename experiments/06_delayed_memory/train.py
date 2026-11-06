"""Paired FM-only delayed-memory training; no attribute labels enter the models."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import time

import torch

from model import ROOT, ARMS, paired_models, precision, width
from data import load_dataset, episode_latents


def preload(root, rows):
    """Read each episode once; intern identical hidden chunks by scene."""
    cached, blanks = [], {}
    for row in rows:
        item = episode_latents(root, row)
        hidden = item['hidden'].half().contiguous()
        key = json.dumps(row['scene'], sort_keys=True)
        if key in blanks:
            if not torch.equal(blanks[key], hidden):
                raise ValueError('Hidden chunk differs within a scene')
        else:
            blanks[key] = hidden
        cached.append({'cues': item['cues'].half().contiguous(),
                       'hidden': blanks[key], 'return': item['return'].half().contiguous()})
    return cached


def pack(cache, indices, device):
    return {key: torch.stack([cache[i][key] for i in indices]).to(device).float()
            for key in ('cues', 'hidden', 'return')}


def condition_pair(model, batch, delay, position):
    """Teacher-forced full BPTT. position is a uniform hidden target in [0,d)."""
    state = None
    for cue in batch['cues'].unbind(1):
        state = model.observe(cue, state)
    boundary = cue[:, :, -1:].half().clone()
    hidden_state = hidden_boundary = None
    for j in range(delay):
        if j == position:
            hidden_state, hidden_boundary = state, boundary
        state = model.observe(batch['hidden'], state)
        boundary = batch['hidden'][:, :, -1:].half().clone()
    return torch.cat((hidden_state, state)), torch.cat((hidden_boundary, boundary))


def noise_batch(batch, generator, drop_generator, device, dropout=.1):
    future = torch.cat((batch['hidden'], batch['return']))
    t = torch.rand(len(future), device=device, generator=generator)
    noise = torch.randn(future.shape, device=device, generator=generator)
    w = t[:, None, None, None, None]
    dropped = torch.rand(len(future), device=device, generator=drop_generator) < dropout
    phase = torch.cat((torch.zeros(len(future)//2, device=device, dtype=torch.long),
                       torch.ones(len(future)//2, device=device, dtype=torch.long)))
    return (1-w)*noise+w*future, future-noise, t, dropped, phase


@torch.inference_mode()
def diagnostic(models, cache, rows, device):
    """Fixed diagnostic only: no checkpoint choice and no final-eval imports."""
    rng = torch.Generator(device=device).manual_seed(93421)
    totals = {arm: [0., 0., 0] for arm in ARMS}
    grouped = defaultdict(list)
    for i, row in enumerate(rows):
        grouped[(len(cache[i]['cues']), int(row['delay']))].append(i)
    for model in models.values():
        model.eval()
    for (_, delay), indices in grouped.items():
        for start in range(0, len(indices), 8):
            batch = pack(cache, indices[start:start+8], device)
            noisy, target, t, dropped, phase = noise_batch(batch, rng, rng, device, dropout=0.)
            for arm, model in models.items():
                with precision(device):
                    state, boundary = condition_pair(model, batch, delay, delay//2)
                    output = model.predict(boundary, noisy, t, state, phase, dropped)
                per = (output.float()-target).square().flatten(1).mean(1)
                h, r = per.chunk(2)
                totals[arm][0] += h.sum().item()
                totals[arm][1] += r.sum().item()
                totals[arm][2] += len(h)
    return {k: {'hidden_fm': v[0]/v[2], 'return_fm': v[1]/v[2],
                'fm': (v[0]+v[1])/(2*v[2])} for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    width.setup()
    device = torch.device(args.device)
    run = args.root/'runs'/f'delayed_memory_seed{args.seed}'
    if (run/'last.pt').exists() and not args.resume:
        raise FileExistsError('Run exists; pass --resume')
    previous = torch.load(run/'last.pt', map_location='cpu', weights_only=False) if args.resume else None
    rows = load_dataset(args.root, 'train')
    cache = preload(args.root, rows)
    val_rows = [r for r in load_dataset(args.root, 'val') if int(r['delay']) == 3][:48]
    if not val_rows:
        raise ValueError('Expected delay-3 diagnostic validation rows')
    val_cache = preload(args.root, val_rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        delay = int(row['delay'])
        if delay < 1:
            raise ValueError('Training delay must be positive')
        groups[(row['mode'], delay, len(cache[i]['cues']))].append(i)
    group_keys = sorted(groups)
    models = paired_models(args.seed, device)
    optimizers = {arm: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
                  for arm, model in models.items()}
    data_rng = random.Random(args.seed+100)
    noise_rng = torch.Generator(device=device).manual_seed(args.seed+200)
    drop_rng = torch.Generator(device=device).manual_seed(args.seed+400)
    config = {'seed': args.seed, 'steps': args.steps, 'arms': ARMS, 'batch_size': 8,
              'lr': 2e-4, 'weight_decay': .01, 'gradient_clip': 1., 'dropout': .1,
              'precision': 'BF16 autocast CUDA; FP32 loss and CPU debug',
              'state': '16x256 FP16; 8 KiB', 'boundary': '16x1x24x24 FP16; 18 KiB',
              'update': 'g=sigmoid(MLP(concat(old,E))); (1-g)*old+g*E; initial U(0,E)',
              'phase': 'known hidden=0/return=1 embedding added to denoiser time path',
              'loss': 'FM only; half uniform hidden position, half return; no cue loss',
              'bptt': 'all observed cues and hidden chunks; no state detach',
              'dropout_scope': 'joint read-side state/boundary; recurrence unchanged',
              'paired': 'identical parameters, episode batches, position, noise, time, dropout',
              'checkpoint_selection': 'final fixed budget; diagnostic never selects checkpoint',
              'diagnostic_rows': [r['id'] for r in val_rows],
              'peak_memory_definition': 'CUDA total allocated peak during each arm update; includes all resident models/optimizers',
              'parameters': {a: sum(p.numel() for p in m.parameters()) for a, m in models.items()}}
    start, elapsed = 0, 0.
    arm_seconds, peak_bytes = {a: 0. for a in ARMS}, {a: 0 for a in ARMS}
    if previous is not None:
        if previous['config']['seed'] != args.seed or previous['step'] > args.steps:
            raise ValueError('Resume seed/step mismatch')
        for arm in ARMS:
            models[arm].load_state_dict(previous['models'][arm])
            optimizers[arm].load_state_dict(previous['optimizers'][arm])
        data_rng.setstate(previous['rng']['data'])
        noise_rng.set_state(previous['rng']['noise'])
        drop_rng.set_state(previous['rng']['drop'])
        torch.set_rng_state(previous['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(previous['rng']['cuda'])
        start, elapsed = previous['step'], previous['elapsed_seconds']
        arm_seconds, peak_bytes = previous['arm_seconds'], previous['peak_allocated_bytes']
        del previous
    run.mkdir(parents=True, exist_ok=True)
    (run/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    began = time.perf_counter()
    recent = {a: [] for a in ARMS}
    for step in range(start+1, args.steps+1):
        key = data_rng.choice(group_keys)
        indices = data_rng.choices(groups[key], k=8)
        batch = pack(cache, indices, device)
        delay = key[1]
        position = data_rng.randrange(delay)
        noisy, target, t, dropped, phase = noise_batch(batch, noise_rng, drop_rng, device)
        for arm, model in models.items():
            model.train()
            optimizers[arm].zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            arm_start = time.perf_counter()
            with precision(device):
                state, boundary = condition_pair(model, batch, delay, position)
                output = model.predict(boundary, noisy, t, state, phase, dropped)
            loss = (output.float()-target).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite loss/gradient at {step}, {arm}')
            optimizers[arm].step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                peak_bytes[arm] = max(peak_bytes[arm], torch.cuda.max_memory_allocated(device))
            arm_seconds[arm] += time.perf_counter()-arm_start
            recent[arm].append(loss.item())
            del state, boundary, output, loss
        if step % 50 == 0:
            print(json.dumps({'step': step, 'train_fm': {a: sum(v[-50:])/len(v[-50:])
                                                       for a, v in recent.items()}}), flush=True)
        if step % 1000 == 0 or step == args.steps:
            metrics = {'step': step, 'diagnostic': diagnostic(models, val_cache, val_rows, device),
                       'train_fm': {a: sum(v)/len(v) for a, v in recent.items()},
                       'elapsed_seconds': elapsed+time.perf_counter()-began,
                       'arm_seconds': arm_seconds, 'peak_allocated_bytes': peak_bytes}
            with (run/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics)+'\n')
            checkpoint = {'step': step, 'config': config,
                          'models': {a: m.state_dict() for a, m in models.items()},
                          'optimizers': {a: o.state_dict() for a, o in optimizers.items()},
                          'elapsed_seconds': metrics['elapsed_seconds'], 'arm_seconds': arm_seconds,
                          'peak_allocated_bytes': peak_bytes,
                          'rng': {'data': data_rng.getstate(), 'noise': noise_rng.get_state(),
                                  'drop': drop_rng.get_state(), 'torch': torch.get_rng_state(),
                                  'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}}
            torch.save(checkpoint, run/'last.tmp.pt')
            (run/'last.tmp.pt').replace(run/'last.pt')
            if step == args.steps:
                torch.save({'step': step, 'config': config, 'models': checkpoint['models']}, run/'final.pt')
            print(json.dumps(metrics), flush=True)
            recent = {a: [] for a in ARMS}


if __name__ == '__main__':
    main()
