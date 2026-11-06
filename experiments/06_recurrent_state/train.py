"""Paired recent-only versus recurrent ABC training with two-position BPTT."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import (ROOT, compact_metrics, load_split, paired_models,
                   save_checkpoint, setup, width, write_json)

evaluation_spec = importlib.util.spec_from_file_location(
    'q6_evaluation', Path(__file__).with_name('evaluate.py'))
evaluation = importlib.util.module_from_spec(evaluation_spec)
evaluation_spec.loader.exec_module(evaluation)


def validation(models, rows, z):
    return evaluation.evaluate_models(models, rows, z, 'val', repeats=1, interventions=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--steps', type=int, default=6000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    setup()
    device = torch.device(args.device)
    run = args.root/'runs'/f'recurrent_state_seed{args.seed}'
    if (run/'last.pt').exists() and not args.resume:
        raise FileExistsError('Existing run; use --resume')
    previous = torch.load(run/'last.pt', map_location='cpu', weights_only=False) if args.resume else None
    _, train_z = load_split(args.root, 'train')
    rows, val_z = load_split(args.root, 'val')
    models = paired_models(args.seed, device=device)
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
                  for name, model in models.items()}
    generators = {'data': torch.Generator().manual_seed(args.seed+100),
                  'noise': torch.Generator(device=device).manual_seed(args.seed+200),
                  'drop': torch.Generator(device=device).manual_seed(args.seed+400)}
    config = {
        'seed': args.seed, 'arms': list(models), 'tokens': 16, 'width': 256,
        'grid': [4, 2, 2], 'batch_size': 8, 'sequence_batch_size': 8,
        'effective_prediction_batch': 16, 'loss_positions': ['B_given_A', 'C_given_AB'],
        'loss_reduction': 'equal average over both positions and all latent elements',
        'lr': 2e-4, 'weight_decay': .01, 'clip_grad_norm': 1., 'history_dropout': .1,
        'dropout': 'read-side joint state/boundary mask; internal recurrence unchanged',
        'state_dtype': 'float16', 'state_kib': 8, 'boundary_kib': 18,
        'state_update': 'shared tokenwise GRUCell(256,256); recent resets each chunk',
        'bptt': 'no detach through A/B state updates; one backward per model per step',
        'precision': 'BF16 autocast on CUDA, FP32 loss; FP32 CPU debug',
        'denoiser_init_seed': args.seed, 'encoder_init_seed': args.seed+300,
        'gru_init_seed': args.seed+500,
        'initialization': 'Q4 K16 encoder prefix 256 rows; projection prefix 256 columns times sqrt(1024/256); identical arms',
        'cache': 'cache/ucf6_12fps_192_abc; 49 frames; remove anchor latent; A/B/C each 4 latent slices',
        'objective': 'FM only; no teacher features',
        'checkpoint_selection': 'final equal budget; no best selection',
        'device': str(device),
        'parameters': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}}
    start, elapsed = 0, 0.
    if previous is not None:
        if previous['config']['seed'] != args.seed or previous['step'] > args.steps:
            raise ValueError('Resume seed/step mismatch')
        for name, model in models.items():
            model.load_state_dict(previous['models'][name])
            optimizers[name].load_state_dict(previous['optimizers'][name])
        for name, rng in generators.items():
            rng.set_state(previous['rng'][name])
        torch.set_rng_state(previous['rng']['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(previous['rng']['cuda'])
        start, elapsed = previous['step'], previous.get('elapsed_seconds', 0.)
        del previous
    run.mkdir(parents=True, exist_ok=True)
    write_json(run/'config.json', config)
    write_json(run/'val_manifest.json', rows)
    began = time.perf_counter()
    recent = {name: [] for name in models}
    for step in range(start+1, args.steps+1):
        indices = torch.randint(len(train_z), (8,), generator=generators['data'])
        z = train_z[indices].to(device, non_blocking=True).float()
        a, b, c = z[:, :, :4], z[:, :, 4:8], z[:, :, 8:]
        history, future = torch.cat((a, b)), torch.cat((b, c))
        t = torch.rand(16, device=device, generator=generators['noise'])
        noise = torch.randn(future.shape, device=device, generator=generators['noise'])
        dropped = torch.rand(16, device=device, generator=generators['drop']) < .1
        w = t[:, None, None, None, None]
        noisy, target = (1-w)*noise+w*future, future-noise
        for name, model in models.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            with width.token.precision(device):
                m_a, m_b = model.states(a, b)
                output = model.predict(history, noisy, t, torch.cat((m_a, m_b)), dropped)
            loss = (output.float()-target).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite loss/gradient: {name}, step {step}')
            optimizers[name].step()
            recent[name].append(loss.item())
        if step % 50 == 0:
            print(json.dumps({'step': step, 'fm_loss': {k: float(np.mean(v[-50:]))
                                                       for k, v in recent.items()}}), flush=True)
        if step % 1000 == 0 or step == args.steps:
            errors, _, _, _ = validation(models, rows, val_z)
            metrics = {'step': step, 'split': 'val', 'noise_repeats': 1,
                       **compact_metrics(rows, errors),
                       'train_fm_loss': {k: float(np.mean(v)) for k, v in recent.items()}}
            with (run/'metrics.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metrics)+'\n')
            save_checkpoint(run/'last.pt', {
                'step': step, 'config': config,
                'models': {k: m.state_dict() for k, m in models.items()},
                'optimizers': {k: o.state_dict() for k, o in optimizers.items()},
                'rng': {**{k: g.get_state() for k, g in generators.items()},
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []},
                'elapsed_seconds': elapsed+time.perf_counter()-began})
            print(json.dumps(metrics), flush=True)
            recent = {name: [] for name in models}
    save_checkpoint(run/'final.pt', {'step': args.steps, 'config': config,
                                   'models': {k: m.state_dict() for k, m in models.items()}})


if __name__ == '__main__':
    main()
