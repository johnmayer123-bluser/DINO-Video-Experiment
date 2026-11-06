"""Paired B/C evaluation and interventions on the old recurrent state."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '01_history_semantics'))
from common import evaluation_noise
from model import ROOT, LEVELS, RecurrentModel, compact_metrics, load_split, setup, write_json

ARMS = ('recent_B', 'recurrent_B', 'recent_C', 'recurrent_C')


def precision(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def donor_mapping(rows):
    """Choose donors using metadata only, invariant to manifest permutation."""
    rng = np.random.default_rng(2026)
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]['id'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Window ids must be unique')
    donors = np.empty(len(rows), dtype=np.int64)
    for i in ordered:
        row = rows[i]
        category = row['video'].split('/')[0]
        eligible = [j for j in ordered if rows[j]['video'].split('/')[0] == category
                    and rows[j]['video'] != row['video'] and rows[j]['group'] != row['group']]
        donors[i] = eligible[int(rng.integers(len(eligible)))] if eligible else -1
    return donors


@torch.inference_mode()
def evaluate_models(models, rows, z, split, repeats=4, interventions=True, batch_size=8):
    if set(models) != {'recent', 'recurrent'}:
        raise ValueError('Expected models keyed recent and recurrent')
    if split not in ('val', 'test') or repeats < 1 or batch_size < 1:
        raise ValueError('Invalid evaluation configuration')
    if len(rows) == 0 or tuple(z.shape) != (len(rows), 16, 12, 24, 24):
        raise ValueError('Expected nonempty, manifest-aligned A/B/C latents [N,16,12,24,24]')
    device = next(models['recurrent'].parameters()).device
    modes = {name: model.training for name, model in models.items()}
    keys = ARMS + (('reset_C', 'shuffled_C') if interventions else ())
    repeated, frames = {key: [] for key in keys}, {key: [] for key in keys}
    extras = {}
    try:
        for model in models.values():
            model.eval()
        # Every donor mA is computed before predictions; no C encoding is ever used.
        if interventions:
            donors = donor_mapping(rows)
            valid = donors >= 0
            extras = {'donor_indices': donors, 'donor_valid': valid,
                      'donor_ids': np.asarray([rows[i]['id'] if i >= 0 else '' for i in donors]),
                      'donor_seed': np.asarray(2026)}
            old_states = []
            for start in range(0, len(z), batch_size):
                a = z[start:start + batch_size, :, :4].to(device).float()
                with precision(device):
                    old_states.append(models['recurrent'].update(models['recurrent'].encode(a)).half().cpu())
            old_states = torch.cat(old_states)
        for repeat in range(repeats):
            errors = {key: [] for key in keys}
            frame_errors = {key: [] for key in keys}
            for start in range(0, len(z), batch_size):
                latent = z[start:start + batch_size].to(device, non_blocking=True).float()
                a, b, c = latent.split(4, dim=2)
                with precision(device):
                    states = {name: model.states(a, b) for name, model in models.items()}
                    conditions = {name + '_B': (model, a, states[name][0]) for name, model in models.items()}
                    conditions.update({name + '_C': (model, b, states[name][1]) for name, model in models.items()})
                    if interventions:
                        model = models['recurrent']
                        encoded_b = model.encode(b)
                        m_a = states['recurrent'][0]
                        # Intervene inside U(E(B), mA), retaining the real B boundary.
                        conditions['reset_C'] = (model, b, model.update(encoded_b, torch.zeros_like(m_a)))
                        indices = donors[start:start + len(a)].copy()
                        # Ineligible rows carry a documented self-state placeholder;
                        # donor_valid excludes these values from shuffled statistics.
                        invalid = indices < 0
                        indices[invalid] = np.arange(start, start + len(a))[invalid]
                        shuffled = old_states[indices].to(device)
                        conditions['shuffled_C'] = (model, b, model.update(encoded_b, shuffled))
                for chunk, future, offset in (('B', b, 0), ('C', c, 500_003)):
                    noise = evaluation_noise(future.shape, start, repeat,
                                             (10042 if split == 'val' else 20042) + offset).to(device)
                    target = future - noise
                    chunk_keys = [key for key in keys if key.endswith('_' + chunk)]
                    batch, by_frame = {key: [] for key in chunk_keys}, {key: [] for key in chunk_keys}
                    for level in LEVELS:
                        t = torch.full((len(a),), level, device=device)
                        noisy = (1 - level) * noise + level * future
                        for key in chunk_keys:
                            model, history, state = conditions[key]
                            with precision(device):
                                output = model.predict(history, noisy, t, state)
                            square = (output.float() - target).square()
                            batch[key].append(square.flatten(1).mean(1).cpu())
                            by_frame[key].append(square.mean((1, 3, 4)).cpu())
                    for key in chunk_keys:
                        errors[key].append(torch.stack(batch[key], 1))
                        frame_errors[key].append(torch.stack(by_frame[key], 1))
            for key in keys:
                repeated[key].append(torch.cat(errors[key]).numpy())
                frames[key].append(torch.cat(frame_errors[key]).numpy())
        repeated = {key: np.stack(value) for key, value in repeated.items()}
        frames = {key: np.stack(value) for key, value in frames.items()}
        return {key: value.mean(0) for key, value in repeated.items()}, repeated, frames, extras
    finally:
        for name, model in models.items():
            model.train(modes[name])


def save_evaluation(run, models, rows, z, split, seed, step, batch_size=8):
    errors, repeated, frames, extras = evaluate_models(models, rows, z, split, batch_size=batch_size)
    stem = f'step{step:05d}_{split}'
    run.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run / f'errors_{stem}.npz', **errors,
                        **{k + '_repeats': v for k, v in repeated.items()},
                        **{k + '_by_frame_repeats': v for k, v in frames.items()}, **extras,
                        ids=np.asarray([r['id'] for r in rows]), noise_levels=np.asarray(LEVELS),
                        noise_seed=10042 if split == 'val' else 20042, c_noise_offset=500003, noise_repeats=4)
    valid = extras['donor_valid']
    eligible_rows = [row for row, keep in zip(rows, valid) if keep]
    def metrics(values):
        full = compact_metrics(rows, {k: v for k, v in values.items() if k != 'shuffled_C'})
        full['shuffled_eligible'] = (compact_metrics(eligible_rows, {k: values[k][valid]
                                      for k in ('shuffled_C', 'recurrent_C')}) if eligible_rows else None)
        return full
    summary = {'seed': seed, 'step': step, 'split': split, 'metrics': metrics(errors),
               'first': metrics({k: v.mean(0)[..., 0] for k, v in frames.items()}),
               'later': metrics({k: v.mean(0)[..., 1:].mean(-1) for k, v in frames.items()}),
               'cohorts': {'full_windows': len(rows), 'shuffled_eligible_windows': int(valid.sum()),
                           'shuffled_excluded_windows': int((~valid).sum()),
                           'shuffled_rule': 'same split/class; different video and source group'},
               'noise_seed': 10042 if split == 'val' else 20042, 'c_noise_offset': 500003,
               'noise_repeats': 4, 'donor_seed': 2026, 'state_kib': 8, 'boundary_kib': 18}
    write_json(run / f'{split}_manifest.json', rows)
    write_json(run / f'summary_{stem}.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    setup()
    rows, z = load_split(args.root, args.split)
    for seed in args.seeds:
        run = args.root / f'runs/recurrent_state_seed{seed}'
        checkpoint = torch.load(run / 'final.pt', map_location='cpu', weights_only=False)
        if checkpoint['step'] != 6000:
            raise ValueError('Use final 6000-step weights')
        models = {kind: RecurrentModel(kind).cuda() for kind in ('recent', 'recurrent')}
        for kind, model in models.items():
            model.load_state_dict(checkpoint['models'][kind])
        save_evaluation(run, models, rows, z, args.split, seed, checkpoint['step'], args.batch_size)


if __name__ == '__main__':
    main()
