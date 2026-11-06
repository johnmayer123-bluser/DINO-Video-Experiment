"""Frozen, paired 32-step rollout evaluation of the delayed-memory experiment.

Decode chunks independently: duplicate the generated first latent as a synthetic
anchor, decode five slices, and retain the last 16 RGB frames. No true return
anchor or adjacent RGB/latent chunk enters the decoder. Validation reconstruction
calibration uses exactly this same approximation.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def precision(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()


def cpu_noise(ids, delay, chunk, seed=2026):
    """Stable under batching/order; identical across arms and history modes."""
    values = []
    for row_id in ids:
        pair_id = row_id.rsplit('_v', 1)[0]
        key = f'{seed}|{pair_id}|{delay}|{chunk}'.encode()
        number = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63 - 1)
        values.append(torch.randn((16, 4, 24, 24), generator=torch.Generator(device='cpu').manual_seed(number)))
    return torch.stack(values)


@torch.inference_mode()
def sample_chunk(model, boundary, state, noise, phase, steps=32):
    if steps != 32:
        raise ValueError('The preregistered sampler uses exactly 32 Euler steps')
    x = noise.to(boundary.device).float()
    for step in range(steps):
        t = torch.full((len(x),), step / steps, device=x.device)
        with precision(x.device):
            velocity = model.predict(boundary, x, t, state, phase)
        x = x + velocity.float() / steps
    return x


class IndependentDecoder:
    def __init__(self, root, device):
        from diffusers import AutoencoderKLWan
        self.device = torch.device(device)
        self.vae = AutoencoderKLWan.from_pretrained(
            Path(root) / 'models/wan-vae', torch_dtype=torch.float32,
            local_files_only=True).eval().requires_grad_(False).to(device)
        self.mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
        self.std = torch.tensor(self.vae.config.latents_std, device=device).view(1, 16, 1, 1, 1)

    @torch.inference_mode()
    def __call__(self, chunks):
        chunks = chunks.to(self.device).float()
        anchored = torch.cat((chunks[:, :, :1], chunks), dim=2)
        rgb = self.vae.decode(anchored * self.std + self.mean).sample
        return rgb[:, :, -16:].add(1).div(2).clamp(0, 1).permute(0, 2, 3, 4, 1).cpu().numpy()


def _rgb(frames):
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError('Expected [T,H,W,3] RGB frames')
    frames = frames.astype(np.float32)
    return frames / 255 if frames.max() > 1.5 else frames


def object_mask(frames):
    frames = _rgb(frames)
    # Shapes use saturated colors; neutral background/grid are excluded.
    maximum, minimum = frames.max(-1), frames.min(-1)
    return ((maximum - minimum) > 0.22) & (maximum > 0.35)


class PreparedTemplates(tuple):
    """Ordered metric-only masks, areas and RGB means; no RGB clips retained."""


def prepare_templates(templates):
    masks = {str(key): object_mask(value) for key, value in templates.items()}
    prepared = []
    for key, template in templates.items():
        key = str(key)
        reference = _rgb(template)
        target_mask = masks[key]
        target_area = max(1, int(target_mask.sum()))
        target_color = reference[target_mask].mean(0) if target_mask.any() else np.zeros(3)
        prepared.append((key, target_mask, target_area, target_color))
    return PreparedTemplates(prepared)


def score_frames(frames, templates, expected_attr=None, calibration=None):
    """Classify visible RGB with explicit missing/fuzzy rejection.

    `templates` maps all 12 attribute identifiers to known-scene RGB return
    renders. Labels are used only by this metric, never by the model/sampler.
    Color and shape accuracies include presence failures in their denominator.
    """
    frames = _rgb(frames)
    if not isinstance(templates, PreparedTemplates):
        templates = prepare_templates(templates)
    observed = object_mask(frames)
    settings = {'min_area_ratio': 0.20, 'max_area_ratio': 2.5,
                'min_iou': 0.35, 'max_color_distance': 0.45}
    if calibration:
        settings.update(calibration)
    best, diagnostics = None, {}
    for key, target_mask, target_area, target_color in templates:
        # Restrict color estimates to the known object's region, while IoU
        # penalizes saturated clutter anywhere in the frame.
        intersection = (observed & target_mask).sum()
        union = (observed | target_mask).sum()
        iou = float(intersection / max(1, union))
        ratio = float(observed.sum() / target_area)
        valid = observed & target_mask
        color = frames[valid].mean(0) if valid.any() else np.zeros(3)
        distance = float(np.linalg.norm(color - target_color))
        diagnostics[key] = {'iou': iou, 'area_ratio': ratio, 'color_distance': distance}
        merit = iou - distance
        if best is None or merit > best[0]:
            best = merit, key
    if best is None:
        raise ValueError('All 12 templates are required')
    candidate = best[1]
    detail = diagnostics[candidate]
    present = settings['min_area_ratio'] <= detail['area_ratio'] <= settings['max_area_ratio']
    accepted = present and detail['iou'] >= settings['min_iou'] and detail['color_distance'] <= settings['max_color_distance']
    predicted = candidate if accepted else None
    result = {'presence': bool(present), 'predicted_attr': predicted,
              'candidate_attr': candidate, 'visible_mask_iou': detail['iou'],
              **detail, 'accepted': bool(accepted)}
    if expected_attr is not None:
        expected = str(expected_attr)
        # Callers attach a mapping for color/shape; full accuracy is generic.
        result['expected_attr'] = expected
        result['full_attr_accuracy'] = float(predicted == expected)
        result['visible_mask_iou'] = diagnostics[expected]['iou']
    return result


def balanced_subset(rows, pairs):
    """Round-robin metadata strata, retain complete counterfactual pairs."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['pair_id']].append(row)
    buckets = defaultdict(list)
    for pair_id, members in sorted(groups.items()):
        signature = (members[0]['mode'], tuple(sorted(str(row['final_attr']) for row in members)))
        buckets[signature].append(pair_id)
    chosen = []
    while len(chosen) < min(pairs, len(groups)):
        for signature in sorted(buckets):
            if buckets[signature] and len(chosen) < pairs:
                chosen.append(buckets[signature].pop(0))
    return [row for pair_id in chosen for row in sorted(groups[pair_id], key=lambda row: row['id'])]


def aggregate(cases):
    keys = ('presence', 'color_accuracy', 'shape_accuracy', 'full_attr_accuracy', 'visible_mask_iou')
    result = {}
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case['arm'], case['mode'], case['history'], case['delay'], case['intervention'])].append(case)
    for group, members in grouped.items():
        result['|'.join(map(str, group))] = {'n': len(members), **{
            key: float(np.mean([item[key] for item in members])) for key in keys}}
    return result


@lru_cache(maxsize=64)
def _scene_templates(scene_key):
    from data import render_chunk
    row = {'scene': json.loads(scene_key)}
    return prepare_templates({str(attr): render_chunk(row, attr, kind='return') for attr in range(12)})


def row_templates(row):
    return _scene_templates(json.dumps(row['scene'], sort_keys=True))


def completed_arm(output, seed, arm, config, checkpoint_metadata, rows, interventions):
    """Reuse only complete, scientifically matching arm results."""
    case_path = output / f'cases_seed{seed}_{arm}.json'
    summary_path = output / f'summary_seed{seed}_{arm}.json'
    if not case_path.exists() and not summary_path.exists():
        return None
    if not case_path.exists() or not summary_path.exists():
        raise ValueError(f'Incomplete resume artifacts for seed={seed} arm={arm}; both cases and summary are required')
    cases = json.loads(case_path.read_text(encoding='utf-8'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    expected_metadata = json.loads(json.dumps(
        {**config, 'seed': seed, 'arm': arm, 'checkpoint_metadata': checkpoint_metadata}))
    mismatches = [key for key, value in expected_metadata.items() if key not in summary or summary[key] != value]
    if mismatches:
        raise ValueError(f'Resume mismatch for seed={seed} arm={arm}: {mismatches}')
    expected = set()
    for row in rows:
        for history in ('truth_history', 'closed_loop'):
            conditions = ('none', 'zero_state', 'swap_state') if interventions and arm == 'memory' and history == 'truth_history' else ('none',)
            for delay in config['delays']:
                for condition in conditions:
                    expected.add((seed, arm, row['id'], row['pair_id'], row['mode'], history, delay, condition, str(row['final_attr'])))
    actual = [(case['seed'], case['arm'], case['id'], case['pair_id'], case['mode'], case['history'],
               case['delay'], case['intervention'], str(case['expected_attr'])) for case in cases]
    if len(actual) != len(expected) or set(actual) != expected or aggregate(cases) != summary.get('metrics'):
        raise ValueError(f'Resume cases are incomplete or inconsistent for seed={seed} arm={arm}')
    return cases


def metrics_for(frames, row):
    result = score_frames(frames, row_templates(row), row['final_attr'])
    predicted = result['predicted_attr']
    expected = int(row['final_attr'])
    result['color_accuracy'] = float(predicted is not None and int(predicted) // 3 == expected // 3)
    result['shape_accuracy'] = float(predicted is not None and int(predicted) % 3 == expected % 3)
    return result


@torch.inference_mode()
def calibrate(root, decoder, batch_size=8, pairs=24):
    """Validate fixed metric thresholds on held-out VAE reconstructions + blanks.

    This is a pass/fail calibration gate; evaluation never adjusts thresholds
    against test outcomes. A failed gate stops evaluation and is saved for audit.
    """
    from data import load_dataset, episode_latents
    rows = balanced_subset(load_dataset(root, 'val'), pairs)
    positives, negative_presence = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        latents = [episode_latents(root, row) for row in batch]
        decoded = decoder(torch.stack([value['return'] for value in latents]))
        blank = decoder(torch.stack([value['hidden'] for value in latents]))
        for row, frames, hidden in zip(batch, decoded, blank):
            positives.append({'id': row['id'], **metrics_for(frames, row)})
            negative_presence.append(score_frames(hidden, row_templates(row))['presence'])
    seen = sorted({int(row['final_attr']) for row in rows})
    accuracy = float(np.mean([case['full_attr_accuracy'] for case in positives])) if positives else 0.
    presence = float(np.mean([case['presence'] for case in positives])) if positives else 0.
    false_positive = float(np.mean(negative_presence)) if negative_presence else 1.
    return {'passed': len(seen) == 12 and accuracy >= .90 and presence >= .95 and false_positive <= .05,
            'n_positive': len(positives), 'n_blank': len(negative_presence), 'attributes': seen,
            'full_attr_accuracy': accuracy, 'presence': presence,
            'blank_false_positive_rate': false_positive, 'cases': positives,
            'thresholds': {'min_area_ratio': .20, 'max_area_ratio': 2.5, 'min_iou': .35,
                           'max_color_distance': .45},
            'decode_anchor': 'duplicate_generated_first_latent_slice; decode then retain last16 frames'}


@torch.inference_mode()
def evaluate_arm(model, root, rows, decoder, seed, arm, delays=(1, 3, 6), batch_size=8,
                 interventions=True, output=None, render_ids=(), save_latents=False):
    from data import episode_latents
    if batch_size < 2 or batch_size % 2:
        raise ValueError('An even batch-size >=2 preserves complete paired interventions')
    device = next(model.parameters()).device
    cases = []
    for mode in ('keep', 'update'):
        selected = [row for row in rows if row['mode'] == mode]
        for delay in delays:
            for start in range(0, len(selected), batch_size):
                batch = selected[start:start + batch_size]
                latent = [episode_latents(root, row, delay) for row in batch]
                ids = [row['id'] for row in batch]
                hidden = torch.stack([value['hidden'] for value in latent]).to(device).float()
                # External cue observations precede both evaluation paths.
                with precision(device):
                    initial = None
                    for cue_index in range(len(latent[0]['cues'])):
                        cue = torch.stack([value['cues'][cue_index] for value in latent]).to(device).float()
                        initial = model.observe(cue, initial)
                    initial_boundary = cue[:, :, -1:].half().clone()
                teacher_state = None
                for history in ('truth_history', 'closed_loop'):
                    state, boundary = initial.clone(), initial_boundary.clone()
                    for chunk in range(delay):
                        if history == 'truth_history':
                            current = hidden
                        else:
                            current = sample_chunk(model, boundary, state,
                                                   cpu_noise(ids, delay, chunk), phase=0)
                        with precision(device):
                            state = model.observe(current, state)
                        boundary = current[:, :, -1:].half().clone()
                    if history == 'truth_history':
                        teacher_state = state.clone()
                    conditions = [('none', state, boundary)]
                    if interventions and arm == 'memory' and history == 'truth_history':
                        if len(batch) % 2 or any(batch[i]['pair_id'] != batch[i ^ 1]['pair_id'] for i in range(len(batch))):
                            raise ValueError('Donor intervention requires adjacent scene-paired variants')
                        donor_indices = torch.tensor([i ^ 1 for i in range(len(batch))], device=device)
                        conditions.extend([('zero_state', torch.zeros_like(teacher_state), boundary),
                                           ('swap_state', teacher_state[donor_indices], boundary)])
                    noise = cpu_noise(ids, delay, delay)
                    for intervention, return_state, return_boundary in conditions:
                        generated = sample_chunk(model, return_boundary, return_state, noise, phase=1)
                        rgb = decoder(generated)
                        for i, (row, frames) in enumerate(zip(batch, rgb)):
                            result = {'seed': seed, 'arm': arm, 'id': row['id'],
                                      'pair_id': row['pair_id'], 'mode': mode, 'history': history,
                                      'delay': delay, 'intervention': intervention, **metrics_for(frames, row)}
                            if intervention == 'swap_state':
                                donor = batch[i ^ 1]
                                predicted = result['predicted_attr']
                                result.update(donor_id=donor['id'], donor_attr=donor['final_attr'],
                                              donor_attr_accuracy=float(predicted == str(donor['final_attr'])))
                            cases.append(result)
                            if output is not None and seed == 42 and row['id'] in render_ids:
                                stem = f"{row['id']}_{arm}_{history}_d{delay}_{intervention}"
                                sample_dir = Path(output) / 'samples'
                                sample_dir.mkdir(parents=True, exist_ok=True)
                                Image.fromarray((frames[-1] * 255).round().astype(np.uint8)).save(sample_dir / f'{stem}.png')
                                if save_latents:
                                    torch.save(generated[i].half().cpu(), sample_dir / f'{stem}.pt')
                    del generated, rgb
                print(f'eval seed={seed} arm={arm} mode={mode} delay={delay} rows={start+len(batch)}/{len(selected)}', flush=True)
    return cases


def main():
    from data import load_dataset
    from model import ROOT, ARMS, load_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--arms', choices=ARMS, nargs='+', default=list(ARMS))
    parser.add_argument('--delays', type=int, nargs='+', default=[1, 3, 6])
    parser.add_argument('--pairs', type=int, default=60, help='Metadata-selected pairs; 0 means entire split')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--no-interventions', action='store_true')
    parser.add_argument('--save-latents', action='store_true')
    parser.add_argument('--calibration-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip completed arms only when saved configuration, checkpoint metadata and cases match')
    args = parser.parse_args()
    if any(delay not in (1, 3, 6) for delay in args.delays):
        parser.error('Evaluation delays are 1, 3, 6')
    if args.pairs < 0:
        parser.error('--pairs must be >=0')
    torch.set_num_threads(4)
    output = args.output or args.root / 'results/delayed_memory' / args.split
    output.mkdir(parents=True, exist_ok=True)
    all_rows = load_dataset(args.root, args.split)
    total_pairs = len({row['pair_id'] for row in all_rows})
    rows = balanced_subset(all_rows, args.pairs or total_pairs)
    write_json(output / 'evaluation_manifest.json', rows)
    decoder = IndependentDecoder(args.root, args.device)
    calibration = calibrate(args.root, decoder, args.batch_size)
    write_json(output / 'classifier_calibration_val.json', calibration)
    if not calibration['passed']:
        raise RuntimeError(f'VAE reconstruction classifier calibration failed; inspect {output / "classifier_calibration_val.json"}')
    if args.calibration_only:
        return
    # Six scene pairs are fixed before looking at model outputs.
    render_ids = {row['id'] for row in balanced_subset(rows, 6)}
    config = {'split': args.split, 'subset_pairs': len(rows)//2, 'split_total_pairs': total_pairs,
              'subset_rule': 'deterministic round-robin mode/final-attribute-pair strata; complete pairs',
              'full_split': len(rows)//2 == total_pairs, 'seeds': args.seeds, 'arms': args.arms,
              'delays': args.delays, 'euler_steps': 32, 'noise_seed': 2026,
              'noise_rule': 'SHA256(seed,pair_id,delay,chunk); CPU normal; shared by paired variants and all arms/history modes',
              'state_kib': 8, 'boundary_kib': 18,
              'decode_anchor': 'duplicate generated first latent slice; last16 decoded frames',
              'render_ids': sorted(render_ids)}
    write_json(output / 'configuration.json', config)
    for seed in args.seeds:
        checkpoint_path = args.root / 'runs' / f'delayed_memory_seed{seed}' / 'final.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        checkpoint_metadata = {key: checkpoint[key] for key in ('step', 'config') if key in checkpoint}
        del checkpoint
        seed_cases = []
        for arm in args.arms:
            if args.resume:
                existing = completed_arm(output, seed, arm, config, checkpoint_metadata, rows,
                                         not args.no_interventions)
                if existing is not None:
                    seed_cases.extend(existing)
                    print(f'resume seed={seed} arm={arm}: reused {len(existing)} completed cases', flush=True)
                    continue
            started = time.perf_counter()
            model = load_model(args.root, seed, arm, args.device)
            cases = evaluate_arm(model, args.root, rows, decoder, seed, arm, args.delays,
                                 args.batch_size, not args.no_interventions, output, render_ids,
                                 args.save_latents)
            elapsed = time.perf_counter() - started
            write_json(output / f'cases_seed{seed}_{arm}.json', cases)
            write_json(output / f'summary_seed{seed}_{arm}.json',
                       {**config, 'seed': seed, 'arm': arm, 'checkpoint': str(checkpoint_path),
                        'checkpoint_metadata': checkpoint_metadata, 'wall_seconds': elapsed,
                        'metrics': aggregate(cases),
                        'swap_donor_response': float(np.mean([case['donor_attr_accuracy'] for case in cases
                            if case['intervention'] == 'swap_state'])) if arm == 'memory' and not args.no_interventions else None})
            seed_cases.extend(cases)
            del model
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
        write_json(output / f'summary_seed{seed}.json', {**config, 'seed': seed,
                   'checkpoint_metadata': checkpoint_metadata, 'metrics': aggregate(seed_cases)})


if __name__ == '__main__':
    main()
