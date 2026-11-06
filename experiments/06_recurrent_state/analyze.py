"""Source-group paired analysis of recurrent history and old-state interventions."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '01_history_semantics'))
from metrics import NOISE_LEVELS, _aggregate, _pair

SEEDS = (42, 43, 44)
ARMS = ('recent_B', 'recurrent_B', 'recent_C', 'recurrent_C', 'reset_C', 'shuffled_C')
REGIONS = ('overall', 'first', 'later')
PAIRS = {'recent_C_vs_recurrent_C': ('recent_C', 'recurrent_C'),
         'reset_C_vs_recurrent_C': ('reset_C', 'recurrent_C'),
         'shuffled_C_vs_recurrent_C': ('shuffled_C', 'recurrent_C'),
         'recent_B_vs_recurrent_B': ('recent_B', 'recurrent_B')}
LABELS = {'recent_C_vs_recurrent_C': 'Recurrent vs recent-only history',
          'reset_C_vs_recurrent_C': 'Correct vs reset old state',
          'shuffled_C_vs_recurrent_C': 'Correct vs shuffled old state'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def describe(values):
    x = np.asarray(values, dtype=float)
    return {'mean': float(x.mean()), 'sample_sd': float(x.std(ddof=1)),
            'min': float(x.min()), 'max': float(x.max())}


def compact_rows(rows):
    return [{key: row[key] for key in ('id', 'video', 'group')} for row in rows]


def import_runs(run_root, out, split):
    out.mkdir(parents=True, exist_ok=True)
    reference = None
    for seed in SEEDS:
        run = run_root / f'recurrent_state_seed{seed}'
        rows = compact_rows(read(run / f'{split}_manifest.json'))
        if reference is not None and rows != reference:
            raise ValueError('Training seeds have different evaluation manifests')
        reference = rows
        source, target = run / f'errors_step06000_{split}.npz', out / f'seed{seed}.npz'
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    (out / f'{split}_manifest.json').write_text(json.dumps(reference, indent=2), encoding='utf-8')


def validate_donors(data, rows):
    donors = data['donor_indices']
    if donors.shape != (len(rows),) or donors.dtype.kind not in 'iu':
        raise ValueError('Invalid donor index array')
    if np.any(donors < -1) or np.any(donors >= len(rows)):
        raise ValueError('Donor index out of bounds')
    if not np.array_equal(data['donor_ids'], [rows[i]['id'] if i >= 0 else '' for i in donors]):
        raise ValueError('Donor ids and indices disagree')
    if int(data['donor_seed']) != 2026:
        raise ValueError('Unexpected donor seed')
    if data['donor_valid'].dtype.kind != 'b' or not np.array_equal(data['donor_valid'], donors >= 0):
        raise ValueError('Invalid donor eligibility mask')
    # Reconstruct selection without importing GPU model dependencies.
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]['id'])
    rng = np.random.default_rng(2026)
    for i in ordered:
        row = rows[i]
        candidates = [j for j in ordered if rows[j]['video'].split('/')[0] == row['video'].split('/')[0]
                      and rows[j]['video'] != row['video'] and rows[j]['group'] != row['group']]
        expected = candidates[int(rng.integers(len(candidates)))] if candidates else -1
        if donors[i] != expected:
            raise ValueError('Donor mapping differs from fixed metadata-only selection')
    return donors.copy()


def analyze(out, split):
    rows = read(out / f'{split}_manifest.json')
    runs, records, reference_donors = [], [], None
    for seed in SEEDS:
        with np.load(out / f'seed{seed}.npz', allow_pickle=False) as data:
            if not np.array_equal(data['ids'], [r['id'] for r in rows]):
                raise ValueError('NPZ ids do not match the manifest')
            if not np.array_equal(data['noise_levels'], NOISE_LEVELS):
                raise ValueError('Unexpected noise levels')
            if (int(data['noise_seed']) != (20042 if split == 'test' else 10042)
                    or int(data['c_noise_offset']) != 500003 or int(data['noise_repeats']) != 4):
                raise ValueError('Unexpected evaluation-noise protocol')
            donors = validate_donors(data, rows)
            if reference_donors is not None and not np.array_equal(reference_donors, donors):
                raise ValueError('Training seeds have different donor mappings')
            reference_donors = donors
            valid = donors >= 0
            eligible_rows = [row for row, keep in zip(rows, valid) if keep]
            if not eligible_rows:
                raise ValueError('No eligible shuffled-state cohort in this split')
            errors = {region: {} for region in REGIONS}
            for arm in ARMS:
                repeated, frames = data[arm + '_repeats'], data[arm + '_by_frame_repeats']
                if repeated.shape != (4, len(rows), 5) or frames.shape != (4, len(rows), 5, 4):
                    raise ValueError(f'Unexpected error shapes for {arm}')
                if not np.isfinite(repeated).all() or not np.isfinite(frames).all():
                    raise ValueError(f'Nonfinite errors for {arm}')
                if not np.array_equal(data[arm], repeated.mean(0)):
                    raise ValueError(f'Repeat means disagree for {arm}')
                if not np.allclose(repeated, frames.mean(-1), rtol=1e-6, atol=1e-7):
                    raise ValueError(f'Frame and whole-chunk errors disagree for {arm}')
                errors['overall'][arm] = data[arm].copy()
                errors['first'][arm] = frames.mean(0)[..., 0]
                errors['later'][arm] = frames.mean(0)[..., 1:].mean(-1)
        run = {'seed': seed, 'regions': {}}
        for region, values in errors.items():
            full_values = {arm: value for arm, value in values.items() if arm != 'shuffled_C'}
            arms, point, samples, counts = _aggregate(rows, full_values, 5000, 2026)
            reports = {key: _pair(arms, point, samples, a, b) for key, (a, b) in PAIRS.items()
                       if key != 'shuffled_C_vs_recurrent_C'}
            shuffle_arms, shuffle_point, shuffle_samples, shuffle_counts = _aggregate(
                eligible_rows, {arm: values[arm][valid] for arm in ('shuffled_C', 'recurrent_C')}, 5000, 2026)
            reports['shuffled_C_vs_recurrent_C'] = _pair(
                shuffle_arms, shuffle_point, shuffle_samples, 'shuffled_C', 'recurrent_C')
            run['regions'][region] = {'mse': {arm: float(point[i].mean()) for i, arm in enumerate(arms)},
                                      'by_t': {arm: point[i].tolist() for i, arm in enumerate(arms)},
                                      'pairs': reports}
            # Only the shuffled arm uses the eligible cohort in the arm summaries.
            run['regions'][region]['mse']['shuffled_C'] = float(shuffle_point[0].mean())
            run['regions'][region]['by_t']['shuffled_C'] = shuffle_point[0].tolist()
            run['regions'][region]['shuffled_eligible'] = {
                'counts': shuffle_counts, 'recurrent_C_mse': float(shuffle_point[1].mean()),
                'recurrent_C_by_t': shuffle_point[1].tolist()}
            for key, report in reports.items():
                p = report['primary']
                lo, hi = p['ci95']['relative_improvement']
                records.append({'seed': seed, 'region': region, 'comparison': key,
                                'cohort': 'shuffled_eligible' if key.startswith('shuffled') else 'full',
                                **{f'cohort_{k}': v for k, v in
                                   (shuffle_counts if key.startswith('shuffled') else counts).items()},
                                'role': ('secondary_B' if key.startswith('recent_B') else
                                         'primary' if key.startswith('recent_C') and region == 'overall' else
                                         'intervention' if key.startswith(('reset', 'shuffled')) else 'secondary_C'),
                                'mse_reference': p['mse_a'], 'mse_treatment': p['mse_b'],
                                'gain_percent': 100 * p['relative_improvement'],
                                'ci95_low_percent': 100 * lo, 'ci95_high_percent': 100 * hi})
        runs.append(run)
    summary = {'split': split, 'steps': 6000, 'seeds': list(SEEDS), 'counts': counts,
               'primary': 'recent_C_vs_recurrent_C overall',
               'pair_direction': 'a_vs_b: 1 - mse_b/mse_a; positive favors recurrent/correct old state',
               'regions': {'overall': 'all four target latent slices', 'first': 'first target latent slice',
                           'later': 'mean of remaining three target latent slices'},
               'statistics': 'Four-noise mean -> window -> video -> source group -> class macro. '
                             'Within-class paired source-group bootstrap: 5000 draws, seed 2026. '
                             'Three-training-seed mean and sample SD are descriptive; C is primary, B secondary.',
               'donor_protocol': 'Fixed seed 2026; id-sorted metadata-only selection within split/class, '
                                 'different video and source group; identical mapping for all training seeds. '
                                 'Ineligible windows have index -1, empty donor id, donor_valid=false; '
                                 'their shuffled NPZ values are self-state placeholders excluded from analysis.',
               'cohorts': {'full': counts, 'shuffled_eligible': shuffle_counts,
                           'excluded_windows': int((~valid).sum()),
                           'full_classes': sorted({r['video'].split('/')[0] for r in rows}),
                           'shuffled_eligible_classes': sorted({r['video'].split('/')[0] for r in eligible_rows}),
                           'arm_rule': 'shuffled_C MSE/by_t and shuffled comparison use eligible rows; '
                                       'all other arm MSE/by_t and comparisons use full rows'},
               'shuffled_eligible_recurrent_C_mse': {
                   region: describe([r['regions'][region]['shuffled_eligible']['recurrent_C_mse'] for r in runs])
                   for region in REGIONS},
               'mse': {region: {arm: describe([r['regions'][region]['mse'][arm] for r in runs])
                                for arm in ARMS} for region in REGIONS},
               'gain_percent': {region: {key: describe([100 * r['regions'][region]['pairs'][key]['primary']['relative_improvement']
                                                       for r in runs]) for key in PAIRS} for region in REGIONS},
               'per_seed': records, 'runs': runs}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot(runs, out)
    return summary


def plot(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, key in zip(axes, LABELS):
        for j, run in enumerate(runs):
            for i, region in enumerate(REGIONS):
                p = run['regions'][region]['pairs'][key]['primary']
                lo, hi = 100 * np.asarray(p['ci95']['relative_improvement'])
                x = i + (j - 1) * .16
                ax.vlines(x, lo, hi, color=f'C{j}')
                ax.plot(x, 100 * p['relative_improvement'], 'o', color=f'C{j}',
                        label=f"Seed {run['seed']}" if i == 0 else None)
        ax.axhline(0, color='grey', lw=.8)
        ax.set(xticks=range(3), xticklabels=['All C slices', 'First C slice', 'Later C slices'],
               title=LABELS[key] + ('\n(eligible same-class donor subset)' if key.startswith('shuffled') else '\n(full evaluation cohort)'),
               ylabel='Velocity MSE reduction (%)')
        ax.legend(title='95% source-group CI', fontsize=8, title_fontsize=9)
    fig.suptitle('Recurrent history and old-state interventions: target chunk C, step 6000')
    fig.savefig(out / 'comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    args = parser.parse_args()
    out = args.results_dir or (args.run_root / f'recurrent_state_results_{args.split}' if args.run_root else HERE / 'results')
    if args.run_root:
        import_runs(args.run_root, out, args.split)
    result = analyze(out, args.split)
    print(json.dumps({'mse': result['mse'], 'gain_percent': result['gain_percent']}, indent=2))


if __name__ == '__main__':
    main()
