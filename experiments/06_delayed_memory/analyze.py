"""Summarize frozen delayed-memory evaluation, paired uncertainty and resources.

Every training seed is a separate replicate. Within-seed paired bootstrap draws
scene pairs after averaging their counterfactual variants. Cross-seed mean/SD
is reported separately and never treats repeated seeds as independent videos.
"""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ('presence', 'color_accuracy', 'shape_accuracy', 'full_attr_accuracy', 'visible_mask_iou')
ARMS = ('recent', 'first', 'memory')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()),
            'sd': float(values.std(ddof=1)) if len(values) > 1 else None,
            'n_seeds': len(values)}


def paired_bootstrap(pair_values, repeats=5000, seed=420006):
    values = np.asarray(pair_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError('Bootstrap requires nonempty one-dimensional scene-pair differences')
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(repeats, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'ci95_low': float(np.quantile(draws, .025)),
            'ci95_high': float(np.quantile(draws, .975)), 'n_pairs': len(values),
            'bootstrap_repeats': repeats}


def load_cases(source):
    cases, sources, unique = [], [], set()
    # An uncompressed file wins if the same logical file also has a gzip copy.
    paths = {path.name[:-3]: path for path in source.glob('cases_seed*.json.gz')}
    paths.update({path.name: path for path in source.glob('cases_seed*.json')})
    index_path = source / 'archive_index.json'
    if index_path.exists():
        active = set(json.loads(index_path.read_text(encoding='utf-8'))['logical_files'])
        paths = {name: path for name, path in paths.items() if name in active}
    for _, path in sorted(paths.items()):
        content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        rows = json.loads(content.decode('utf-8'))
        for row in rows:
            key = tuple(row[name] for name in ('seed', 'arm', 'id', 'mode', 'history', 'delay', 'intervention'))
            if key in unique:
                raise ValueError(f'Duplicate evaluation case {key} in {path}')
            unique.add(key)
            cases.append(row)
        sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'uncompressed_sha256': hashlib.sha256(content).hexdigest(),
                        'n_cases': len(rows)})
    if not cases:
        raise FileNotFoundError(f'No cases_seed*.json evaluation files found in {source}')
    return cases, sources


def archive_cases(sources, config, output):
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    logical_files = []
    for source in sources:
        path = Path(source['path'])
        content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
        logical_name = path.name[:-3] if path.suffix == '.gz' else path.name
        target = raw / (logical_name + '.gz')
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
        temporary.replace(target)
        source['bundled_path'] = str(target.relative_to(output))
        logical_files.append(logical_name)
    if config is not None:
        write_json(raw / 'configuration.json', config)
    write_json(raw / 'archive_index.json', {'logical_files': sorted(logical_files),
                                           'encoding': 'UTF-8 JSON, gzip; original case bytes preserved'})


def training_report(root, output):
    previous_path = output / 'training.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    result = {}
    for seed in (42, 43, 44):
        path = root / 'runs' / f'delayed_memory_seed{seed}' / 'metrics.jsonl'
        last = None
        if path.exists():
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():
                        last = line
        if last is not None:
            result[str(seed)] = {'available': True, 'source': str(path.resolve()),
                                 'last_metrics_record': json.loads(last)}
        elif str(seed) in previous:
            result[str(seed)] = previous[str(seed)]
        else:
            result[str(seed)] = {'available': False, 'expected_path': str(path.resolve())}
    write_json(previous_path, result)
    return result


def dataset_report(root, output):
    previous_path = output / 'dataset.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    result = {}
    for name in ('summary', 'metadata'):
        path = root / 'cache/delayed_memory' / (name + '.json')
        if path.exists():
            result[name] = {'available': True, 'source': str(path.resolve()),
                            'data': json.loads(path.read_text(encoding='utf-8'))}
        elif name in previous:
            result[name] = previous[name]
        else:
            result[name] = {'available': False, 'expected_path': str(path.resolve())}
    write_json(previous_path, result)
    return result


def absolute_metrics(cases):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in cases:
        if row['intervention'] == 'none':
            group = tuple(row[key] for key in ('arm', 'mode', 'history', 'delay'))
            grouped[group][int(row['seed'])].append(row)
    reports, csv_rows = [], []
    for group, by_seed in sorted(grouped.items()):
        identity = dict(zip(('arm', 'mode', 'history', 'delay'), group))
        for metric in METRICS:
            seed_values = {str(seed): float(np.mean([float(row[metric]) for row in rows]))
                           for seed, rows in sorted(by_seed.items())}
            counts = {str(seed): len(rows) for seed, rows in sorted(by_seed.items())}
            item = {**identity, 'metric': metric, **mean_sd(list(seed_values.values())),
                    'seed_values': seed_values, 'n_cases_per_seed': counts}
            reports.append(item)
            csv_rows.append({'kind': 'absolute_seed_mean', **identity, 'metric': metric,
                             **mean_sd(list(seed_values.values())),
                             'seed_values': json.dumps(seed_values), 'n_cases_per_seed': json.dumps(counts)})
    return reports, csv_rows


def paired_comparisons(cases, repeats=5000):
    normal = {tuple(row[key] for key in ('seed', 'id', 'mode', 'history', 'delay', 'arm')): row
              for row in cases if row['intervention'] == 'none'}
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    unmatched = defaultdict(int)
    for key, memory in sorted(normal.items()):
        seed, case_id, mode, history, delay, arm = key
        if arm != 'memory':
            continue
        for baseline in ('recent', 'first'):
            other = normal.get((seed, case_id, mode, history, delay, baseline))
            if other is None:
                unmatched[baseline] += 1
                continue
            if other['pair_id'] != memory['pair_id']:
                raise ValueError(f'Paired case scene mismatch: {case_id}')
            for metric in METRICS:
                grouped[(baseline, mode, history, delay, metric)][seed][memory['pair_id']].append(
                    float(memory[metric]) - float(other[metric]))
    reports, csv_rows = [], []
    for group, by_seed in sorted(grouped.items()):
        identity = dict(zip(('baseline', 'mode', 'history', 'delay', 'metric'), group))
        seed_reports = {}
        for seed, pairs in sorted(by_seed.items()):
            incomplete = [pair_id for pair_id, values in pairs.items() if len(values) != 2]
            if incomplete:
                raise ValueError(f'Bootstrap requires two matched variants per pair; incomplete={incomplete[:3]}')
            bootstrap = paired_bootstrap([np.mean(values) for _, values in sorted(pairs.items())],
                                         repeats, seed=420006 + int(seed))
            seed_reports[str(seed)] = bootstrap
            csv_rows.append({'kind': 'paired_within_seed_bootstrap', **identity, 'seed': seed, **bootstrap})
        across = mean_sd([report['mean'] for report in seed_reports.values()])
        reports.append({**identity, 'direction': 'memory minus baseline',
                        'seed_results': seed_reports, 'across_training_seeds': across})
        csv_rows.append({'kind': 'paired_across_seed_mean', **identity, **across})
    return reports, csv_rows, dict(unmatched)


def donor_responses(cases):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in cases:
        if row['intervention'] == 'swap_state':
            grouped[(row['mode'], row['history'], row['delay'])][row['seed']].append(row)
    result = []
    for group, by_seed in sorted(grouped.items()):
        identity = dict(zip(('mode', 'history', 'delay'), group))
        for metric in ('donor_attr_accuracy', 'full_attr_accuracy', 'presence'):
            seed_values = {str(seed): float(np.mean([float(row[metric]) for row in rows]))
                           for seed, rows in sorted(by_seed.items())}
            result.append({**identity, 'metric': metric, **mean_sd(list(seed_values.values())),
                           'seed_values': seed_values,
                           'n_cases_per_seed': {str(seed): len(rows) for seed, rows in sorted(by_seed.items())}})
    return result


def plot_comparison(reports, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, constrained_layout=True)
    colors = dict(zip(ARMS, ('#65758b', '#db9132', '#137d70')))
    for ax, (mode, history) in zip(axes.flat, ((m, h) for m in ('keep', 'update')
                                               for h in ('truth_history', 'closed_loop'))):
        for arm in ARMS:
            points = sorted([row for row in reports if row['arm'] == arm and row['mode'] == mode
                             and row['history'] == history and row['metric'] == 'full_attr_accuracy'],
                            key=lambda row: row['delay'])
            if points:
                ax.errorbar([row['delay'] for row in points], [row['mean'] for row in points],
                            yerr=[row['sd'] or 0. for row in points], color=colors[arm], marker='o',
                            capsize=3, label=arm)
        ax.set(title=f'{mode} / {history.replace("_", " ")}', xlabel='Hidden chunks', ylim=(-.03, 1.03),
               xticks=(1, 3, 6), ylabel='Full attribute accuracy')
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    figure.suptitle('Return generation: training-seed means; error bars = seed SD')
    figure.savefig(output / 'comparison.png', dpi=180)
    plt.close(figure)


def resource_report(path, output):
    if not path.exists():
        bundled = output / 'resources.json'
        if bundled.exists():
            return json.loads(bundled.read_text(encoding='utf-8'))
        return {'available': False, 'expected_path': str(path.resolve())}
    raw = json.loads(path.read_text(encoding='utf-8'))
    metadata_path = path.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else None
    result = {'available': True, 'source': str(path.resolve()), 'arms': raw, 'metadata': metadata}
    write_json(output / 'resources.json', result)
    records = []
    for arm, values in raw.items():
        row = {'arm': arm}
        for name, value in values.items():
            if isinstance(value, dict):
                row.update({name + '_' + key: json.dumps(item) if isinstance(item, list) else item
                            for key, item in value.items()})
            else:
                row[name] = json.dumps(value) if isinstance(value, list) else value
        records.append(row)
    write_csv(output / 'resources.csv', records)
    if not records:
        return result
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms = [arm for arm in ARMS if arm in raw]
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    series = [('cache_bytes', 'Persistent state + boundary', 'KiB', 1024),
              ('later_sample_median_ms', 'Later chunk sampling (32 steps)', 'ms', 1),
              ('hidden_update_median_ms', 'Hidden-chunk state update', 'ms', 1),
              ('peak_generator_allocated_bytes', 'Peak generator allocation', 'GiB', 1024**3)]
    for ax, (key, title, unit, scale) in zip(axes, series):
        selected = [arm for arm in arms if key in raw[arm]]
        values = []
        for arm in selected:
            value = raw[arm][key]
            if isinstance(value, list):
                if len(value) != 1:
                    raise ValueError(f'Resource cache size changed across cases: {arm} {value}')
                value = value[0]
            values.append(value / scale)
        ax.bar(selected, values, color=['#65758b', '#db9132', '#137d70'][:len(selected)])
        ax.set(title=title, ylabel=unit)
        ax.grid(axis='y', alpha=.2)
    figure.suptitle('Measured resources: batch 1, one loaded arm; VAE measured separately')
    figure.savefig(output / 'resources.png', dpi=180)
    plt.close(figure)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', type=Path, help='Default ROOT/results/delayed_memory/test')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--resources', type=Path, help='Default ROOT/runs/delayed_memory_seed42/resources/summary.json')
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    source = args.source or args.root / 'results/delayed_memory/test'
    if args.source is None and not any(source.glob('cases_seed*.json*')):
        source = args.output / 'raw'
    resource_path = args.resources or args.root / 'runs/delayed_memory_seed42/resources/summary.json'
    cases, sources = load_cases(source)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = source / 'configuration.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else None
    archive_cases(sources, config, args.output)
    absolute, csv_rows = absolute_metrics(cases)
    paired, paired_rows, unmatched = paired_comparisons(cases, args.bootstrap)
    donors = donor_responses(cases)
    seeds = sorted({int(row['seed']) for row in cases})
    arms = sorted({row['arm'] for row in cases})
    summary = {'configuration': config, 'observed_training_seeds': seeds, 'observed_arms': arms,
               'complete_three_seed_arm_grid': all(any(row['seed'] == seed and row['arm'] == arm for row in cases)
                                                   for seed in (42, 43, 44) for arm in ARMS),
               'n_loaded_cases': len(cases), 'absolute_metrics': absolute,
               'paired_comparisons': paired, 'swap_donor_responses': donors,
               'unmatched_memory_cases': unmatched,
               'uncertainty': 'Sample SD across training-seed means (ddof=1). Within each seed, bootstrap complete scene pairs after averaging the two counterfactual variant differences. No CI pools seeds as independent cases.',
               'resources': resource_report(resource_path, args.output),
               'training': training_report(args.root, args.output),
               'dataset': dataset_report(args.root, args.output)}
    write_json(args.output / 'summary.json', summary)
    donor_rows = [{'kind': 'swap_donor_response', **{key: value for key, value in item.items()
                   if key not in ('seed_values', 'n_cases_per_seed')},
                   'seed_values': json.dumps(item['seed_values']),
                   'n_cases_per_seed': json.dumps(item['n_cases_per_seed'])} for item in donors]
    write_csv(args.output / 'summary.csv', csv_rows + paired_rows + donor_rows)
    plot_comparison(absolute, args.output)
    write_json(args.output / 'sources.json', {'evaluation_case_files': sources, 'configuration': config})
    print(json.dumps({'output': str(args.output.resolve()), 'n_cases': len(cases), 'seeds': seeds,
                      'arms': arms, 'resources_available': summary['resources']['available']}))


if __name__ == '__main__':
    main()
