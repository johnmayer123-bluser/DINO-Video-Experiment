"""Run preparation, three training seeds, generation evaluation and resource profiling."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--wait-for-cache', action='store_true',
                        help='Wait for existing cache job and logs/delay_cache.exit')
    parser.add_argument('--start-at', choices=('cache', 'evaluation', 'samples', 'resources', 'analysis'),
                        default='cache', help='Start at this stage, skipping all earlier stages')
    args = parser.parse_args()
    root = args.root.resolve()
    logs = root / 'logs'
    logs.mkdir(exist_ok=True)
    prefix = [sys.executable, '-u']
    experiment = 'experiments/06_delayed_memory/'
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    stages = []
    order = ('cache', 'calibration', 'train_seed42', 'train_seed43', 'train_seed44',
             'evaluation', 'samples', 'resources', 'analysis')
    start_index = order.index(args.start_at)
    previous_status = logs / 'delay_pipeline_status.json'
    if previous_status.exists():
        previous_text = previous_status.read_text(encoding='utf-8')
        (logs / 'delay_pipeline_previous.json').write_text(previous_text, encoding='utf-8')
        previous = json.loads(previous_text)
        stages.extend(item for item in previous.get('finished_stages', [])
                      if item.get('stage') in order and order.index(item['stage']) < start_index)

    def status(stage, state, **extra):
        record = {'stage': stage, 'state': state, 'runner_pid': os.getpid(),
                  'updated_utc': datetime.now(timezone.utc).isoformat(),
                  'finished_stages': stages, **extra}
        (logs / 'delay_pipeline_status.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record), flush=True)

    def run(stage, command):
        if order.index(stage) < start_index:
            return
        status(stage, 'running', command=command)
        started = time.perf_counter()
        with (logs / f'delay_{stage}.log').open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        (logs / f'delay_{stage}.exit').write_text(str(result.returncode), encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{stage} exited {result.returncode}; see logs/delay_{stage}.log')
        stages.append({'stage': stage, 'seconds': time.perf_counter() - started})

    try:
        if args.start_at != 'cache':
            pass
        elif args.wait_for_cache:
            status('cache', 'waiting_for_existing_job')
            marker = logs / 'delay_cache.exit'
            while not marker.exists():
                time.sleep(10)
            if int(marker.read_text().strip()) != 0:
                raise RuntimeError('Cache preparation failed; see logs/delay_cache.log')
            stages.append({'stage': 'cache', 'existing_job': True})
        else:
            run('cache', prefix + [experiment + 'data.py', '--root', str(root)])
        run('calibration', prefix + [experiment + 'evaluate.py', '--root', str(root),
            '--split', 'val', '--pairs', '24', '--calibration-only',
            '--output', 'runs/delayed_memory_calibration'])
        for seed in (42, 43, 44):
            if start_index > order.index(f'train_seed{seed}'):
                continue
            command = prefix + [experiment + 'train.py', '--root', str(root),
                                '--seed', str(seed), '--steps', '6000']
            if (root / 'runs' / f'delayed_memory_seed{seed}' / 'last.pt').exists():
                command.append('--resume')
            run(f'train_seed{seed}', command)
        run('evaluation', prefix + [experiment + 'evaluate.py', '--root', str(root),
            '--split', 'test', '--pairs', '60', '--seeds', '42', '43', '44',
            '--delays', '1', '3', '6', '--resume'])
        run('samples', prefix + [experiment + 'sample.py', '--root', str(root),
                                '--seed', '42', '--delays', '3', '6'])
        run('resources', prefix + [experiment + 'resources.py', '--root', str(root),
                                  '--seed', '42', '--decode'])
        run('analysis', prefix + [experiment + 'analyze.py', '--root', str(root)])
        status('all', 'completed')
    except Exception as error:
        status('pipeline', 'failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
