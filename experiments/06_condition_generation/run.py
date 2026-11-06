"""Run seed 42 preparation, paired training, generation evaluation, and timing."""
import argparse
import subprocess
import sys
import time
from common import WORK_ROOT, write_json


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    logs = WORK_ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    status = {'seed': 42, 'steps_per_arm': 6000, 'arms': ['N', 'C'],
              'status': 'running', 'completed': [], 'started_unix': time.time()}
    stages = [
        ('data', ['data.py', '--device', 'cuda', '--batch-size', '4', '--preview-pairs', '2']),
        ('train', ['train.py', '--seed', '42', '--steps', '6000', '--arms', 'N', 'C']),
        ('evaluate', ['evaluate.py', '--seed', '42', '--pairs', '60', '--device', 'cuda', '--preview-pairs', '4']),
        ('benchmark', ['benchmark.py']),
    ]
    for name, args in stages:
        status.update(stage=name, stage_started_unix=time.time())
        write_json(logs / 'status.json', status)
        print(f'Starting {name}', flush=True)
        with (logs / f'{name}.log').open('a', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-u', *args], cwd=WORK_ROOT,
                                    stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='failed', returncode=result.returncode, ended_unix=time.time())
            write_json(logs / 'status.json', status)
            raise SystemExit(result.returncode)
        status['completed'].append(name)
        print(f'Completed {name}', flush=True)
    status.update(status='complete', ended_unix=time.time())
    write_json(logs / 'status.json', status)


if __name__ == '__main__':
    main()
