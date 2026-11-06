"""Run the seed42 recursive closed-loop experiment through analysis."""
import argparse
import subprocess
import sys
import time
from common import WORK_ROOT, write_json


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    logs = WORK_ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    status = {'seed': 42, 'steps_per_arm': 6000, 'arms': ['F','L','R'],
              'status': 'running', 'completed': [], 'started_unix': time.time()}
    stages = [
        ('data', ['data.py', '--device','cuda','--batch-size','8']),
        ('train', ['train.py','--seed','42','--steps','6000','--arms','F','L','R','--resume']),
        ('evaluate', ['evaluate.py','--pairs','60','--seed','42','--batch-pairs','8',
                      '--delays','1','4','8','16','32','--preview-pairs','2']),
        ('resources', ['resources.py','--seed','42','--repeats','3']),
        ('analyze', ['analyze.py']),
    ]
    for stage, args in stages:
        status.update(stage=stage,stage_started_unix=time.time())
        write_json(logs/'status.json',status)
        print(f'Starting {stage}',flush=True)
        with (logs/f'{stage}.log').open('a',encoding='utf-8') as log:
            result = subprocess.run([sys.executable,'-u',*args],cwd=WORK_ROOT,
                                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='failed',returncode=result.returncode,ended_unix=time.time())
            write_json(logs/'status.json',status)
            raise SystemExit(result.returncode)
        status['completed'].append(stage)
        print(f'Completed {stage}',flush=True)
    status.update(status='complete',ended_unix=time.time())
    write_json(logs/'status.json',status)


if __name__ == '__main__':
    main()
