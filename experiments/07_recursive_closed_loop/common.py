"""Paths and source reuse for the single-seed recursive closed-loop experiment."""
import importlib.util
import json
import os
from pathlib import Path
import sys

WORK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get('DINO_VIDEO_ROOT', str(WORK_ROOT.parents[1]))).resolve()
Q6_ROOT = Path(os.environ.get('Q6_WORK_ROOT', str(
    WORK_ROOT.parent / '06_condition_generation'))).resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
