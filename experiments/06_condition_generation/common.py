"""Repository-relative paths for the keep-only condition experiment."""
import json
import os
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get('DINO_VIDEO_ROOT', str(WORK_ROOT.parents[1]))).resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
