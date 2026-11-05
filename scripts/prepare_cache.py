"""Prepare UCF101 chunk pairs, Wan latents, and independently encoded V-JEPA targets."""

import argparse
import gc
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F


CLASSES = (
    "BasketballDunk", "GolfSwing", "LongJump",
    "TennisSwing", "VolleyballSpiking", "Diving",
)
CACHE_NAME = "ucf6_12fps_192"


def read_video(path, rgb=False):
    times, frames = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / float(stream.average_rate)
            times.append(timestamp)
            if rgb:
                frames.append(frame.to_ndarray(format="rgb24"))
    times = np.asarray(times)
    if len(times):
        times -= times[0]
    return times, frames


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_tensor(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def make_manifest(root, cache):
    manifest_path = cache / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    lists = root / "data/ucfTrainTestlist"
    train = [line.split()[0] for line in (lists / "trainlist01.txt").read_text().splitlines()]
    test = (lists / "testlist01.txt").read_text().splitlines()
    train = sorted(name for name in train if name.split("/")[0] in CLASSES)
    test = sorted(name for name in test if name.split("/")[0] in CLASSES)
    group_of = lambda name: name.split("/")[0] + "_" + re.search(r"_g(\d+)_", name).group(1)
    rng = random.Random(42)
    val_groups = set()
    for category in CLASSES:
        groups = sorted({group_of(name) for name in train if name.startswith(category + "/")})
        val_groups.update(rng.sample(groups, max(1, round(0.2 * len(groups)))))
    assert not ({group_of(name) for name in train} & {group_of(name) for name in test})
    records, skipped = [], []
    for number, name in enumerate(train + test, 1):
        split = "test" if name in test else "val" if group_of(name) in val_groups else "train"
        path = root / "data/UCF-101" / name
        times, _ = read_video(path)
        if not len(times) or times[-1] < 32 / 12:
            skipped.append(name)
            continue
        last_start = int(np.floor(times[-1] * 12)) - 32
        starts = np.unique(np.linspace(0, last_start, min(4, last_start + 1)).astype(int))
        for start in starts:
            requested = (start + np.arange(33)) / 12
            right = np.searchsorted(times, requested).clip(0, len(times) - 1)
            left = np.maximum(right - 1, 0)
            indices = np.where(requested - times[left] <= times[right] - requested, left, right)
            if len(np.unique(indices)) < 33:
                continue
            records.append({
                "id": f"{Path(name).stem}_s{start:05d}",
                "video": name,
                "group": group_of(name),
                "split": split,
                "indices": indices.tolist(),
                "timestamps": times[indices].tolist(),
                "start_seconds": float(start / 12),
            })
        if number % 100 == 0:
            print(f"index {number}/{len(train) + len(test)} videos", flush=True)
    order = {"train": 0, "val": 1, "test": 2}
    records.sort(key=lambda row: (order[row["split"]], row["id"]))
    write_json(manifest_path, records)
    write_json(cache / "summary.json", {
        "videos_total": len(train) + len(test),
        "windows": dict(Counter(row["split"] for row in records)),
        "videos_with_windows": {
            split: len({r["video"] for r in records if r["split"] == split}) for split in order
        },
        "skipped_too_short": skipped,
    })
    print(f"manifest: {len(records)} windows, {len(skipped)} short videos skipped", flush=True)
    return records


def iter_clips(root, records):
    previous, frames = None, None
    for row in records:
        if row["video"] != previous:
            _, frames = read_video(root / "data/UCF-101" / row["video"], rgb=True)
            previous = row["video"]
        video = torch.from_numpy(np.stack([frames[i] for i in row["indices"]]))
        height, width = video.shape[1:3]
        size = min(height, width)
        top, left = (height - size) // 2, (width - size) // 2
        video = video[:, top:top + size, left:left + size]
        video = video.permute(0, 3, 1, 2).float().div_(255)
        yield row, video, [top, left, size, size]


@torch.inference_mode()
def cache_vae(root, cache, records):
    from diffusers import AutoencoderKLWan

    output = cache / "vae"
    output.mkdir(exist_ok=True)
    pending = [row for row in records if not (output / (row["id"] + ".pt")).exists()]
    if not pending:
        return
    model = AutoencoderKLWan.from_pretrained(
        root / "models/wan-vae", torch_dtype=torch.float32, local_files_only=True,
    ).eval().requires_grad_(False).cuda()
    mean = torch.tensor(model.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1)
    std = torch.tensor(model.config.latents_std, device="cuda").view(1, 16, 1, 1, 1)
    for index, (row, video, crop) in enumerate(iter_clips(root, pending), 1):
        video = F.interpolate(video, (192, 192), mode="bilinear", align_corners=False, antialias=True)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).cuda().mul_(2).sub_(1)
        latent = (model.encode(video).latent_dist.mode() - mean) / std
        assert latent.shape == (1, 16, 9, 24, 24)
        save_tensor(output / (row["id"] + ".pt"), {
            "latent": latent[0].cpu().half(),
            "crop_tlhw": crop,
        })
        print(f"vae {index}/{len(pending)} {row['id']}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def cache_jepa(root, cache, records):
    sys.path.insert(0, str(root / "third_party/jepa"))
    from src.models.vision_transformer import vit_large

    output = cache / "jepa"
    output.mkdir(exist_ok=True)
    pending = [row for row in records if not (output / (row["id"] + ".pt")).exists()]
    if not pending:
        return
    model = vit_large(img_size=224, patch_size=16, num_frames=16, tubelet_size=2,
                      uniform_power=True, use_sdpa=True)
    # This is the verified official Meta checkpoint, including its EMA teacher.
    checkpoint = torch.load(root / "models/vjepa/vitl16.pth.tar", map_location="cpu",
                            weights_only=False, mmap=True)
    state = {
        key.removeprefix("module.").removeprefix("backbone."): value
        for key, value in checkpoint["target_encoder"].items()
    }
    model.load_state_dict(state, strict=True)
    del state, checkpoint
    model.eval().requires_grad_(False).cuda()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1)
    for index, (row, video, crop) in enumerate(iter_clips(root, pending), 1):
        video = F.interpolate(video, (224, 224), mode="bilinear", align_corners=False, antialias=True)
        features = {}
        for name, start in (("phi_a", 1), ("phi_b", 17)):
            block = video[start:start + 16].permute(1, 0, 2, 3).unsqueeze(0).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model((block - mean) / std)
            assert tokens.shape == (1, 1568, 1024)
            grid = tokens.float().reshape(1, 8, 14, 14, 1024).permute(0, 4, 1, 2, 3)
            features[name] = F.adaptive_avg_pool3d(grid, (4, 4, 4))[0].permute(1, 2, 3, 0).cpu().half()
        features["crop_tlhw"] = crop
        save_tensor(output / (row["id"] + ".pt"), features)
        print(f"jepa {index}/{len(pending)} {row['id']}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=("index", "vae", "jepa", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit encoding to this many indexed windows.")
    args = parser.parse_args()
    root = args.root.resolve()
    cache = root / "cache" / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)
    records = make_manifest(root, cache)
    if args.limit:
        records = records[:args.limit]
    if args.stage in ("vae", "all"):
        cache_vae(root, cache, records)
    if args.stage in ("jepa", "all"):
        cache_jepa(root, cache, records)
    jepa_commit = subprocess.check_output(
        ["git", "-C", str(root / "third_party/jepa"), "rev-parse", "HEAD"], text=True,
    ).strip()
    write_json(cache / "metadata.json", {
        "classes": CLASSES, "official_split": 1, "validation_seed": 42,
        "fps": 12, "frames": 33, "max_windows_per_video": 4,
        "crop": "center square; same field of view for both encoders",
        "vae_resolution": 192, "vae_dtype": "float32",
        "vae_normalization": "(posterior.mode - latents_mean) / latents_std",
        "jepa_resolution": 224, "jepa_state": "target_encoder",
        "jepa_grid": [4, 4, 4, 1024], "jepa_commit": jepa_commit,
        "teacher_blocks": {"A": [1, 17], "B": [17, 33]},
        "latent_slices": {"initial": [0, 1], "A": [1, 5], "B": [5, 9]},
        "torch": torch.__version__,
    })
    print(f"DONE stage={args.stage} selected={len(records)} cache={cache}", flush=True)


if __name__ == "__main__":
    main()
