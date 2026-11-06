"""Index anchor+A+B+C windows with inherited splits, then cache frozen Wan latents.

Run from the repository: python experiments/06_recurrent_state/prepare_cache.py
Use --index-only to inspect the actual eligible population without loading the VAE.
"""

import argparse
import gc
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CACHE_NAME = "ucf6_12fps_192_abc"
SOURCE_CACHE_NAME = "ucf6_12fps_192"
FPS, FRAMES, RESOLUTION = 12, 49, 192
SPLITS = ("train", "val", "test")


def load_helpers(root):
    """Share the existing decoder, center crop and atomic writers verbatim."""
    spec = importlib.util.spec_from_file_location(
        "q6_original_cache_helpers", root / "scripts/prepare_cache.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inherited_videos(source):
    videos, group_splits = {}, {}
    for row in source:
        video, split, group = row["video"], row["split"], row["group"]
        if split not in SPLITS:
            raise ValueError(f"Unknown inherited split: {split}")
        assignment = {"split": split, "group": group}
        if video in videos and videos[video] != assignment:
            raise ValueError(f"Source video crosses splits/groups: {video}")
        if group in group_splits and group_splits[group] != split:
            raise ValueError(f"Source group crosses splits: {group}")
        videos[video], group_splits[group] = assignment, split
    return videos


def validate_records(records, videos):
    ids, starts = set(), {}
    for row in records:
        video = row["video"]
        if video not in videos or any(row[k] != videos[video][k] for k in ("split", "group")):
            raise ValueError(f"Manifest does not inherit source assignment: {video}")
        indices = row["indices"]
        timestamps = row["timestamps"]
        if (len(indices) != FRAMES or len(set(indices)) != FRAMES
                or any(a >= b for a, b in zip(indices, indices[1:]))
                or len(timestamps) != FRAMES
                or any(a > b for a, b in zip(timestamps, timestamps[1:]))):
            raise ValueError(f"Invalid ordered 49-frame window: {row['id']}")
        start = row["start_seconds"]
        if start < 0 or row["id"] != f"{Path(video).stem}_s{round(start * FPS):05d}":
            raise ValueError(f"Invalid start/id: {row['id']}")
        if row["id"] in ids or start in starts.setdefault(video, set()):
            raise ValueError(f"Duplicate window: {row['id']}")
        ids.add(row["id"])
        starts[video].add(start)
        if len(starts[video]) > 4:
            raise ValueError(f"More than four windows: {video}")


def make_manifest(root, cache, helpers, videos, progress_every=10):
    path = cache / "manifest.json"
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
        validate_records(records, videos)
        return records
    records = []
    for number, (video, assignment) in enumerate(sorted(videos.items()), 1):
        times, _ = helpers.read_video(root / "data/UCF-101" / video)
        # 49 observations span exactly four seconds; never wrap or concatenate.
        if len(times) and times[-1] >= (FRAMES - 1) / FPS:
            last_start = int(np.floor(times[-1] * FPS)) - (FRAMES - 1)
            starts = np.unique(np.linspace(0, last_start, min(4, last_start + 1)).astype(int))
            for start in starts:
                requested = (start + np.arange(FRAMES)) / FPS
                right = np.searchsorted(times, requested).clip(0, len(times) - 1)
                left = np.maximum(right - 1, 0)
                indices = np.where(requested - times[left] <= times[right] - requested, left, right)
                if len(np.unique(indices)) != FRAMES:
                    continue
                records.append({
                    "id": f"{Path(video).stem}_s{start:05d}",
                    "video": video, **assignment,
                    "indices": indices.tolist(),
                    "timestamps": times[indices].tolist(),
                    "start_seconds": float(start / FPS),
                })
        if number % progress_every == 0 or number == len(videos):
            print(f"index {number}/{len(videos)} videos; {len(records)} eligible windows", flush=True)
    records.sort(key=lambda row: (SPLITS.index(row["split"]), row["id"]))
    validate_records(records, videos)
    helpers.write_json(path, records)
    return records


def summarize(records, videos, cache):
    result = {"source_videos": len(videos), "windows": len(records), "splits": {}}
    used = {row["video"] for row in records}
    result["videos_without_eligible_window"] = sorted(set(videos) - used)
    result["vae_cached_windows"] = sum((cache / "vae" / (row["id"] + ".pt")).exists() for row in records)
    for split in SPLITS:
        rows = [r for r in records if r["split"] == split]
        split_videos = {r["video"] for r in rows}
        classes = sorted({v.split("/")[0] for v, a in videos.items() if a["split"] == split})
        result["splits"][split] = {
            "windows": len(rows), "videos": len(split_videos),
            "groups": len({r["group"] for r in rows}),
            "classes": {c: {
                "windows": sum(r["video"].split("/")[0] == c for r in rows),
                "videos": sum(v.split("/")[0] == c for v in split_videos),
                "groups": len({r["group"] for r in rows if r["video"].split("/")[0] == c}),
            } for c in classes},
        }
    return result


@torch.inference_mode()
def cache_vae(root, cache, records, helpers, progress_every=10):
    output = cache / "vae"
    output.mkdir(exist_ok=True)
    pending = [row for row in records if not (output / (row["id"] + ".pt")).exists()]
    if not pending:
        print("vae: all selected windows already cached", flush=True)
        return
    from diffusers import AutoencoderKLWan

    model = AutoencoderKLWan.from_pretrained(
        root / "models/wan-vae", torch_dtype=torch.float32, local_files_only=True,
    ).eval().requires_grad_(False).cuda()
    mean = torch.tensor(model.config.latents_mean, dtype=torch.float32, device="cuda").view(1, 16, 1, 1, 1)
    std = torch.tensor(model.config.latents_std, dtype=torch.float32, device="cuda").view(1, 16, 1, 1, 1)
    started = time.perf_counter()
    # Wan's temporal encoder is causal: encoding this ordered clip preserves
    # anchor / A / B / C slices without feeding future targets into a prefix.
    for index, (row, video, crop) in enumerate(helpers.iter_clips(root, pending), 1):
        video = F.interpolate(video, (RESOLUTION, RESOLUTION), mode="bilinear", align_corners=False, antialias=True)
        video = video.permute(1, 0, 2, 3).unsqueeze(0).cuda().mul_(2).sub_(1)
        latent = (model.encode(video).latent_dist.mode() - mean) / std
        if latent.shape != (1, 16, 13, 24, 24):
            raise ValueError(f"Unexpected Wan latent shape: {tuple(latent.shape)}")
        helpers.save_tensor(output / (row["id"] + ".pt"), {
            "latent": latent[0].cpu().half(), "crop_tlhw": crop,
        })
        if index % progress_every == 0 or index == len(pending):
            elapsed = time.perf_counter() - started
            print(f"vae {index}/{len(pending)} {row['id']} elapsed={elapsed:.1f}s ({elapsed / index:.2f}s/window)", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be positive")
    started = time.perf_counter()
    root = args.root.resolve()
    helpers = load_helpers(root)
    source_path = root / "cache" / SOURCE_CACHE_NAME / "manifest.json"
    videos = inherited_videos(json.loads(source_path.read_text(encoding="utf-8")))
    cache = root / "cache" / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)
    records = make_manifest(root, cache, helpers, videos, args.progress_every)
    index_seconds = time.perf_counter() - started
    helpers.write_json(cache / "metadata.json", {
        "source_manifest": str(source_path.relative_to(root)),
        "split_policy": "inherit video and group assignments; no re-splitting",
        "fps": FPS, "frames": FRAMES, "max_windows_per_video": 4,
        "frame_sampling": "nearest timestamps; unique ordered source frames; no wrap/padding",
        "frame_blocks": {"anchor": [0, 1], "A": [1, 17], "B": [17, 33], "C": [33, 49]},
        "latent_slices": {"anchor": [0, 1], "A": [1, 5], "B": [5, 9], "C": [9, 13]},
        "slice_convention": "half-open [start, end)",
        "latent_shape": [16, 13, 24, 24],
        "crop": "center square (shared scripts.prepare_cache.iter_clips)",
        "vae_resolution": RESOLUTION, "vae_dtype": "float32", "storage_dtype": "float16",
        "pixel_normalization": "RGB / 255, bilinear antialiased resize, then 2*x-1",
        "vae_normalization": "(posterior.mode - latents_mean) / latents_std",
        "causality": "Wan causal temporal encode; chunk targets are subsequent latent slices",
    })
    summary = summarize(records, videos, cache)
    summary["index_seconds"] = index_seconds
    helpers.write_json(cache / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.index_only:
        cache_vae(root, cache, records, helpers, args.progress_every)
    summary = summarize(records, videos, cache)
    summary.update(index_seconds=index_seconds, elapsed_seconds=time.perf_counter() - started,
                   index_only=args.index_only)
    helpers.write_json(cache / "summary.json", summary)
    print(f"DONE windows={len(records)} vae_cached={summary['vae_cached_windows']} "
          f"elapsed={summary['elapsed_seconds']:.1f}s cache={cache}", flush=True)


if __name__ == "__main__":
    main()
