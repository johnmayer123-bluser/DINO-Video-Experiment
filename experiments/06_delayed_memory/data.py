"""Procedural paired delayed visual memory data and independent frozen Wan encoding.

Prepare: python experiments/06_delayed_memory/data.py --root . --device cuda
Inspect without torch/models: add --index-only --preview-pairs 2.
The manifest contains two variant rows per scene pair. Each tensors/<pair_id>.pt
contains one shared hidden chunk and variants mapping to cues and return tensors.
No raw videos are stored. Evaluation calls episode_latents(..., delay=1/3/6);
repeat its single hidden tensor delay times, then generate return. Only cues are
observations (including the second, external observation in update episodes).
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw


CACHE_NAME = "delayed_memory"
VERSION = 3
FRAMES, RESOLUTION, FPS = 16, 192, 12
LATENT_SHAPE = (16, 4, 24, 24)
SPLITS = ("train", "val", "test")
EVAL_DELAYS = (1, 3, 6)
COLORS = ((240, 40, 40), (35, 220, 70), (45, 85, 245), (245, 210, 25))
SHAPES = ("circle", "square", "triangle")
ATTRIBUTES = [{"id": c * 3 + s, "color": list(color), "shape": shape}
              for c, color in enumerate(COLORS) for s, shape in enumerate(SHAPES)]


def cache_dir(root):
    return Path(root).resolve() / "cache" / CACHE_NAME


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_manifest(counts, seed=420006):
    """Attributes use a separate RNG; scenes never depend on attribute or mode."""
    rows = []
    for split_index, split in enumerate(SPLITS):
        count = counts[split]
        attr_rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, 901]))
        initial_rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, 902]))
        delay_rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, 903]))
        delays = delay_rng.permutation(1 + np.arange(count) % 3)
        # Permute scenes independently before assigning balanced labels/modes.
        assignment = attr_rng.permutation(count)
        color_shape_order = attr_rng.permutation(12)
        for scene_index in range(count):
            rank = int(assignment[scene_index])
            mode = "keep" if rank % 2 == 0 else "update"
            scene_seed = seed + split_index * 10_000_000 + scene_index
            rng = np.random.default_rng(np.random.SeedSequence([scene_seed, 187]))
            scene = {
                "seed": int(scene_seed), "background": int(rng.integers(102, 147)),
                "occluder_gray": int(rng.integers(46, 78)),
                "occluder_x0": int(rng.integers(70, 75)),
                "occluder_x1": int(rng.integers(118, 123)),
                "occluder_y0": 20, "occluder_y1": 172,
                "object_y": int(rng.integers(72, 121)),
                "radius": int(rng.integers(11, 16)),
                "cue_start_x": int(rng.integers(30, 39)),
                "return_end_x": int(rng.integers(155, 165)),
            }
            pair_id = f"{split}_{scene_index:05d}"
            final_attrs = [int(color_shape_order[((rank // 2) + 6 * v) % 12]) for v in (0, 1)]
            # Update pairs have identical first observations. Only the second
            # observed cue differs, and the common initial label predicts neither
            # variant through a fixed mapping.
            pair_initial = None
            if mode == "update":
                alternatives = [a for a in range(12) if a not in final_attrs]
                pair_initial = int(initial_rng.choice(alternatives))
            for variant in (0, 1):
                final_attr = final_attrs[variant]
                initial_attr = pair_initial if mode == "update" else final_attr
                rows.append({"id": f"{pair_id}_v{variant}", "pair_id": pair_id,
                             "split": split, "variant": variant, "mode": mode,
                             "delay": int(delays[scene_index]),
                             "initial_attr": initial_attr, "final_attr": final_attr,
                             "scene": scene})
    return rows


def render_chunk(row, attr, kind="cue"):
    """Return deterministic RGB uint8 [16,192,192,3]; occluder drawn last."""
    if kind not in ("cue", "return", "hidden"):
        raise ValueError(f"Unknown chunk kind: {kind}")
    scene = row["scene"]
    middle = (scene["occluder_x0"] + scene["occluder_x1"]) / 2
    start, end = ((scene["cue_start_x"], middle) if kind == "cue" else
                  (middle, scene["return_end_x"]))
    frames = []
    for x in np.linspace(start, end, FRAMES):
        image = Image.new("RGB", (RESOLUTION, RESOLUTION), (scene["background"],) * 3)
        draw = ImageDraw.Draw(image)
        if kind != "hidden":
            color, shape = COLORS[int(attr) // 3], SHAPES[int(attr) % 3]
            x, y, r = int(round(x)), scene["object_y"], scene["radius"]
            if shape == "circle":
                draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
            elif shape == "square":
                draw.rectangle((x-r, y-r, x+r, y+r), fill=color)
            else:
                draw.polygon(((x, y-r), (x-r, y+r), (x+r, y+r)), fill=color)
        draw.rectangle((scene["occluder_x0"], scene["occluder_y0"],
                        scene["occluder_x1"], scene["occluder_y1"]),
                       fill=(scene["occluder_gray"],) * 3)
        frames.append(np.asarray(image))
    return np.stack(frames)


def load_dataset(root, split):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    records = json.loads((cache_dir(root) / "manifest.json").read_text(encoding="utf-8"))
    return [row for row in records if row["split"] == split]


@lru_cache(maxsize=8)
def _load_pair(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


@lru_cache(maxsize=None)
def _shared_hidden(path):
    # Preserve object identity across variants and evaluation delays, even if the
    # much larger cue/return pair cache has been evicted.
    return _load_pair(path)["hidden"]


def episode_latents(root, row, delay=None):
    """CPU FP16 tensors; delay never expands or copies the shared hidden tensor."""
    selected_delay = row["delay"] if delay is None else delay
    if int(selected_delay) != selected_delay or selected_delay < 1:
        raise ValueError("delay must be a positive integer")
    path = str(cache_dir(root) / "tensors" / (row["pair_id"] + ".pt"))
    pair = _load_pair(path)
    variant = pair["variants"][str(row["variant"])]
    return {"cues": variant["cues"], "hidden": _shared_hidden(path),
            "return": variant["return"]}


def preview(cache, rows, number):
    """Contact sheets show observed cues, blank delay, return; paired rows adjacent."""
    output = cache / "previews"
    output.mkdir(exist_ok=True)
    for row in rows[:number * 2]:
        attrs = [row["initial_attr"]] + ([row["final_attr"]] if row["mode"] == "update" else [])
        clips = [render_chunk(row, a, "cue") for a in attrs]
        clips += [render_chunk(row, 0, "hidden"), render_chunk(row, row["final_attr"], "return")]
        sheet = Image.new("RGB", (4 * RESOLUTION, len(clips) * (RESOLUTION + 22)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, clip in enumerate(clips):
            for j, frame in enumerate((0, 5, 10, 15)):
                sheet.paste(Image.fromarray(clip[frame]), (j * RESOLUTION, i * (RESOLUTION+22)+22))
            label = f"observed cue {i+1}" if i < len(attrs) else ("hidden (repeat delay)" if i == len(attrs) else "return target")
            draw.text((5, i * (RESOLUTION+22)+4), label, fill="black")
        sheet.save(output / (row["id"] + ".png"))


def encode_chunks(model, chunks, mean, std, device, batch_size):
    """Each encode call gets complete independent 17-frame examples, no past cache."""
    import torch
    output = []
    for start in range(0, len(chunks), batch_size):
        pixels = np.stack(chunks[start:start+batch_size])
        pixels = np.concatenate((pixels[:, :1], pixels), axis=1)
        video = torch.from_numpy(pixels).permute(0, 4, 1, 2, 3).to(device=device, dtype=torch.float32)
        video = video.div_(127.5).sub_(1)
        # diffusers AutoencoderKLWan.encode resets internal causal feature caches
        # for every sample/call; explicitly clear before and after for clarity.
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        with torch.inference_mode():
            latent = (model.encode(video).latent_dist.mode() - mean) / std
        if tuple(latent.shape[1:]) != (16, 5, 24, 24):
            raise ValueError(f"Unexpected Wan shape: {tuple(latent.shape)}")
        output.extend(latent[:, :, 1:].cpu().half().unbind(0))
        if hasattr(model, "clear_cache"):
            model.clear_cache()
    return output


def prepare(args):
    started = time.perf_counter()
    counts = dict(zip(SPLITS, (args.train_pairs, args.val_pairs, args.test_pairs)))
    if any(n < 0 for n in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("Pair counts must be nonnegative and total must be positive")
    if args.batch_size < 1 or (args.limit_pairs is not None and args.limit_pairs < 1):
        raise ValueError("batch-size and limit-pairs must be positive")
    cache = cache_dir(args.root)
    cache.mkdir(parents=True, exist_ok=True)
    config = {"version": VERSION, "seed": args.seed, "pairs": counts}
    config_path = cache / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing delayed_memory cache has a different config; use a different root or explicitly remove it")
    write_json(config_path, config)
    rows = make_manifest(counts, args.seed)
    write_json(cache / "manifest.json", rows)
    metadata = {
        **config, "attributes": ATTRIBUTES, "frames": FRAMES, "resolution": RESOLUTION,
        "fps": FPS, "latent_shape": list(LATENT_SHAPE), "storage_dtype": "float16",
        "vae": "models/wan-vae (frozen, local_files_only)", "vae_dtype": "float32",
        "encoding": "independent [duplicate first frame,16 chunk frames], posterior.mode, discard latent time index 0",
        "normalization": "(posterior.mode - config.latents_mean) / config.latents_std",
        "causal_cache": "reset for each independent chunk; never concatenate episode chunks",
        "split_policy": "unique scene seeds by split, both variants stay in pair's split",
        "manifest": "list of variant rows; tensors/<pair_id>.pt stores hidden and variants[str(variant)]={cues,return}",
        "observations": "keep: initial cue; update: initial cue then externally observed changed-attribute cue",
        "targets": "delay copies of shared hidden chunk, then return; cues never targets",
        "phase_flags": {"hidden": 0, "return": 1}, "attribute_labels": "analysis only; never model inputs",
        "train_delays": [1, 2, 3], "eval_delays": list(EVAL_DELAYS),
        "delay_assignment": "balanced scene assignment shuffled by independent purpose903 RNG",
        "update_pair_initial": "one shared label drawn uniformly from the ten labels excluding both final labels",
        "limit_pairs": args.limit_pairs, "index_only": args.index_only,
    }
    write_json(cache / "metadata.json", metadata)
    pairs = [rows[i:i+2] for i in range(0, len(rows), 2)]
    chosen = pairs if args.limit_pairs is None else pairs[:args.limit_pairs]
    tensor_dir = cache / "tensors"
    tensor_dir.mkdir(exist_ok=True)

    def save_summary():
        present = sum((tensor_dir / (p[0]["pair_id"] + ".pt")).exists() for p in pairs)
        summary = {"pairs": len(pairs), "variant_rows": len(rows), "cached_pairs": present,
                   "complete": present == len(pairs), "selected_pairs": len(chosen),
                   "index_only": args.index_only, "limit_pairs": args.limit_pairs,
                   "elapsed_seconds": time.perf_counter() - started}
        write_json(cache / "summary.json", summary)
        return summary

    save_summary()
    if args.preview_pairs:
        preview(cache, rows, args.preview_pairs)
    pending = [p for p in chosen if not (tensor_dir / (p[0]["pair_id"] + ".pt")).exists()]
    if not args.index_only and pending:
        import torch
        from diffusers import AutoencoderKLWan
        device = torch.device(args.device)
        model_started = time.perf_counter()
        model = AutoencoderKLWan.from_pretrained(
            Path(args.root) / "models" / "wan-vae", torch_dtype=torch.float32,
            local_files_only=True).to(device).eval().requires_grad_(False)
        mean = torch.tensor(model.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
        std = torch.tensor(model.config.latents_std, device=device).view(1, 16, 1, 1, 1)
        metadata["vae_load_seconds"] = time.perf_counter() - model_started
        metadata["vae_config_mean"] = list(model.config.latents_mean)
        metadata["vae_config_std"] = list(model.config.latents_std)
        write_json(cache / "metadata.json", metadata)
        for index, pair in enumerate(pending, 1):
            pair_started = time.perf_counter()
            chunks = [render_chunk(pair[0], 0, "hidden")]
            layout = []
            for row in pair:
                attrs = [row["initial_attr"]] + ([row["final_attr"]] if row["mode"] == "update" else [])
                layout.append(len(attrs))
                chunks.extend(render_chunk(row, a, "cue") for a in attrs)
                chunks.append(render_chunk(row, row["final_attr"], "return"))
            latents = encode_chunks(model, chunks, mean, std, device, args.batch_size)
            payload = {"hidden": latents[0], "variants": {}}
            cursor = 1
            for row, ncue in zip(pair, layout):
                payload["variants"][str(row["variant"])] = {
                    "cues": torch.stack(latents[cursor:cursor+ncue]),
                    "return": latents[cursor+ncue]}
                cursor += ncue+1
            path = tensor_dir / (pair[0]["pair_id"] + ".pt")
            temp = path.with_suffix(".pt.tmp")
            torch.save(payload, temp)
            temp.replace(path)
            print(f"cache {index}/{len(pending)} {pair[0]['pair_id']} {time.perf_counter()-pair_started:.2f}s", flush=True)
            if index % 10 == 0:
                save_summary()
        del model
    summary = save_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--train-pairs", type=int, default=1000)
    parser.add_argument("--val-pairs", type=int, default=200)
    parser.add_argument("--test-pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=420006)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1, help="Independent chunks per VAE encode; 1 is memory-safe")
    parser.add_argument("--limit-pairs", type=int, default=None, help="Explicitly incomplete functional cache; full manifest retained")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--preview-pairs", type=int, default=0)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
