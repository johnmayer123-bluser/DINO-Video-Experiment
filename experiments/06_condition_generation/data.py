"""Keep-only paired scenes and independent frozen Wan VAE cache.

python data.py --index-only --preview-pairs 2
python data.py --device cuda
Masks and attribute labels are supervision/analysis only, never model inputs.
"""
import argparse
from functools import lru_cache
import json
import time

import numpy as np
from PIL import Image, ImageDraw

from common import REPO_ROOT, WORK_ROOT, write_json

FRAMES, RESOLUTION, FPS = 16, 192, 12
LATENT_SHAPE = (16, 4, 24, 24)
SPLITS = ("train", "val", "test")
COLORS = ((240, 40, 40), (35, 220, 70), (45, 85, 245), (245, 210, 25))
SHAPES = ("circle", "square", "triangle")
ATTRIBUTES = [{"id": c * 3 + s, "color": list(color), "shape": shape}
              for c, color in enumerate(COLORS) for s, shape in enumerate(SHAPES)]
CACHE = WORK_ROOT / "cache"


def make_manifest(counts=None, seed=42):
    counts = counts or dict(zip(SPLITS, (1000, 200, 300)))
    rows = []
    for split_index, split in enumerate(SPLITS):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split_index, 901]))
        # Complete directed comparison cycles keep both variant marginals balanced.
        comparisons = {
            "color": [(c * 3+s, d * 3+s) for s in range(3)
                      for c in range(4) for d in range(4) if c != d],
            "shape": [(c * 3+s, c * 3+t) for c in range(4)
                      for s in range(3) for t in range(3) if s != t],
        }
        assignments = []
        for change in ("color", "shape"):
            n = (counts[split] + (change == "color")) // 2
            choices = comparisons[change]
            # Remainders are spread by independent shuffled full cycles.
            order = []
            while len(order) < n:
                order.extend(int(i) for i in rng.permutation(len(choices)))
            assignments.extend((change, choices[i]) for i in order[:n])
        rng.shuffle(assignments)
        for scene_index, (change, attrs) in enumerate(assignments):
            scene_seed = seed + split_index * 10_000_000 + scene_index
            geometry = np.random.default_rng(np.random.SeedSequence([scene_seed, 187]))
            scene = {
                "seed": int(scene_seed), "background": int(geometry.integers(102, 147)),
                "occluder_gray": int(geometry.integers(46, 78)),
                "occluder_x0": 58, "occluder_x1": 134,
                "occluder_y0": 20, "occluder_y1": 172,
                "object_y": int(geometry.integers(65, 128)),
                "radius": int(geometry.integers(24, 29)),
                "cue_start_x": int(geometry.integers(30, 32)),
                "return_end_x": int(geometry.integers(161, 163)),
            }
            pair_id = f"{split}_{scene_index:05d}"
            for variant, attr in enumerate(attrs):
                rows.append({"id": f"{pair_id}_v{variant}", "pair_id": pair_id,
                             "variant": variant, "split": split, "color_id": attr // 3,
                             "shape_id": attr % 3, "attr": attr, "scene": scene,
                             "pair_change": change})
    return rows


def render_chunk(row, kind="cue", return_mask=False):
    """RGB uint8 [16,192,192,3], optionally visible-object uint8 [16,192,192]."""
    if kind not in ("cue", "return", "blank"):
        raise ValueError(f"Unknown chunk kind: {kind}")
    scene = row["scene"]
    middle = (scene["occluder_x0"] + scene["occluder_x1"]) / 2
    start, end = ((scene["cue_start_x"], middle) if kind == "cue" else
                  (middle, scene["return_end_x"]))
    frames, masks = [], []
    for x in np.linspace(start, end, FRAMES):
        image = Image.new("RGB", (RESOLUTION, RESOLUTION), (scene["background"],) * 3)
        draw = ImageDraw.Draw(image)
        mask_image = Image.new("L", (RESOLUTION, RESOLUTION), 0)
        mask_draw = ImageDraw.Draw(mask_image)
        if kind != "blank":
            x, y, r = int(round(x)), scene["object_y"], scene["radius"]
            color, shape = COLORS[row["color_id"]], SHAPES[row["shape_id"]]
            bounds = (x-r, y-r, x+r, y+r)
            if shape == "circle":
                draw.ellipse(bounds, fill=color)
                mask_draw.ellipse(bounds, fill=1)
            elif shape == "square":
                draw.rectangle(bounds, fill=color)
                mask_draw.rectangle(bounds, fill=1)
            else:
                points = ((x, y-r), (x-r, y+r), (x+r, y+r))
                draw.polygon(points, fill=color)
                mask_draw.polygon(points, fill=1)
        bounds = (scene["occluder_x0"], scene["occluder_y0"],
                  scene["occluder_x1"], scene["occluder_y1"])
        draw.rectangle(bounds, fill=(scene["occluder_gray"],) * 3)
        mask_draw.rectangle(bounds, fill=0)
        frames.append(np.asarray(image))
        masks.append(np.asarray(mask_image))
    rgb = np.stack(frames)
    return (rgb, np.stack(masks)) if return_mask else rgb


def latent_mask(pixel_mask):
    """Union over each 4-frame/8x8 block; dilate spatial support by one cell."""
    if pixel_mask.shape != (16, 192, 192):
        raise ValueError(f"Unexpected mask shape: {pixel_mask.shape}")
    pooled = pixel_mask.reshape(4, 4, 24, 8, 24, 8).max(axis=(1, 3, 5))
    padded = np.pad(pooled, ((0, 0), (1, 1), (1, 1)))
    expanded = np.maximum.reduce([padded[:, dy:dy+24, dx:dx+24]
                                  for dy in range(3) for dx in range(3)])
    return expanded[None].astype(np.float16)


def load_rows(split):
    return [r for r in json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
            if r["split"] == split]


@lru_cache(maxsize=8)
def load_pair(pair_id):
    import torch
    return torch.load(CACHE / "tensors" / f"{pair_id}.pt", map_location="cpu", weights_only=True)


def preview(rows, number):
    directory = CACHE / "previews"
    directory.mkdir(exist_ok=True)
    for row in rows[:2 * number]:
        sheet = Image.new("RGB", (4 * RESOLUTION, 3 * (RESOLUTION+22)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, kind in enumerate(("cue", "blank", "return")):
            rgb, mask = render_chunk(row, kind, True)
            draw.text((5, k * (RESOLUTION+22)+4), f"{row['id']} {kind}", fill="black")
            for col, frame in enumerate((0, 5, 10, 15)):
                sheet.paste(Image.fromarray(rgb[frame]), (col*RESOLUTION, k*(RESOLUTION+22)+22))
        sheet.save(directory / f"{row['id']}.png")
        _, mask = render_chunk(row, "return", True)
        strip = np.concatenate([mask[i]*255 for i in (0, 5, 10, 15)], axis=1)
        Image.fromarray(strip).save(directory / f"{row['id']}_visible_mask.png")


def encode_chunks(model, chunks, mean, std, device, batch_size=1):
    import torch
    output = []
    for start in range(0, len(chunks), batch_size):
        pixels = np.stack(chunks[start:start+batch_size])
        pixels = np.concatenate((pixels[:, :1], pixels), axis=1)
        video = torch.from_numpy(pixels).permute(0, 4, 1, 2, 3).to(device=device, dtype=torch.float32)
        video = video.div_(127.5).sub_(1)
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
    CACHE.mkdir(parents=True, exist_ok=True)
    tensors = CACHE / "tensors"
    tensors.mkdir(exist_ok=True)
    counts = dict(zip(SPLITS, (1000, 200, 300)))
    config = {"version": 1, "seed": 42, "pairs": counts}
    config_path = CACHE / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing cache has a different configuration")
    write_json(config_path, config)
    rows = make_manifest(counts)
    write_json(CACHE / "manifest.json", rows)
    pairs = [rows[i:i+2] for i in range(0, len(rows), 2)]
    chosen = pairs if args.limit_pairs is None else pairs[:args.limit_pairs]
    metadata_path = CACHE / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata.update({**config, "frames": FRAMES, "resolution": RESOLUTION, "fps": FPS,
        "attributes": ATTRIBUTES, "latent_shape": list(LATENT_SHAPE), "storage_dtype": "float16",
        "encoding": "frozen local Wan VAE; independent [repeat first,16 frames], posterior.mode, discard first latent",
        "normalization": "(posterior.mode - config.latents_mean) / config.latents_std",
        "boundary": "last retained latent of independent scene-specific blank chunk [16,1,24,24]",
        "mask": "visible return object only; 4-frame temporal union, 8x8 spatial union, 1-cell spatial dilation; loss only",
        "split_policy": "disjoint scene seeds; paired variants share scene and split; color/shape changes split equally",
        "tensor_fields": {"boundary": [16,1,24,24], "cue": [2,16,4,24,24],
                          "target": [2,16,4,24,24], "mask": [2,1,4,24,24]}})
    if "pixel_mask_mean_area" not in metadata:
        pixel_sum, latent_sum = 0.0, 0.0
        for row in rows:
            _, mask = render_chunk(row, "return", True)
            pixel_sum += float(mask.mean())
            latent_sum += float(latent_mask(mask).mean(dtype=np.float64))
        metadata["pixel_mask_mean_area"] = pixel_sum / len(rows)
        metadata["latent_mask_mean_area"] = latent_sum / len(rows)
        metadata["mask_area_units"] = "fraction of all return spatiotemporal cells, over all manifest rows"
    write_json(metadata_path, metadata)
    if args.preview_pairs:
        preview(rows, args.preview_pairs)
    pending = [p for p in chosen if not (tensors / f"{p[0]['pair_id']}.pt").exists()]
    if pending and not args.index_only:
        import torch
        from diffusers import AutoencoderKLWan
        torch.set_num_threads(4)
        device = torch.device(args.device)
        model = AutoencoderKLWan.from_pretrained(REPO_ROOT / "models" / "wan-vae",
            torch_dtype=torch.float32, local_files_only=True).to(device).eval().requires_grad_(False)
        mean = torch.tensor(model.config.latents_mean, device=device).view(1,16,1,1,1)
        std = torch.tensor(model.config.latents_std, device=device).view(1,16,1,1,1)
        metadata["vae_config_mean"] = list(model.config.latents_mean)
        metadata["vae_config_std"] = list(model.config.latents_std)
        write_json(metadata_path, metadata)
        for i, pair in enumerate(pending, 1):
            clips = [render_chunk(pair[0], "blank")]
            clips.extend(render_chunk(row, "cue") for row in pair)
            return_chunks = [render_chunk(row, "return", True) for row in pair]
            clips.extend(rgb for rgb, _ in return_chunks)
            encoded = encode_chunks(model, clips, mean, std, device, args.batch_size)
            payload = {"boundary": encoded[0][:,-1:].clone(), "cue": torch.stack(encoded[1:3]),
                       "target": torch.stack(encoded[3:5]),
                       "mask": torch.from_numpy(np.stack([latent_mask(m) for _, m in return_chunks]))}
            path = tensors / f"{pair[0]['pair_id']}.pt"
            temporary = path.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            print(f"cache {i}/{len(pending)} {pair[0]['pair_id']}", flush=True)
        del model
    present = sum((tensors / f"{p[0]['pair_id']}.pt").exists() for p in pairs)
    summary = {"pairs": len(pairs), "variant_rows": len(rows), "cached_pairs": present,
               "complete": present == len(pairs), "selected_pairs": len(chosen),
               "index_only": args.index_only, "elapsed_seconds": time.perf_counter()-started}
    write_json(CACHE / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--preview-pairs", type=int, default=2)
    parser.add_argument("--limit-pairs", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_pairs is not None and args.limit_pairs < 1):
        parser.error("batch-size and limit-pairs must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
