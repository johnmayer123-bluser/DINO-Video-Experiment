"""Shared data, paired evaluation and reporting for history-semantics experiments."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from denoiser_model import Denoiser
from train_semantic_head import macro_average, save_checkpoint
from metrics import NOISE_LEVELS, summarize_pair

ARMS = ("N", "S", "SD", "L")
PAIRS = {"N_vs_SD": ("N", "SD"), "N_vs_S": ("N", "S"),
         "N_vs_L": ("N", "L"), "S_vs_SD": ("S", "SD")}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def setup():
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(root, split):
    cache = root / "cache/ucf6_12fps_192"
    records = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    rows = [row for row in records if row["split"] == split]
    if not rows:
        raise ValueError(f"Empty split: {split}")
    latents, semantics = [], []
    for row in rows:
        name = row["id"] + ".pt"
        z = torch.load(cache / "vae" / name, map_location="cpu", weights_only=True)["latent"][:, 1:9]
        h = torch.load(cache / "jepa" / name, map_location="cpu", weights_only=True)["phi_a"]
        if z.shape != (16, 8, 24, 24) or h.shape not in ((64, 1024), (4, 4, 4, 1024)):
            raise ValueError(f"Bad cache shape: {row['id']}")
        if not torch.isfinite(z).all() or not torch.isfinite(h).all():
            raise ValueError(f"Nonfinite cache: {row['id']}")
        latents.append(z.clone())
        semantics.append(F.normalize(h.float().reshape(64, 1024), dim=-1).half())
    return rows, torch.stack(latents).pin_memory(), torch.stack(semantics).pin_memory()


def evaluation_noise(shape, start, repeat, seed):
    return torch.stack([torch.randn(shape[1:], generator=torch.Generator().manual_seed(
        seed + repeat * 1_000_003 + start + offset)) for offset in range(shape[0])])


def apply_dropout(semantics, dropped):
    return torch.where(dropped[:, None, None], torch.zeros_like(semantics), semantics)


def compact_metrics(rows, errors):
    return {"mse": {k: macro_average(v.mean(1), rows)[0] for k, v in errors.items()},
            "by_t": {str(t): {k: macro_average(v[:, i], rows)[0] for k, v in errors.items()}
                     for i, t in enumerate(NOISE_LEVELS)}}


@torch.inference_mode()
def evaluate(models, latent, semantics, split, repeats=4, batch_size=16):
    device = next(next(iter(models.values())).parameters()).device
    modes = {name: model.training for name, model in models.items()}
    all_errors = {name: [] for name in models}
    try:
        for model in models.values():
            model.eval()
        for repeat in range(repeats):
            errors = {name: [] for name in models}
            for start in range(0, len(latent), batch_size):
                z = latent[start:start+batch_size].to(device, non_blocking=True).float()
                history, future = z[:, :, :4], z[:, :, 4:]
                h = semantics[start:start+batch_size].to(device, non_blocking=True)
                zero_z, zero_h = torch.zeros_like(history), torch.zeros_like(h)
                conditions = {"N": (zero_z, zero_h), "S": (zero_z, h),
                              "SD": (zero_z, h), "L": (history, zero_h)}
                noise = evaluation_noise(future.shape, start, repeat, 10042 if split == "val" else 20042).to(device)
                target = future-noise
                batch_errors = {name: [] for name in models}
                for level in NOISE_LEVELS:
                    t = torch.full((len(z),), level, device=device)
                    noisy = (1-level)*noise + level*future
                    for name, model in models.items():
                        past, tokens = conditions[name]
                        precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                        with precision:
                            output = model(past, noisy, t, tokens)
                        batch_errors[name].append((output.float()-target).square().flatten(1).mean(1).cpu())
                for name in models:
                    errors[name].append(torch.stack(batch_errors[name], dim=1))
            for name in models:
                all_errors[name].append(torch.cat(errors[name]).numpy())
        repeated = {name: np.stack(values) for name, values in all_errors.items()}
        return {name: values.mean(0) for name, values in repeated.items()}, repeated
    finally:
        for name, model in models.items():
            model.train(modes[name])


def save_evaluation(run, models, rows, z, h, split, step, seed, batch_size=16):
    errors, repeated = evaluate(models, z, h, split, batch_size=batch_size)
    stem = f"step{step:05d}_{split}"
    np.savez_compressed(run / f"errors_{stem}.npz", **errors,
                        **{name+"_repeats": value for name, value in repeated.items()},
                        ids=np.asarray([r["id"] for r in rows]), videos=np.asarray([r["video"] for r in rows]),
                        groups=np.asarray([r["group"] for r in rows]), noise_levels=np.asarray(NOISE_LEVELS),
                        noise_seed=10042 if split == "val" else 20042, noise_repeats=4)
    summary = {"seed": seed, "step": step, "split": split, "metrics": compact_metrics(rows, errors),
               "noise_repeats": 4, "noise_seed": 10042 if split == "val" else 20042,
               "pair_direction": "a_vs_b: improvement=1-mse_b/mse_a",
               "aggregation": "noise mean -> window -> video -> source group -> class macro"}
    for name, (a, b) in PAIRS.items():
        summary[name] = summarize_pair(rows, errors[a], errors[b], bootstraps=5000)
        summary[name]["scope"] = f"{split} source-group sampling uncertainty for one training seed; not training-run variation"
    write_json(run / f"{split}_manifest.json", rows)
    write_json(run / f"summary_{stem}.json", summary)
    print(json.dumps({"seed": seed, "split": split, **{name: summary[name]["primary"] for name in PAIRS}}), flush=True)
