"""Frozen A-only predictor and paired teacher/predicted-condition evaluation."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/01_history_semantics"))
from common import (Denoiser, apply_dropout, compact_metrics, evaluation_noise,
                    load_split, save_checkpoint, setup, write_json)
from metrics import NOISE_LEVELS, summarize_pair
from train_semantic_head import SemanticHead, macro_average

PAIRS = {"N_vs_T": ("N", "T"), "N_vs_D": ("N", "D"), "N_vs_P": ("N", "P"),
         "T_vs_D": ("T", "D"), "T_vs_P": ("T", "P"), "D_vs_P": ("D", "P")}


def load_predictor(root):
    path = root / "runs/semantic_head_seed42/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["step"] != 2800:
        raise ValueError("Use the fixed semantic_head_seed42 best checkpoint at step 2800")
    model = SemanticHead().cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"])
    return model


@torch.inference_mode()
def predict_semantics(model, latents, batch_size=16):
    # Match scripts/train_denoiser_ab.py exactly: input is A, never B.
    result = []
    for start in range(0, len(latents), batch_size):
        z = latents[start:start+batch_size, :, :4].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(z)
        result.append(F.normalize(predicted.float(), dim=-1).flatten(1, 3).half().cpu())
    return torch.cat(result).pin_memory()


def feature_metrics(rows, teacher, predicted):
    cosine = (F.normalize(teacher.float(), dim=-1)*F.normalize(predicted.float(), dim=-1)).sum(-1).mean(1)
    return {"q_teacher_token_cosine": macro_average(cosine.numpy(), rows)[0],
            "role": "auxiliary feature agreement; not a checkpoint-selection metric"}


@torch.inference_mode()
def evaluate(models, latent, teacher, predicted, split, repeats=4, batch_size=16):
    device = next(next(iter(models.values())).parameters()).device
    modes = {name: model.training for name, model in models.items()}
    repeated = {name: [] for name in models}
    try:
        for model in models.values():
            model.eval()
        for repeat in range(repeats):
            errors = {name: [] for name in models}
            for start in range(0, len(latent), batch_size):
                z = latent[start:start+batch_size].to(device, non_blocking=True).float()
                future, zero = z[:, :, 4:], torch.zeros_like(z[:, :, :4])
                h = teacher[start:start+batch_size].to(device, non_blocking=True)
                q = predicted[start:start+batch_size].to(device, non_blocking=True)
                conditions = {"N": torch.zeros_like(h), "T": h, "D": q, "P": q}
                noise = evaluation_noise(future.shape, start, repeat, 10042 if split == "val" else 20042).to(device)
                target = future-noise
                batch_errors = {name: [] for name in models}
                for level in NOISE_LEVELS:
                    t = torch.full((len(z),), level, device=device)
                    noisy = (1-level)*noise+level*future
                    for name, model in models.items():
                        precision = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                        with precision:
                            output = model(zero, noisy, t, conditions[name])
                        batch_errors[name].append((output.float()-target).square().flatten(1).mean(1).cpu())
                for name in models:
                    errors[name].append(torch.stack(batch_errors[name], 1))
            for name in models:
                repeated[name].append(torch.cat(errors[name]).numpy())
        repeated = {name: np.stack(values) for name, values in repeated.items()}
        return {name: values.mean(0) for name, values in repeated.items()}, repeated
    finally:
        for name, model in models.items():
            model.train(modes[name])


def save_evaluation(run, models, rows, z, h, q, split, seed, direct_only):
    errors, repeated = evaluate(models, z, h, q, split)
    stem = f"direct_{split}" if direct_only else f"step06000_{split}"
    np.savez_compressed(run / f"errors_{stem}.npz", **errors,
        **{name+"_repeats": value for name, value in repeated.items()},
        ids=np.asarray([r["id"] for r in rows]), videos=np.asarray([r["video"] for r in rows]),
        groups=np.asarray([r["group"] for r in rows]), noise_levels=np.asarray(NOISE_LEVELS),
        noise_seed=10042 if split == "val" else 20042, noise_repeats=4)
    summary = {"seed": seed, "step": 6000, "split": split, "direct_only": direct_only,
               "metrics": compact_metrics(rows, errors), "feature_metrics": feature_metrics(rows, h, q),
               "q_checkpoint": "runs/semantic_head_seed42/best.pt", "q_step": 2800,
               "reference_checkpoint": f"runs/history_semantics_seed{seed}/final.pt",
               "noise_seed": 10042 if split == "val" else 20042, "noise_repeats": 4,
               "pair_direction": "a_vs_b: improvement=1-mse_b/mse_a"}
    for name, (a, b) in PAIRS.items():
        if a in errors and b in errors:
            summary[name] = summarize_pair(rows, errors[a], errors[b])
            summary[name]["scope"] = f"{split} source-group sampling uncertainty for one training seed; not training-run variation"
    write_json(run / f"{split}_manifest.json", rows)
    write_json(run / f"summary_{stem}.json", summary)
    print(json.dumps({"seed": seed, "split": split, "metrics": summary["metrics"]}), flush=True)
