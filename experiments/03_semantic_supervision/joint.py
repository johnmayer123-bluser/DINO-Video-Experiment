"""Jointly trainable history encoder and paired flow-matching evaluation."""

import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/01_history_semantics"))
from common import (Denoiser, apply_dropout, compact_metrics, evaluation_noise,
                    load_split, save_checkpoint, setup, write_json)
from metrics import NOISE_LEVELS, summarize_pair
from train_semantic_head import SemanticHead, macro_average

PAIRS = {"C_vs_J": ("C", "J"), "N_vs_C": ("N", "C"), "N_vs_J": ("N", "J"),
         "N_vs_P": ("N", "P"), "P_vs_C": ("P", "C"), "P_vs_J": ("P", "J")}


def normalized_tokens(prediction):
    return F.normalize(prediction.float(), dim=-1).flatten(1, 3)


def alignment_loss(tokens, teacher):
    return 1-(tokens*F.normalize(teacher.float(), dim=-1)).sum(-1).mean()


def objective(flow_loss, tokens, teacher, supervised):
    return flow_loss+alignment_loss(tokens, teacher) if supervised else flow_loss


class JointModel(nn.Module):
    def __init__(self, denoiser=None, encoder=None):
        super().__init__()
        self.denoiser = Denoiser() if denoiser is None else denoiser
        self.encoder = SemanticHead() if encoder is None else encoder

    def encode(self, history):
        return normalized_tokens(self.encoder(history))

    def forward(self, history, noisy, timestep, dropped=None):
        tokens = self.encode(history)
        condition = tokens.half()
        if dropped is not None:
            condition = apply_dropout(condition, dropped)
        output = self.denoiser(torch.zeros_like(history), noisy, timestep, condition)
        return output, tokens


def paired_models(seed, device="cuda"):
    torch.manual_seed(seed)
    denoiser = Denoiser().to(device)
    torch.manual_seed(seed+300)
    encoder = SemanticHead().to(device)
    control = JointModel(denoiser, encoder)
    return {"C": control, "J": copy.deepcopy(control)}


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def predict_reference(predictor, latent, batch_size=16):
    """Preserve Q2's original CPU-half A slice and CUDA transfer layout for P."""
    result = []
    for start in range(0, len(latent), batch_size):
        history = latent[start:start+batch_size, :, :4].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = predictor(history)
        result.append(F.normalize(predicted.float(), dim=-1).flatten(1, 3).half().cpu())
    return torch.cat(result).pin_memory()


@torch.inference_mode()
def evaluate(models, latent, split, teacher=None, predictor=None, repeats=4, batch_size=16):
    device = next(next(iter(models.values())).parameters()).device
    reference = predict_reference(predictor, latent, batch_size) if "P" in models else None
    modes = {name: model.training for name, model in models.items()}
    repeated = {name: [] for name in models}
    alignments = {name: [] for name in ("C", "J") if name in models and teacher is not None}
    try:
        for model in models.values():
            model.eval()
        for repeat in range(repeats):
            errors = {name: [] for name in models}
            for start in range(0, len(latent), batch_size):
                z = latent[start:start+batch_size].to(device, non_blocking=True).float()
                history, future = z[:, :, :4], z[:, :, 4:]
                zero = torch.zeros_like(history)
                tokens = {"N": torch.zeros(len(z), 64, 1024, device=device, dtype=torch.float16)}
                with precision(device):
                    for name in ("C", "J"):
                        if name in models:
                            predicted = models[name].encode(history)
                            tokens[name] = predicted.half()
                            if repeat == 0 and name in alignments:
                                h = teacher[start:start+batch_size].to(device)
                                loss = 1-(predicted*F.normalize(h.float(), dim=-1)).sum(-1).mean(1)
                                alignments[name].append(loss.cpu())
                if "P" in models:
                    tokens["P"] = reference[start:start+batch_size].to(device, non_blocking=True)
                noise = evaluation_noise(future.shape, start, repeat, 10042 if split == "val" else 20042).to(device)
                target = future-noise
                batch_errors = {name: [] for name in models}
                for level in NOISE_LEVELS:
                    t = torch.full((len(z),), level, device=device)
                    noisy = (1-level)*noise+level*future
                    for name, model in models.items():
                        denoiser = model.denoiser if name in ("C", "J") else model
                        with precision(device):
                            output = denoiser(zero, noisy, t, tokens[name])
                        batch_errors[name].append((output.float()-target).square().flatten(1).mean(1).cpu())
                for name in models:
                    errors[name].append(torch.stack(batch_errors[name], 1))
            for name in models:
                repeated[name].append(torch.cat(errors[name]).numpy())
        repeated = {name: np.stack(values) for name, values in repeated.items()}
        return ({name: values.mean(0) for name, values in repeated.items()}, repeated,
                {name: torch.cat(values).numpy() for name, values in alignments.items()})
    finally:
        for name, model in models.items():
            model.train(modes[name])


def save_evaluation(run, models, rows, z, h, predictor, split, seed, step):
    errors, repeated, alignment = evaluate(models, z, split, h, predictor)
    stem = f"step{step:05d}_{split}"
    np.savez_compressed(run / f"errors_{stem}.npz", **errors,
        **{name+"_repeats": value for name,value in repeated.items()},
        ids=np.asarray([r["id"] for r in rows]), videos=np.asarray([r["video"] for r in rows]),
        groups=np.asarray([r["group"] for r in rows]), noise_levels=np.asarray(NOISE_LEVELS),
        noise_seed=10042 if split == "val" else 20042, noise_repeats=4)
    summary = {"seed": seed, "step": step, "split": split, "primary_comparison": "C_vs_J",
               "metrics": compact_metrics(rows, errors), "alignment_loss": {k: macro_average(v, rows)[0] for k,v in alignment.items()},
               "noise_seed": 10042 if split == "val" else 20042, "noise_repeats": 4,
               "pair_direction": "a_vs_b: improvement=1-mse_b/mse_a"}
    for name, (a, b) in PAIRS.items():
        if a in errors and b in errors:
            summary[name] = summarize_pair(rows, errors[a], errors[b])
            summary[name]["scope"] = f"{split} source-group sampling uncertainty for one training seed; not training-run variation"
    write_json(run / f"{split}_manifest.json", rows)
    write_json(run / f"summary_{stem}.json", summary)
    print(json.dumps({"seed": seed, "split": split, "metrics": summary["metrics"]}), flush=True)
