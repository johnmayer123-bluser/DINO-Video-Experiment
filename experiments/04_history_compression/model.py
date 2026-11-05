"""Spatial/temporal token pooling with the original trainable parameter shapes."""

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
                    save_checkpoint, setup, write_json)
from denoiser_model import SemanticCrossAttention
from train_semantic_head import SemanticHead
from metrics import NOISE_LEVELS, summarize_pair

GRIDS = {64: (4, 4, 4), 16: (4, 2, 2), 4: (4, 1, 1), 1: (1, 1, 1)}


def load_split(root, split):
    """Only the manifest and VAE latents are needed; no teacher features are read."""
    cache = root / "cache/ucf6_12fps_192"
    rows = [r for r in json.loads((cache / "manifest.json").read_text(encoding="utf-8")) if r["split"] == split]
    if not rows:
        raise ValueError(f"Empty split: {split}")
    latents = []
    for row in rows:
        z = torch.load(cache / "vae" / (row["id"]+".pt"), map_location="cpu", weights_only=True)["latent"][:, 1:9]
        if z.shape != (16, 8, 24, 24) or not torch.isfinite(z).all():
            raise ValueError(f"Invalid VAE latent: {row['id']}")
        latents.append(z.clone())
    return rows, torch.stack(latents).pin_memory()


class PooledCrossAttention(SemanticCrossAttention):
    def forward(self, features, semantics):
        batch, channels, frames, height, width = features.shape
        count = int(np.prod(self.grid))
        if (frames, height, width) != (4, 6, 6) or semantics.shape != (batch, count, 1024):
            raise ValueError("Unexpected latent grid or compressed semantic shape")
        query = features.flatten(2).transpose(1, 2)
        query_input = self.query_norm(query)+self.query_position
        normalized = F.normalize(semantics.float(), dim=-1).to(features.dtype)
        context = self.semantic_projection(self.semantic_norm(normalized))
        active = semantics.detach().abs().sum((1, 2), keepdim=True).gt(0).to(context.dtype)
        position = self.semantic_position.reshape(1, 4, 4, 4, channels).permute(0, 4, 1, 2, 3)
        position = F.adaptive_avg_pool3d(position, self.grid).flatten(2).transpose(1, 2)
        context = context+active*position
        attended, _ = self.attention(query_input, context, context, need_weights=False)
        query = query+attended
        query = query+self.ff(self.ff_norm(query))
        return query.transpose(1, 2).reshape(batch, channels, frames, height, width)


class CompressionModel(nn.Module):
    def __init__(self, tokens=64, denoiser=None, encoder=None):
        super().__init__()
        self.denoiser = Denoiser() if denoiser is None else denoiser
        self.encoder = SemanticHead() if encoder is None else encoder
        self.set_tokens(tokens)

    def set_tokens(self, tokens):
        self.grid = GRIDS[tokens]
        attention = self.denoiser.semantic_attention
        attention.__class__ = SemanticCrossAttention if tokens == 64 else PooledCrossAttention
        attention.grid = self.grid

    def encode(self, history):
        predicted = self.encoder(history)
        if self.grid != GRIDS[64]:
            predicted = F.adaptive_avg_pool3d(predicted.permute(0, 4, 1, 2, 3), self.grid).permute(0, 2, 3, 4, 1)
        return F.normalize(predicted.float(), dim=-1).flatten(1, 3)

    def forward(self, history, noisy, timestep, dropped=None):
        tokens = self.encode(history).half()
        if dropped is not None:
            tokens = apply_dropout(tokens, dropped)
        return self.denoiser(torch.zeros_like(history), noisy, timestep, tokens)


def paired_models(seed, device="cuda"):
    torch.manual_seed(seed)
    denoiser = Denoiser().to(device)
    torch.manual_seed(seed+300)
    encoder = SemanticHead().to(device)
    base = CompressionModel(64, denoiser, encoder)
    models = {f"K{k}": copy.deepcopy(base) for k in (16, 4, 1)}
    for name, model in models.items():
        model.set_tokens(int(name[1:]))
    return models


def precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate(models, latent, split, repeats=4, batch_size=16):
    device = next(next(iter(models.values())).parameters()).device
    modes = {k:m.training for k,m in models.items()}
    repeated = {k:[] for k in models}
    try:
        for model in models.values():
            model.eval()
        for repeat in range(repeats):
            errors = {k:[] for k in models}
            for start in range(0, len(latent), batch_size):
                z = latent[start:start+batch_size].to(device, non_blocking=True).float()
                history, future = z[:, :, :4], z[:, :, 4:]
                with precision(device):
                    conditions = {k:m.encode(history).half() for k,m in models.items()}
                zero = torch.zeros_like(history)
                noise = evaluation_noise(future.shape, start, repeat, 10042 if split == "val" else 20042).to(device)
                target = future-noise
                batch_errors = {k:[] for k in models}
                for level in NOISE_LEVELS:
                    t = torch.full((len(z),), level, device=device)
                    noisy = (1-level)*noise+level*future
                    for name, model in models.items():
                        with precision(device):
                            output = model.denoiser(zero, noisy, t, conditions[name])
                        batch_errors[name].append((output.float()-target).square().flatten(1).mean(1).cpu())
                for name in models:
                    errors[name].append(torch.stack(batch_errors[name], 1))
            for name in models:
                repeated[name].append(torch.cat(errors[name]).numpy())
        repeated = {k:np.stack(v) for k,v in repeated.items()}
        return {k:v.mean(0) for k,v in repeated.items()}, repeated
    finally:
        for name, model in models.items():
            model.train(modes[name])


def save_evaluation(run, models, rows, z, split, seed, step):
    errors, repeated = evaluate(models, z, split)
    stem = f"step{step:05d}_{split}"
    np.savez_compressed(run / f"errors_{stem}.npz", **errors, **{k+"_repeats":v for k,v in repeated.items()},
        ids=np.asarray([r["id"] for r in rows]), videos=np.asarray([r["video"] for r in rows]), groups=np.asarray([r["group"] for r in rows]),
        noise_levels=np.asarray(NOISE_LEVELS), noise_seed=10042 if split == "val" else 20042, noise_repeats=4)
    summary = {"seed":seed, "step":step, "split":split, "metrics":compact_metrics(rows, errors),
               "noise_seed":10042 if split == "val" else 20042, "noise_repeats":4,
               "pair_direction":"a_vs_b: improvement=1-mse_b/mse_a"}
    for name in ("K16", "K4", "K1"):
        summary[f"K64_vs_{name}"] = summarize_pair(rows, errors["K64"], errors[name])
        summary[f"K64_vs_{name}"]["scope"] = f"{split} source-group uncertainty for one training seed; not training-run variation"
    write_json(run / f"{split}_manifest.json", rows)
    write_json(run / f"summary_{stem}.json", summary)
    print(json.dumps({"seed":seed, "split":split, "metrics":summary["metrics"]}), flush=True)
