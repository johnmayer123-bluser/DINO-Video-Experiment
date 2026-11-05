"""Sixteen history tokens with a genuinely narrower encoder/context interface."""

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Load the token experiment explicitly; the two experiments never share a `model` import.
spec = importlib.util.spec_from_file_location("q4_token_model", Path(__file__).resolve().parents[1] / "model.py")
token = importlib.util.module_from_spec(spec)
spec.loader.exec_module(token)
ROOT = token.ROOT
load_split, evaluate = token.load_split, token.evaluate
save_checkpoint, setup, write_json = token.save_checkpoint, token.setup, token.write_json
compact_metrics, summarize_pair = token.compact_metrics, token.summarize_pair


class WidthCrossAttention(token.PooledCrossAttention):
    def forward(self, features, semantics):
        batch, channels, frames, height, width = features.shape
        if (frames, height, width) != (4, 6, 6) or semantics.shape != (batch, 16, self.semantic_projection.in_features):
            raise ValueError("Unexpected latent grid or narrow semantic shape")
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


def narrow(model, width):
    """Deterministic parameter prefixes; preserve the original head's fan-in scale."""
    model.width = width
    if width == 1024:
        return
    head = model.encoder.layers[-1]
    head.weight = nn.Parameter(head.weight[:width].clone())
    head.bias = nn.Parameter(head.bias[:width].clone())
    head.out_channels = width
    attention = model.denoiser.semantic_attention
    projection = attention.semantic_projection
    projection.weight = nn.Parameter(projection.weight[:, :width].clone()*math.sqrt(1024/width))
    projection.in_features = width
    attention.semantic_norm.normalized_shape = (width,)
    attention.__class__ = WidthCrossAttention


class WidthModel(token.CompressionModel):
    def __init__(self, width=1024, denoiser=None, encoder=None):
        super().__init__(16, denoiser, encoder)
        narrow(self, width)


def paired_models(seed, device="cuda"):
    torch.manual_seed(seed)
    denoiser = token.Denoiser().to(device)
    torch.manual_seed(seed+300)
    encoder = token.SemanticHead().to(device)
    base = WidthModel(1024, denoiser, encoder)
    models = {f"D{d}":copy.deepcopy(base) for d in (512,256,128)}
    for name, model in models.items():
        narrow(model, int(name[1:]))
    return models


def save_evaluation(run, models, rows, z, split, seed, step):
    errors, repeated = evaluate(models,z,split)
    stem = f"step{step:05d}_{split}"
    np.savez_compressed(run / f"errors_{stem}.npz", **errors, **{k+"_repeats":v for k,v in repeated.items()},
        ids=np.asarray([r["id"] for r in rows]),videos=np.asarray([r["video"] for r in rows]),groups=np.asarray([r["group"] for r in rows]),
        noise_levels=np.asarray(token.NOISE_LEVELS),noise_seed=10042 if split=="val" else 20042,noise_repeats=4)
    summary = {"seed":seed,"step":step,"split":split,"metrics":compact_metrics(rows,errors),
               "parameters":{k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
               "state_kib":{k:2*16*m.width/1024 for k,m in models.items()},
               "noise_seed":10042 if split=="val" else 20042,"noise_repeats":4,
               "pair_direction":"a_vs_b: improvement=1-mse_b/mse_a"}
    for name in ("D512","D256","D128"):
        summary[f"D1024_vs_{name}"] = summarize_pair(rows,errors["D1024"],errors[name])
        summary[f"D1024_vs_{name}"]["scope"] = f"{split} source-group uncertainty for one training seed; not training-run variation"
    write_json(run / f"{split}_manifest.json",rows)
    write_json(run / f"summary_{stem}.json",summary)
    print(json.dumps({"seed":seed,"split":split,"metrics":summary["metrics"]}),flush=True)
