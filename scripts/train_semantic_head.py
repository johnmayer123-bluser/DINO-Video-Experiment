"""Fit a small history-latent predictor against frozen V-JEPA targets."""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class SemanticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(16, 64, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv3d(256, 1024, 1),
        )

    def forward(self, latent):
        return self.layers(latent).permute(0, 2, 3, 4, 1)


def load_split(cache, records, split):
    rows = [row for row in records if row["split"] == split]
    latent, targets = [], []
    for row in rows:
        name = row["id"] + ".pt"
        z = torch.load(cache / "vae" / name, map_location="cpu", weights_only=True)["latent"]
        h = torch.load(cache / "jepa" / name, map_location="cpu", weights_only=True)["phi_a"]
        latent.append(z[:, 1:5].clone())
        targets.append(h)
    return rows, torch.stack(latent), torch.stack(targets)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def macro_average(values, rows):
    by_video = defaultdict(list)
    for value, row in zip(values, rows):
        by_video[(row["video"], row["group"])].append(float(value))
    by_group = defaultdict(list)
    for (video, group), scores in by_video.items():
        by_group[(video.split("/")[0], group)].append(float(np.mean(scores)))
    by_class = defaultdict(list)
    for (category, _), scores in by_group.items():
        by_class[category].append(float(np.mean(scores)))
    per_class = {category: float(np.mean(scores)) for category, scores in by_class.items()}
    return float(np.mean(list(per_class.values()))), per_class


def fixed_negatives(rows, seed):
    rng = random.Random(seed)
    candidates = []
    for row in rows:
        videos = defaultdict(list)
        category = row["video"].split("/")[0]
        for index, other in enumerate(rows):
            if other["video"].split("/")[0] == category and other["group"] != row["group"]:
                videos[other["video"]].append(index)
        candidates.append(videos)
    count = min(8, min(len(videos) for videos in candidates))
    if count < 1:
        raise ValueError("Validation requires another source group in each category.")
    return torch.tensor([
        [rng.choice(videos[name]) for name in rng.sample(sorted(videos), count)]
        for videos in candidates
    ], device="cuda")


@torch.inference_mode()
def evaluate(model, latent, targets, rows, train_mean, negatives, batch_size):
    model.eval()
    predictions = []
    for start in range(0, len(latent), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(latent[start:start + batch_size].cuda(non_blocking=True))
        predictions.append(F.normalize(output.float(), dim=-1))
    predicted = torch.cat(predictions)
    raw = (predicted * targets).sum(-1).mean((1, 2, 3))
    centered_p = F.normalize((predicted - train_mean).flatten(1), dim=-1)
    centered_t = F.normalize((targets - train_mean).flatten(1), dim=-1)
    positive = (centered_p * centered_t).sum(-1)
    scores = centered_p @ centered_t.T
    negative_scores = scores.gather(1, negatives)
    metrics = {
        "cosine": raw,
        "centered_cosine": positive,
        "positive_negative_margin": positive - negative_scores.mean(1),
        "pairwise_accuracy": (positive[:, None] > negative_scores).float().mean(1),
        "retrieval_top1": (positive > negative_scores.max(1).values).float(),
    }
    report, per_class = {}, {}
    for name, values in metrics.items():
        report[name], per_class[name] = macro_average(values.cpu().tolist(), rows)
    report["loss"] = 1 - report["cosine"]
    report["negative_count"] = negatives.shape[1]
    report["retrieval_chance"] = 1 / (1 + negatives.shape[1])
    report["prediction_variance_ratio"] = float(
        predicted.var(0, unbiased=False).mean() / targets.var(0, unbiased=False).mean()
    )
    report["per_class"] = {
        category: {name: scores[category] for name, scores in per_class.items()}
        for category in per_class["cosine"]
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cache = root / "cache/ucf6_12fps_192"
    run = root / f"runs/semantic_head_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(8)
    records = json.loads((cache / "manifest.json").read_text())
    train_rows, train_z, train_h = load_split(cache, records, "train")
    val_rows, val_z, val_h = load_split(cache, records, "val")
    assert not ({r["group"] for r in train_rows} & {r["group"] for r in val_rows})
    # Normalize before centering because the cosine objective leaves scale unspecified.
    train_h = F.normalize(train_h.float(), dim=-1)
    train_mean = train_h.mean(0).cuda()
    val_h = F.normalize(val_h.float().cuda(), dim=-1)
    mean_scores = (F.normalize(train_mean, dim=-1)[None] * val_h).sum(-1).mean((1, 2, 3))
    baseline, baseline_classes = macro_average(mean_scores.cpu().tolist(), val_rows)
    negatives = fixed_negatives(val_rows, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(train_z, train_h.half()), batch_size=args.batch_size,
                        shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    model = SemanticHead().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best_score, start_step = -float("inf"), 0
    if args.resume:
        checkpoint = torch.load(run / "last.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_score, start_step = checkpoint["best_score"], checkpoint["step"]
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "train_windows": len(train_z), "val_windows": len(val_z),
        "parameters": sum(p.numel() for p in model.parameters()),
        "target": "phi_a", "latent_slice": [1, 5],
        "feature_use": "per-token L2 normalized",
        "selection": "highest validation group/class macro centered positive-negative margin",
        "aggregation": "window -> video -> source group -> class macro",
        "train_mean_baseline_cosine": baseline,
        "train_mean_baseline_per_class": baseline_classes,
        "retrieval_negatives": negatives.shape[1],
    }
    (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config), flush=True)
    iterator = iter(loader)
    began = time.perf_counter()
    running_loss = []
    for step in range(start_step + 1, args.steps + 1):
        try:
            z, target = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            z, target = next(iterator)
        model.train()
        z, target = z.cuda(non_blocking=True), target.cuda(non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(z)
        predicted = F.normalize(output.float(), dim=-1)
        loss = 1 - (predicted * target).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss.append(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, val_z, val_h, val_rows, train_mean, negatives, args.batch_size)
            result.update({
                "step": step, "train_loss": float(np.mean(running_loss)),
                "elapsed_seconds": time.perf_counter() - began,
                "mean_baseline_cosine": baseline,
                "cosine_gain_over_mean": result["cosine"] - baseline,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            })
            running_loss.clear()
            improved = result["positive_negative_margin"] > best_score
            best_score = max(best_score, result["positive_negative_margin"])
            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "train_mean": train_mean.cpu(), "step": step, "best_score": best_score,
                "config": config, "metrics": result,
            }
            save_checkpoint(run / "last.pt", checkpoint)
            if improved:
                save_checkpoint(run / "best.pt", checkpoint)
                (run / "best_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "per_class"}), flush=True)
    print(f"DONE {run}", flush=True)


if __name__ == "__main__":
    main()
