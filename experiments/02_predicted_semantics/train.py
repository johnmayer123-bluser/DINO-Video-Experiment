"""Train P from scratch using only frozen q(z_A), with 10% semantic dropout."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from conditions import (ROOT, Denoiser, apply_dropout, compact_metrics, evaluate,
                        load_predictor, load_split, predict_semantics,
                        save_checkpoint, setup, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    setup()
    run = (args.run_root or args.root / "runs") / f"predicted_semantics_seed{args.seed}"
    if (run / "last.pt").exists() and not args.resume:
        raise FileExistsError("Existing run; use --resume")
    previous = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if args.resume else None
    _, train_z, train_h = load_split(args.root, "train")
    rows, val_z, val_h = load_split(args.root, "val")
    predictor = load_predictor(args.root)
    train_q, val_q = predict_semantics(predictor, train_z), predict_semantics(predictor, val_z)
    del predictor, train_h
    # Feature prediction precedes seeding so q initialization cannot alter P's initialization.
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = Denoiser().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
    generators = {"data": torch.Generator().manual_seed(args.seed+100),
                  "noise": torch.Generator(device="cuda").manual_seed(args.seed+200),
                  "drop": torch.Generator(device="cuda").manual_seed(args.seed+400)}
    config = {"seed": args.seed, "batch_size": 16, "lr": 2e-4, "weight_decay": .01,
              "clip_grad_norm": 1., "semantic_dropout": .1, "precision": "BF16 autocast, FP32 loss",
              "q_checkpoint": "runs/semantic_head_seed42/best.pt", "q_step": 2800,
              "condition": "frozen q(z_A), normalized per token; zero latent history",
              "initialization": "fresh original Denoiser, no teacher warm start"}
    start, elapsed = 0, 0.
    if previous is not None:
        model.load_state_dict(previous["model"]); optimizer.load_state_dict(previous["optimizer"])
        for name, rng in generators.items():
            rng.set_state(previous["rng"][name])
        torch.set_rng_state(previous["rng"]["torch"]); torch.cuda.set_rng_state_all(previous["rng"]["cuda"])
        start, elapsed = previous["step"], previous.get("elapsed_seconds", 0.)
        if previous["config"]["seed"] != args.seed or start > args.steps:
            raise ValueError("Resume seed/step mismatch")
        del previous
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", config)
    began, recent = time.perf_counter(), []
    for step in range(start+1, args.steps+1):
        indices = torch.randint(len(train_z), (16,), generator=generators["data"])
        z, q = train_z[indices].cuda().float(), train_q[indices].cuda()
        future, zero = z[:, :, 4:], torch.zeros_like(z[:, :, :4])
        t = torch.rand(len(z), device="cuda", generator=generators["noise"])
        noise = torch.randn(future.shape, device="cuda", generator=generators["noise"])
        dropped = torch.rand(len(z), device="cuda", generator=generators["drop"]) < .1
        w = t[:, None, None, None, None]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(zero, (1-w)*noise+w*future, t, apply_dropout(q, dropped))
        loss = (output.float()-(future-noise)).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f"Nonfinite loss/gradient at step {step}")
        optimizer.step(); recent.append(loss.item())
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": float(np.mean(recent[-50:]))}), flush=True)
        if step % 1000 == 0 or step == args.steps:
            errors, _ = evaluate({"P": model}, val_z, val_h, val_q, "val", repeats=1)
            metrics = {"step": step, "split": "val", "noise_repeats": 1,
                       **compact_metrics(rows, errors), "train_loss": float(np.mean(recent))}
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics)+"\n")
            save_checkpoint(run / "last.pt", {"step": step, "config": config, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": {**{k: g.get_state() for k,g in generators.items()},
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                "elapsed_seconds": elapsed+time.perf_counter()-began})
            print(json.dumps(metrics), flush=True); recent = []
    save_checkpoint(run / "final.pt", {"step": args.steps, "config": config, "model": model.state_dict()})


if __name__ == "__main__":
    main()
