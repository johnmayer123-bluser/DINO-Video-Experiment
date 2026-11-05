"""Train paired no-history, semantic, semantic-dropout and latent-history arms."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from common import (ARMS, ROOT, Denoiser, apply_dropout, compact_metrics, evaluate,
                    load_split, save_checkpoint, save_evaluation, setup, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    setup()
    run = (args.run_root or args.root / "runs") / f"history_semantics_seed{args.seed}"
    if (run / "last.pt").exists() and not args.resume:
        raise FileExistsError("Run exists; use --resume")
    previous = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if args.resume else None
    _, train_z, train_h = load_split(args.root, "train")
    val_rows, val_z, val_h = load_split(args.root, "val")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base = Denoiser().cuda()
    models = {name: base if name == "N" else copy.deepcopy(base) for name in ARMS}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01)
                  for name, model in models.items()}
    generators = {"data": torch.Generator().manual_seed(args.seed+100),
                  "noise": torch.Generator(device="cuda").manual_seed(args.seed+200),
                  "drop": torch.Generator(device="cuda").manual_seed(args.seed+400)}
    config = {"seed": args.seed, "batch_size": 16, "lr": 2e-4, "weight_decay": .01,
              "clip_grad_norm": 1., "semantic_dropout": .1, "precision": "BF16 autocast, FP32 loss",
              "arms": {"N": "zero history and semantics", "S": "zero history, phi_a",
                       "SD": "S with per-example 10% semantic dropout", "L": "clean history, zero semantics"},
              "semantic_source": "A-only cached phi_a", "latent_slice": "latent[:,1:9]; A=first4, B=last4"}
    start, elapsed = 0, 0.
    if previous is not None:
        for name in ARMS:
            models[name].load_state_dict(previous["models"][name])
            optimizers[name].load_state_dict(previous["optimizers"][name])
        for name, generator in generators.items():
            generator.set_state(previous["rng"][name])
        torch.set_rng_state(previous["rng"]["torch"])
        torch.cuda.set_rng_state_all(previous["rng"]["cuda"])
        start, elapsed = previous["step"], previous.get("elapsed_seconds", 0.)
        if previous["config"]["seed"] != args.seed or start > args.steps:
            raise ValueError("Resume seed/step mismatch")
        del previous
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", config)
    began = time.perf_counter()
    recent = {name: [] for name in ARMS}
    for step in range(start+1, args.steps+1):
        indices = torch.randint(len(train_z), (16,), generator=generators["data"])
        z, h = train_z[indices].cuda().float(), train_h[indices].cuda()
        history, future = z[:, :, :4], z[:, :, 4:]
        zero_z, zero_h = torch.zeros_like(history), torch.zeros_like(h)
        t = torch.rand(len(z), device="cuda", generator=generators["noise"])
        noise = torch.randn(future.shape, device="cuda", generator=generators["noise"])
        dropped = torch.rand(len(z), device="cuda", generator=generators["drop"]) < .1
        conditions = {"N": (zero_z, zero_h), "S": (zero_z, h),
                      "SD": (zero_z, apply_dropout(h, dropped)), "L": (history, zero_h)}
        w = t[:, None, None, None, None]
        noisy, target = (1-w)*noise+w*future, future-noise
        for name, model in models.items():
            model.train(); optimizers[name].zero_grad(set_to_none=True)
            past, tokens = conditions[name]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(past, noisy, t, tokens)
            loss = (output.float()-target).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite loss/gradient: {name}, step {step}")
            optimizers[name].step(); recent[name].append(loss.item())
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": {k: float(np.mean(v[-50:])) for k,v in recent.items()}}), flush=True)
        if step % 1000 == 0 or step == args.steps:
            errors, _ = evaluate(models, val_z, val_h, "val", repeats=1)
            metrics = {"step": step, **compact_metrics(val_rows, errors),
                       "train_loss": {k: float(np.mean(v)) for k,v in recent.items()}}
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics)+"\n")
            save_checkpoint(run / "last.pt", {"step": step, "config": config,
                "models": {k: m.state_dict() for k,m in models.items()},
                "optimizers": {k: o.state_dict() for k,o in optimizers.items()},
                "rng": {**{k: g.get_state() for k,g in generators.items()},
                        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                "elapsed_seconds": elapsed+time.perf_counter()-began})
            print(json.dumps(metrics), flush=True)
            recent = {name: [] for name in ARMS}
    save_checkpoint(run / "final.pt", {"step": args.steps, "config": config,
                                      "models": {k: m.state_dict() for k,m in models.items()}})
    save_evaluation(run, models, val_rows, val_z, val_h, "val", args.steps, args.seed)


if __name__ == "__main__":
    main()
