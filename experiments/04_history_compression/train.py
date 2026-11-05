"""Train paired 16/4/1-token history conditions without teacher supervision."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import (ROOT, GRIDS, compact_metrics, evaluate, load_split, paired_models,
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
    run = (args.run_root or args.root / "runs") / f"history_compression_seed{args.seed}"
    if (run / "last.pt").exists() and not args.resume:
        raise FileExistsError("Existing run; use --resume")
    previous = torch.load(run / "last.pt", map_location="cpu", weights_only=False) if args.resume else None
    _, train_z = load_split(args.root, "train")
    rows, val_z = load_split(args.root, "val")
    models = paired_models(args.seed)
    optimizers = {k:torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=.01) for k,m in models.items()}
    generators = {"data":torch.Generator().manual_seed(args.seed+100),
                  "noise":torch.Generator(device="cuda").manual_seed(args.seed+200),
                  "drop":torch.Generator(device="cuda").manual_seed(args.seed+400)}
    config = {"seed":args.seed, "batch_size":16, "lr":2e-4, "weight_decay":.01, "clip_grad_norm":1.,
              "semantic_dropout":.1, "precision":"BF16 autocast, FP32 loss", "grids":{f"K{k}":GRIDS[k] for k in (16,4,1)},
              "denoiser_init_seed":args.seed, "encoder_init_seed":args.seed+300,
              "objective":"FM only; no teacher features", "pooling":"encoder outputs and original semantic positions; before normalization",
              "denoiser_history":"zeros", "checkpoint_selection":"final equal budget; no best selection"}
    start, elapsed = 0, 0.
    if previous is not None:
        for name, model in models.items():
            model.load_state_dict(previous["models"][name]); optimizers[name].load_state_dict(previous["optimizers"][name])
        for name, rng in generators.items():
            rng.set_state(previous["rng"][name])
        torch.set_rng_state(previous["rng"]["torch"]); torch.cuda.set_rng_state_all(previous["rng"]["cuda"])
        start, elapsed = previous["step"], previous.get("elapsed_seconds", 0.)
        if previous["config"]["seed"] != args.seed or start > args.steps:
            raise ValueError("Resume seed/step mismatch")
        del previous
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", config)
    began, recent = time.perf_counter(), {k:[] for k in models}
    for step in range(start+1, args.steps+1):
        indices = torch.randint(len(train_z), (16,), generator=generators["data"])
        z = train_z[indices].cuda().float()
        history, future = z[:, :, :4], z[:, :, 4:]
        t = torch.rand(len(z), device="cuda", generator=generators["noise"])
        noise = torch.randn(future.shape, device="cuda", generator=generators["noise"])
        dropped = torch.rand(len(z), device="cuda", generator=generators["drop"]) < .1
        w = t[:,None,None,None,None]
        noisy, target = (1-w)*noise+w*future, future-noise
        for name, model in models.items():
            model.train(); optimizers[name].zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(history, noisy, t, dropped)
            loss = (output.float()-target).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite loss/gradient: {name}, step {step}")
            optimizers[name].step(); recent[name].append(loss.item())
        if step % 50 == 0:
            print(json.dumps({"step":step,"fm_loss":{k:float(np.mean(v[-50:])) for k,v in recent.items()}}), flush=True)
        if step % 1000 == 0 or step == args.steps:
            errors, _ = evaluate(models, val_z, "val", repeats=1)
            metrics = {"step":step,"split":"val","noise_repeats":1,**compact_metrics(rows,errors),
                       "train_fm_loss":{k:float(np.mean(v)) for k,v in recent.items()}}
            with (run / "metrics.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(metrics)+"\n")
            save_checkpoint(run / "last.pt", {"step":step,"config":config,"models":{k:m.state_dict() for k,m in models.items()},
                "optimizers":{k:o.state_dict() for k,o in optimizers.items()},
                "rng":{**{k:g.get_state() for k,g in generators.items()},"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()},
                "elapsed_seconds":elapsed+time.perf_counter()-began})
            print(json.dumps(metrics),flush=True);recent={k:[] for k in models}
    save_checkpoint(run / "final.pt", {"step":args.steps,"config":config,"models":{k:m.state_dict() for k,m in models.items()}})


if __name__ == "__main__":
    main()
