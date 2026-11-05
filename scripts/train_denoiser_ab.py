"""Paired A/B flow matching: history latent alone vs history plus frozen semantics."""

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from denoiser_model import Denoiser
from denoiser_stats import NOISE_LEVELS, summarize_pair
from train_semantic_head import SemanticHead, macro_average, save_checkpoint


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_latents(cache, records, split):
    rows = [r for r in records if r["split"] == split]
    latents = [torch.load(cache / "vae" / (r["id"] + ".pt"),
                          map_location="cpu", weights_only=True)["latent"][:, 1:9]
               for r in rows]
    return rows, torch.stack(latents).contiguous().pin_memory()


@torch.no_grad()
def predict_semantics(model, latents, batch_size):
    result = []
    for start in range(0, len(latents), batch_size):
        z = latents[start:start + batch_size, :, :4].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model(z)
        result.append(F.normalize(predicted.float(), dim=-1).flatten(1, 3).half().cpu())
    return torch.cat(result).pin_memory()


def counterfactual_indices(rows, seed):
    rng = random.Random(seed)
    shuffled, past, eligible = [], [], []
    for index, row in enumerate(rows):
        category = row["video"].split("/")[0]
        other = [j for j, r in enumerate(rows)
                 if r["video"].split("/")[0] == category and r["group"] != row["group"]]
        shuffled.append(rng.choice(other))
        # Require non-overlapping A blocks from the same video, at least 16/12 s apart.
        earlier = [j for j, r in enumerate(rows) if r["video"] == row["video"]
                   and row["start_seconds"] - r["start_seconds"] >= 16 / 12 - 1e-6]
        eligible.append(bool(earlier))
        past.append(max(earlier, key=lambda j: rows[j]["start_seconds"]) if earlier else index)
    return torch.tensor(shuffled), torch.tensor(past), np.asarray(eligible)


@torch.inference_mode()
def evaluate(models, latent, semantics, batch_size, noise_seed, counterfactuals=None):
    for model in models.values():
        model.eval()
    names = ["A", "B"]
    if counterfactuals is not None:
        names += ["B_zero", "B_shuffled", "B_past"]
    errors = {name: [] for name in names}
    generator = torch.Generator(device="cuda").manual_seed(noise_seed)
    for start in range(0, len(latent), batch_size):
        end = start + batch_size
        z = latent[start:end].cuda(non_blocking=True).float()
        history, future = z[:, :, :4], z[:, :, 4:]
        h = semantics[start:end].cuda(non_blocking=True)
        conditions = {"A": torch.zeros_like(h), "B": h}
        if counterfactuals is not None:
            shuffled, past, _ = counterfactuals
            conditions.update({"B_zero": torch.zeros_like(h),
                               "B_shuffled": semantics[shuffled[start:end]].cuda(),
                               "B_past": semantics[past[start:end]].cuda()})
        noise = torch.randn(future.shape, device="cuda", generator=generator)
        target = future - noise
        batch_errors = {name: [] for name in names}
        for level in NOISE_LEVELS:
            timestep = torch.full((len(z),), level, device="cuda")
            noisy = (1 - level) * noise + level * future
            for name in names:
                model = models["A" if name == "A" else "B"]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(history, noisy, timestep, conditions[name])
                mse = (output.float() - target).square().flatten(1).mean(1)
                batch_errors[name].append(mse.cpu())
        for name in names:
            errors[name].append(torch.stack(batch_errors[name], dim=1))
    return {name: torch.cat(values).numpy() for name, values in errors.items()}


def compact_metrics(rows, errors):
    a = macro_average(errors["A"].mean(1), rows)[0]
    b = macro_average(errors["B"].mean(1), rows)[0]
    return {"mse_a": a, "mse_b": b, "relative_improvement": 1 - b / a,
            "by_t": {str(t): {name: macro_average(values[:, i], rows)[0]
                                for name, values in errors.items()}
                     for i, t in enumerate(NOISE_LEVELS)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    root = args.root.resolve()
    cache = root / "cache/ucf6_12fps_192"
    run = root / f"runs/denoiser_ab_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "last.pt").exists() and not (args.resume or args.eval_only):
        raise FileExistsError("Existing run: use --resume to continue its paired experiment.")
    for filename in ("train_denoiser_ab.py", "denoiser_model.py", "denoiser_stats.py", "train_semantic_head.py"):
        shutil.copyfile(Path(__file__).parent / filename, run / filename)
    records = json.loads((cache / "manifest.json").read_text())
    train_rows, train_z = load_latents(cache, records, "train")
    val_rows, val_z = load_latents(cache, records, "val")
    assert not ({r["group"] for r in train_rows} & {r["group"] for r in val_rows})
    q_path = root / "runs/semantic_head_seed42/best.pt"
    checkpoint = torch.load(q_path, map_location="cpu", weights_only=False)
    q = SemanticHead().cuda().eval().requires_grad_(False)
    q.load_state_dict(checkpoint["model"])
    train_h = predict_semantics(q, train_z, args.eval_batch_size)
    val_h = predict_semantics(q, val_z, args.eval_batch_size)
    q_step = checkpoint["step"]
    del q, checkpoint
    torch.cuda.empty_cache()
    print("Cached frozen q(z_A) for train/validation; no future teacher features used.", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_a = Denoiser().cuda()
    models = {"A": model_a, "B": copy.deepcopy(model_a)}
    assert all(torch.equal(value, models["B"].state_dict()[key])
               for key, value in models["A"].state_dict().items())
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
                  for name, model in models.items()}
    data_rng = torch.Generator().manual_seed(args.seed + 100)
    noise_rng = torch.Generator(device="cuda").manual_seed(args.seed + 200)
    validation_seed = args.seed + 10_000
    start_step, accumulated_seconds = 0, 0.0
    if args.resume or args.eval_only:
        previous = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        for name in models:
            models[name].load_state_dict(previous["models"][name])
            optimizers[name].load_state_dict(previous["optimizers"][name])
        data_rng.set_state(previous["data_rng"])
        noise_rng.set_state(previous["noise_rng"])
        start_step, accumulated_seconds = previous["step"], previous["elapsed_seconds"]
        del previous
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "parameters_per_arm": sum(p.numel() for p in model_a.parameters()),
        "train_windows": len(train_rows), "validation_windows": len(val_rows),
        "semantic_checkpoint": str(q_path), "semantic_checkpoint_step": q_step,
        "semantic_checkpoint_sha256": hashlib.sha256(q_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
        "condition_A": "zero semantic tokens, full clean history latent",
        "condition_B": "frozen normalized q(z_A), full clean history latent",
        "objective": "z_t=(1-t)*epsilon+t*z_B; target=z_B-epsilon; mean squared velocity error",
        "paired_training": "identical initial weights; shared batch indices, t and Gaussian noise each step",
        "sampling": "uniform train-window sampling with replacement",
        "primary_checkpoint": "final equal-budget step for both arms; no best-checkpoint selection",
        "validation_noise_seed": validation_seed, "validation_t": list(NOISE_LEVELS),
        "precision": "bf16 autocast, fp32 loss; no dropout",
        "aggregation": "window -> video -> source group -> class macro",
    }
    if not args.eval_only:
        write_json(run / "config.json", config)
    print(json.dumps(config), flush=True)
    began = time.perf_counter()
    recent = {name: [] for name in models}
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step + 1, args.steps + 1) if not args.eval_only else ():
        indices = torch.randint(len(train_z), (args.batch_size,), generator=data_rng)
        z = train_z[indices].cuda().float()
        history, future = z[:, :, :4], z[:, :, 4:]
        h = train_h[indices].cuda()
        conditions = {"A": torch.zeros_like(h), "B": h}
        timestep = torch.rand(len(z), device="cuda", generator=noise_rng)
        noise = torch.randn(future.shape, device="cuda", generator=noise_rng)
        weight = timestep[:, None, None, None, None]
        noisy = (1 - weight) * noise + weight * future
        target = future - noise
        for name, model in models.items():
            model.train()
            optimizers[name].zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(history, noisy, timestep, conditions[name])
            loss = (output.float() - target).square().mean()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Nonfinite loss/gradient in {name} at step {step}")
            optimizers[name].step()
            recent[name].append(loss.item())
        if step % 50 == 0:
            print(json.dumps({"train_step": step,
                              "train_loss": {k: float(np.mean(v[-50:])) for k, v in recent.items()},
                              "elapsed_seconds": accumulated_seconds + time.perf_counter() - began}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            errors = evaluate(models, val_z, val_h, args.eval_batch_size, validation_seed)
            metrics = compact_metrics(val_rows, errors)
            metrics.update({"step": step, "train_loss": {k: float(np.mean(v)) for k, v in recent.items()},
                            "elapsed_seconds": accumulated_seconds + time.perf_counter() - began,
                            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3})
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            checkpoint = {"models": {k: v.state_dict() for k, v in models.items()},
                          "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                          "step": step, "config": config, "data_rng": data_rng.get_state(),
                          "noise_rng": noise_rng.get_state(), "elapsed_seconds": metrics["elapsed_seconds"]}
            save_checkpoint(run / "last.pt", checkpoint)
            recent = {name: [] for name in models}
    final_step = start_step if args.eval_only else args.steps
    save_checkpoint(run / f"weights_step{final_step:05d}.pt",
                    {"models": {k: v.state_dict() for k, v in models.items()},
                     "step": final_step, "config": config})
    counterfactuals = counterfactual_indices(val_rows, args.seed + 300)
    errors = evaluate(models, val_z, val_h, args.eval_batch_size, validation_seed, counterfactuals)
    np.savez_compressed(run / f"errors_step{final_step:05d}.npz", **errors,
                        ids=np.asarray([r["id"] for r in val_rows]),
                        shuffled_indices=counterfactuals[0].numpy(),
                        past_indices=counterfactuals[1].numpy(), past_eligible=counterfactuals[2])
    write_json(run / "validation_manifest.json", val_rows)
    summary = {"step": final_step,
               "A_vs_B": summarize_pair(val_rows, errors["A"], errors["B"]),
               "B_zero_vs_correct": summarize_pair(val_rows, errors["B_zero"], errors["B"]),
               "B_same_class_shuffled_vs_correct": summarize_pair(val_rows, errors["B_shuffled"], errors["B"])}
    eligible = counterfactuals[2]
    summary["past_control_windows"] = int(eligible.sum())
    if eligible.any():
        subset = [r for r, keep in zip(val_rows, eligible) if keep]
        summary["past_subset_A_vs_B"] = summarize_pair(subset, errors["A"][eligible], errors["B"][eligible])
        summary["B_earlier_same_video_vs_correct"] = summarize_pair(
            subset, errors["B_past"][eligible], errors["B"][eligible])
    summary["past_control_note"] = "Earlier non-overlapping A chunk from same video; distinct action phase is not annotated."
    write_json(run / f"summary_step{final_step:05d}.json", summary)
    print(json.dumps({"FINAL": summary["A_vs_B"]["primary"], "step": final_step}), flush=True)
    print(f"DONE {run}", flush=True)


if __name__ == "__main__":
    main()
