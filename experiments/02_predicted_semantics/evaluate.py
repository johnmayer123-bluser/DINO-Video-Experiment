"""Compare frozen no-history/teacher/direct-substitution models and trained P."""

import argparse
from pathlib import Path

import torch

from conditions import (ROOT, Denoiser, load_predictor, load_split,
                        predict_semantics, save_evaluation, setup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seeds", type=int, nargs="+", choices=(42, 43, 44), default=[42, 43, 44])
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args()
    setup()
    rows, z, h = load_split(args.root, args.split)
    predictor = load_predictor(args.root)
    q = predict_semantics(predictor, z)
    del predictor
    for seed in args.seeds:
        run = (args.run_root or args.root / "runs") / f"predicted_semantics_seed{seed}"
        reference = torch.load(args.root / f"runs/history_semantics_seed{seed}/final.pt",
                               map_location="cpu", weights_only=False)
        if reference["step"] != 6000:
            raise ValueError("Use the completed 6000-step history-semantics references")
        models = {name: Denoiser().cuda() for name in ("N", "T")}
        models["N"].load_state_dict(reference["models"]["N"])
        models["T"].load_state_dict(reference["models"]["SD"])
        models["D"] = models["T"]  # Exactly the same teacher weights; only its condition changes.
        if not args.direct_only:
            checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
            if checkpoint["step"] != 6000:
                raise ValueError("P final evaluation requires step 6000")
            models["P"] = Denoiser().cuda()
            models["P"].load_state_dict(checkpoint["model"])
        run.mkdir(parents=True, exist_ok=True)
        save_evaluation(run, models, rows, z, h, q, args.split, seed, args.direct_only)


if __name__ == "__main__":
    main()
