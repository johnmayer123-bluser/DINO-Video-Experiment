"""Evaluate final paired checkpoints on validation or test with four fixed noises."""

import argparse
from pathlib import Path

import torch

from common import ARMS, ROOT, Denoiser, load_split, save_evaluation, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    setup()
    rows, z, h = load_split(args.root, args.split)
    for seed in args.seeds:
        run = (args.run_root or args.root / "runs") / f"history_semantics_seed{seed}"
        checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        models = {name: Denoiser().cuda() for name in ARMS}
        for name, model in models.items():
            model.load_state_dict(checkpoint["models"][name])
        save_evaluation(run, models, rows, z, h, args.split, checkpoint["step"], seed, args.batch_size)


if __name__ == "__main__":
    main()
