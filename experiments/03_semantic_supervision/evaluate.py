"""Fixed final C/J evaluation, with existing N and frozen-predictor P references."""

import argparse
from pathlib import Path

import torch

from joint import (ROOT, Denoiser, JointModel, SemanticHead, load_split,
                   save_evaluation, setup)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(42, 52), default=list(range(42, 52)))
    args = parser.parse_args()
    setup()
    rows, z, h = load_split(args.root, args.split)
    predictor = None
    for seed in args.seeds:
        run = (args.run_root or args.root / "runs") / f"semantic_supervision_seed{seed}"
        checkpoint = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        if checkpoint["step"] != 6000:
            raise ValueError("Evaluate completed 6000-step checkpoints")
        models = {}
        for name, directory in (("N", "history_semantics"), ("P", "predicted_semantics")):
            path = args.root / f"runs/{directory}_seed{seed}/final.pt"
            if not path.exists():
                continue
            reference = torch.load(path, map_location="cpu", weights_only=False)
            if reference["step"] != 6000:
                raise ValueError("Evaluate completed 6000-step reference checkpoints")
            if name == "P" and predictor is None:
                q_checkpoint = torch.load(args.root / "runs/semantic_head_seed42/best.pt", map_location="cpu", weights_only=False)
                if q_checkpoint["step"] != 2800:
                    raise ValueError("P reference uses fixed semantic_head_seed42 best step 2800")
                predictor = SemanticHead().cuda().eval().requires_grad_(False)
                predictor.load_state_dict(q_checkpoint["model"])
            models[name] = Denoiser().cuda()
            models[name].load_state_dict(reference["models"]["N"] if name == "N" else reference["model"])
        for name in ("C", "J"):
            models[name] = JointModel().cuda()
            models[name].load_state_dict(checkpoint["models"][name])
        save_evaluation(run, models, rows, z, h, predictor, args.split, seed, checkpoint["step"])


if __name__ == "__main__":
    main()
