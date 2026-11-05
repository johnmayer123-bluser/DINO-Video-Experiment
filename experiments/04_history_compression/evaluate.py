"""Compare learned compressed states with the existing 64-token Q3 control."""

import argparse
from pathlib import Path

import torch

from model import ROOT, CompressionModel, load_split, save_evaluation, setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--run-root",type=Path,default=None)
    parser.add_argument("--split",choices=("val","test"),default="val")
    parser.add_argument("--seeds",type=int,nargs="+",choices=(42,43,44),default=[42,43,44])
    args=parser.parse_args()
    setup()
    rows,z=load_split(args.root,args.split)
    for seed in args.seeds:
        run=(args.run_root or args.root / "runs") / f"history_compression_seed{seed}"
        checkpoint=torch.load(run / "final.pt",map_location="cpu",weights_only=False)
        reference=torch.load(args.root / f"runs/semantic_supervision_seed{seed}/final.pt",map_location="cpu",weights_only=False)
        if checkpoint["step"]!=6000 or reference["step"]!=6000:
            raise ValueError("Evaluate completed 6000-step checkpoints")
        models={f"K{k}":CompressionModel(k).cuda() for k in (64,16,4,1)}
        models["K64"].load_state_dict(reference["models"]["C"])
        for name in ("K16","K4","K1"):
            models[name].load_state_dict(checkpoint["models"][name])
        save_evaluation(run,models,rows,z,args.split,seed,checkpoint["step"])


if __name__ == "__main__":
    main()
