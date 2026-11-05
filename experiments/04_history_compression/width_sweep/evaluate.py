"""Evaluate all four widths, reusing the existing K16 model at width 1024."""

import argparse
from pathlib import Path

import torch

from width_model import ROOT,WidthModel,load_split,save_evaluation,setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--run-root",type=Path,default=None)
    parser.add_argument("--split",choices=("val","test"),default="val")
    parser.add_argument("--seeds",type=int,nargs="+",choices=(42,43,44),default=[42,43,44])
    args=parser.parse_args()
    setup()
    rows,z=load_split(args.root,args.split)
    for seed in args.seeds:
        run=(args.run_root or args.root / "runs") / f"history_width_seed{seed}"
        checkpoint=torch.load(run / "final.pt",map_location="cpu",weights_only=False)
        reference=torch.load(args.root / f"runs/history_compression_seed{seed}/final.pt",map_location="cpu",weights_only=False)
        if checkpoint["step"]!=6000 or reference["step"]!=6000:raise ValueError("Evaluate completed 6000-step checkpoints")
        models={f"D{d}":WidthModel(d).cuda() for d in (1024,512,256,128)}
        models["D1024"].load_state_dict(reference["models"]["K16"])
        for name in ("D512","D256","D128"):models[name].load_state_dict(checkpoint["models"][name])
        save_evaluation(run,models,rows,z,args.split,seed,checkpoint["step"])


if __name__=="__main__":main()
