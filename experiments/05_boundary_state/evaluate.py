"""Evaluate final boundary/state models, reusing Q4's 16x256 S8 checkpoint."""
import argparse
from pathlib import Path
import torch
from model import ROOT,ARMS,BoundaryModel,load_split,save_evaluation,setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--split',choices=('val','test'),default='test')
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    args=parser.parse_args();setup()
    rows,z=load_split(args.root,args.split)
    for seed in args.seeds:
        run=args.root/f'runs/boundary_state_seed{seed}'
        checkpoint=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
        reference=torch.load(args.root/f'runs/history_width_seed{seed}/final.pt',map_location='cpu',weights_only=False)
        if checkpoint['step']!=6000 or reference['step']!=6000:raise ValueError('Use final 6000-step weights')
        models={k:BoundaryModel(k).cuda() for k in ARMS}
        models['S8'].load_state_dict(reference['models']['D256'])
        for name in ('B18','H26','S26'):models[name].load_state_dict(checkpoint['models'][name])
        save_evaluation(run,models,rows,z,args.split,seed,checkpoint['step'])


if __name__=='__main__':main()
