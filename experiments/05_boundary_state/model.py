"""Compressed history, last latent slice, and their equal-budget comparison."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('q4_width_model', Path(__file__).resolve().parents[1]/'04_history_compression/width_sweep/width_model.py')
width = importlib.util.module_from_spec(spec)
spec.loader.exec_module(width)
ROOT = width.ROOT
load_split, setup = width.load_split, width.setup
save_checkpoint, write_json = width.save_checkpoint, width.write_json
compact_metrics = width.compact_metrics
LEVELS = width.token.NOISE_LEVELS
ARMS = ('S8', 'B18', 'H26', 'S26')
WIDTHS = {'S8':256, 'B18':256, 'H26':256, 'S26':832}
KIB = {'S8':8, 'B18':18, 'H26':26, 'S26':26}


class BoundaryModel(width.WidthModel):
    def __init__(self, kind, denoiser=None, encoder=None, initial_width=None):
        super().__init__(WIDTHS[kind] if initial_width is None else initial_width, denoiser, encoder)
        self.set_kind(kind)

    def set_kind(self, kind):
        self.kind = kind
        self.encoder.requires_grad_(kind != 'B18')

    def conditions(self, history, dropped=None):
        if self.kind == 'B18':
            state = torch.zeros(len(history),16,self.width,device=history.device,dtype=torch.float16)
        else:
            state = self.encode(history).half()
        if self.kind in ('B18','H26'):
            # Store one FP16 slice; broadcasting is only for the current forward pass.
            boundary = history[:,:,-1:].half().to(history.dtype).expand_as(history)
        else:
            boundary = torch.zeros_like(history)
        if dropped is not None:
            state = width.token.apply_dropout(state,dropped)
            boundary = torch.where(dropped[:,None,None,None,None],torch.zeros_like(boundary),boundary)
        return boundary,state

    def forward(self, history, noisy, timestep, dropped=None):
        boundary,state = self.conditions(history,dropped)
        return self.denoiser(boundary,noisy,timestep,state)


def paired_models(seed, device='cuda'):
    torch.manual_seed(seed)
    denoiser = width.token.Denoiser().to(device)
    torch.manual_seed(seed+300)
    encoder = width.token.SemanticHead().to(device)
    base = BoundaryModel('S26',denoiser,encoder,initial_width=1024)
    models = {k:copy.deepcopy(base) for k in ('B18','H26','S26')}
    for name,model in models.items():
        width.narrow(model,WIDTHS[name])
        model.set_kind(name)
    return models


@torch.inference_mode()
def evaluate(models, latent, split, repeats=4, batch_size=16):
    device = next(next(iter(models.values())).parameters()).device
    modes = {k:m.training for k,m in models.items()}
    repeated,frames = {k:[] for k in models},{k:[] for k in models}
    try:
        for model in models.values(): model.eval()
        for repeat in range(repeats):
            errors,frame_errors = {k:[] for k in models},{k:[] for k in models}
            for start in range(0,len(latent),batch_size):
                z = latent[start:start+batch_size].to(device,non_blocking=True).float()
                history,future = z[:,:,:4],z[:,:,4:]
                with width.token.precision(device):
                    conditions = {k:m.conditions(history) for k,m in models.items()}
                noise = width.token.evaluation_noise(future.shape,start,repeat,10042 if split=='val' else 20042).to(device)
                target = future-noise
                batch,by_frame = {k:[] for k in models},{k:[] for k in models}
                for level in LEVELS:
                    t = torch.full((len(z),),level,device=device)
                    noisy = (1-level)*noise+level*future
                    for name,model in models.items():
                        boundary,state = conditions[name]
                        with width.token.precision(device): output = model.denoiser(boundary,noisy,t,state)
                        square = (output.float()-target).square()
                        batch[name].append(square.flatten(1).mean(1).cpu())
                        by_frame[name].append(square.mean((1,3,4)).cpu())
                for name in models:
                    errors[name].append(torch.stack(batch[name],1))
                    frame_errors[name].append(torch.stack(by_frame[name],1))
            for name in models:
                repeated[name].append(torch.cat(errors[name]).numpy())
                frames[name].append(torch.cat(frame_errors[name]).numpy())
        repeated = {k:np.stack(v) for k,v in repeated.items()}
        frames = {k:np.stack(v) for k,v in frames.items()}
        return {k:v.mean(0) for k,v in repeated.items()},repeated,frames
    finally:
        for name,model in models.items(): model.train(modes[name])


def save_evaluation(run,models,rows,z,split,seed,step):
    errors,repeated,frames = evaluate(models,z,split)
    stem = f'step{step:05d}_{split}'
    np.savez_compressed(run/f'errors_{stem}.npz',**errors,
        **{k+'_repeats':v for k,v in repeated.items()},
        **{k+'_by_frame_repeats':v for k,v in frames.items()},
        ids=np.asarray([r['id'] for r in rows]),noise_levels=np.asarray(LEVELS),
        noise_seed=10042 if split=='val' else 20042,noise_repeats=4)
    summary = {'seed':seed,'step':step,'split':split,'metrics':compact_metrics(rows,errors),
        'first':compact_metrics(rows,{k:v.mean(0)[...,0] for k,v in frames.items()}),
        'later':compact_metrics(rows,{k:v.mean(0)[...,1:].mean(-1) for k,v in frames.items()}),
        'state_kib':KIB,'noise_seed':10042 if split=='val' else 20042,'noise_repeats':4}
    write_json(run/f'{split}_manifest.json',rows)
    write_json(run/f'summary_{stem}.json',summary)
    print(json.dumps(summary),flush=True)
