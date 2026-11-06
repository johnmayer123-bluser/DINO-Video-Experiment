"""First-cue condition ablation with the existing compact 3D U-Net architecture."""
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

from common import REPO_ROOT, WORK_ROOT

_spec = importlib.util.spec_from_file_location(
    'keep_q4_width', REPO_ROOT / 'experiments/04_history_compression/width_sweep/width_model.py')
width = importlib.util.module_from_spec(_spec)
_search_path = list(sys.path)
_common = sys.modules.pop('common', None)
try:
    _spec.loader.exec_module(width)
finally:
    sys.path[:] = _search_path
    sys.modules.pop('common', None)
    if _common is not None:
        sys.modules['common'] = _common
from denoiser_model import timestep_embedding

ROOT = REPO_ROOT
ARMS = ('N', 'C')


def precision(device):
    return width.token.precision(torch.device(device))


class PhaseDenoiser(width.token.Denoiser):
    def __init__(self):
        super().__init__()
        self.phase_embedding = nn.Embedding(2, self.time_width)
        nn.init.normal_(self.phase_embedding.weight, std=.02)

    def forward(self, history, noisy, timestep, semantics, phase):
        if history.shape != noisy.shape or noisy.shape[1:] != (16, 4, 24, 24):
            raise ValueError('Expected matching [B,16,4,24,24] denoiser inputs')
        phase = torch.as_tensor(phase, device=noisy.device, dtype=torch.long)
        phase = phase.expand(len(noisy)) if phase.ndim == 0 else phase
        if phase.shape != (len(noisy),):
            raise ValueError('phase must be scalar or [B]')
        time = self.time_mlp(timestep_embedding(timestep, self.time_width))
        time = time + self.phase_embedding(phase)
        a = self.encoder_24(self.input(torch.cat((history, noisy), 1)), time)
        b = self.encoder_12(self.down_12(a), time)
        c = self.bottleneck_in(self.down_6(b), time)
        c = self.bottleneck_out(self.semantic_attention(c, semantics), time)
        b_out = self.decoder_12(torch.cat((self.up_12(c), b), 1), time)
        a_out = self.decoder_24(torch.cat((self.up_24(b_out), a), 1), time)
        return self.output(F.silu(self.output_norm(a_out)))


class ConditionModel(width.WidthModel):
    def __init__(self):
        super().__init__(256, denoiser=PhaseDenoiser())
        self.update_gate = nn.Sequential(nn.Linear(512, 256), nn.SiLU(),
                                         nn.Linear(256, 256))

    def observe(self, cue):
        """The original first update U(0,E); called exactly once per target."""
        if cue.shape[1:] != (16, 4, 24, 24):
            raise ValueError('cue must be [B,16,4,24,24]')
        encoded = self.encode(cue)
        previous = torch.zeros_like(encoded)
        gate = torch.sigmoid(self.update_gate(torch.cat((previous, encoded), -1)))
        return (gate * encoded).half()

    def predict(self, boundary, noisy, timestep, state, phase=1, dropped=None):
        if boundary.shape != (len(noisy), 16, 1, 24, 24):
            raise ValueError('boundary must be [B,16,1,24,24]')
        if state.shape != (len(noisy), 16, 256):
            raise ValueError('state must be [B,16,256]')
        boundary = boundary.half()
        if dropped is not None:
            boundary = torch.where(dropped[:, None, None, None, None], 0., boundary)
            state = torch.where(dropped[:, None, None], 0., state)
        return self.denoiser(boundary.to(noisy.dtype).expand_as(noisy), noisy,
                             timestep, state, phase)


def load_model(arm, device='cuda', step=6000, seed=42):
    if arm not in ARMS:
        raise ValueError(arm)
    if seed != 42:
        raise ValueError('This experiment fixes seed=42')
    run = WORK_ROOT / 'runs/seed42'
    path = run / (f'final_{arm}.pt' if step == 6000 else f'step{step:05d}_{arm}.pt')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['step'] != step or checkpoint['config']['seed'] != 42:
        raise ValueError('Checkpoint step/seed mismatch')
    model = ConditionModel()
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()
