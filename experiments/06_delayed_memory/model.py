"""Equal-parameter recent, first-cue, and gated-memory controls.

Only the FP16 16x256 state and the last FP16 latent slice survive a chunk.
Known hidden/return phase enters the denoiser's time path, never the state.
"""
import copy
import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

_spec = importlib.util.spec_from_file_location(
    'delayed_q4_width', Path(__file__).resolve().parents[1] /
    '04_history_compression/width_sweep/width_model.py')
width = importlib.util.module_from_spec(_spec)
_search_path = list(sys.path)
try:
    _spec.loader.exec_module(width)
finally:
    # Q4's loaders insert their directories; do not shadow this experiment's train.
    sys.path[:] = _search_path
ROOT = width.ROOT
ARMS = ('recent', 'first', 'memory')
precision = width.token.precision
from denoiser_model import timestep_embedding


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


class DelayedMemoryModel(width.WidthModel):
    def __init__(self, arm='memory'):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(256, denoiser=PhaseDenoiser())
        self.arm = arm
        self.update_gate = nn.Sequential(nn.Linear(512, 256), nn.SiLU(),
                                         nn.Linear(256, 256))

    def observe(self, history, old=None):
        """Consume one observed/generated chunk, then discard its full latent."""
        if history.shape[1:] != (16, 4, 24, 24):
            raise ValueError('history must be [B,16,4,24,24]')
        if old is not None and old.shape != (len(history), 16, 256):
            raise ValueError('old must be [B,16,256]')
        if self.arm == 'first' and old is not None:
            return old
        encoded = self.encode(history)
        previous = torch.zeros_like(encoded) if old is None or self.arm == 'recent' else old.float()
        gate = torch.sigmoid(self.update_gate(torch.cat((previous, encoded), -1)))
        return ((1-gate)*previous + gate*encoded).half()

    def predict(self, boundary, noisy, timestep, state, phase, dropped=None):
        """Read compact conditions; dropout never changes internal recurrence."""
        if boundary.shape != (len(noisy), 16, 1, 24, 24):
            raise ValueError('boundary must be compact [B,16,1,24,24]')
        if state.shape != (len(noisy), 16, 256):
            raise ValueError('state must be [B,16,256]')
        boundary = boundary.half()
        if dropped is not None:
            boundary = torch.where(dropped[:, None, None, None, None], 0., boundary)
            state = torch.where(dropped[:, None, None], 0., state)
        return self.denoiser(boundary.to(noisy.dtype).expand_as(noisy), noisy,
                             timestep, state, phase)


def paired_models(seed, device='cuda'):
    torch.manual_seed(seed)
    base = DelayedMemoryModel()
    return {arm: _set_arm(copy.deepcopy(base), arm).to(device) for arm in ARMS}


def _set_arm(model, arm):
    model.arm = arm
    return model


def load_model(root, seed, arm, device='cuda'):
    if arm not in ARMS:
        raise ValueError(arm)
    path = Path(root)/'runs'/f'delayed_memory_seed{seed}'/'final.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['config']['seed'] != seed:
        raise ValueError('Checkpoint seed mismatch')
    model = DelayedMemoryModel(arm)
    model.load_state_dict(checkpoint['models'][arm])
    return model.to(device).eval()
