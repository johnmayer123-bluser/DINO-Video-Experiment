"""Equal-parameter recent-only and recurrent chunk states; no teacher features."""

import copy
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    'q6_width_model', Path(__file__).resolve().parents[1] /
    '04_history_compression/width_sweep/width_model.py')
width = importlib.util.module_from_spec(spec)
spec.loader.exec_module(width)
ROOT = width.ROOT
LEVELS = width.token.NOISE_LEVELS
setup, write_json = width.setup, width.write_json
save_checkpoint, compact_metrics = width.save_checkpoint, width.compact_metrics


def load_split(root, split):
    """Return manifest rows and CPU [N,16,12,24,24] latents without anchor."""
    cache = Path(root) / 'cache/ucf6_12fps_192_abc'
    rows = [r for r in json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
            if r['split'] == split]
    if not rows:
        raise ValueError(f'Empty split: {split}')
    latents = []
    for row in rows:
        z = torch.load(cache/'vae'/f"{row['id']}.pt", map_location='cpu',
                       weights_only=True)['latent']
        if z.shape != (16, 13, 24, 24) or not z.is_floating_point() or not torch.isfinite(z).all():
            raise ValueError(f"Invalid ABC VAE latent: {row['id']}")
        latents.append(z[:, 1:].clone())
    z = torch.stack(latents)
    # CPU-only PyTorch cannot allocate pinned memory; CUDA training always pins.
    return rows, z.pin_memory() if torch.cuda.is_available() else z


class RecurrentModel(width.WidthModel):
    def __init__(self, kind='recent', denoiser=None, encoder=None):
        if kind not in ('recent', 'recurrent'):
            raise ValueError(f'Unknown state kind: {kind}')
        super().__init__(256, denoiser, encoder)
        self.kind = kind
        # One cell shared by all 16 token positions and both chunk updates.
        self.gru = nn.GRUCell(256, 256).to(next(self.encoder.parameters()).device)

    def update(self, encoded, old=None):
        if encoded.ndim != 3 or encoded.shape[1:] != (16, 256):
            raise ValueError('Expected encoded state [batch,16,256]')
        if old is None or self.kind == 'recent':
            old = torch.zeros_like(encoded)
        if old.shape != encoded.shape:
            raise ValueError('Previous state shape must match encoded tokens')
        # FP16 is the actual state interface. Neither cast severs the BPTT graph.
        state = self.gru(encoded.float().reshape(-1, 256), old.float().reshape(-1, 256))
        return state.reshape_as(encoded).half()

    def states(self, a, b):
        m_a = self.update(self.encode(a))
        m_b = self.update(self.encode(b), m_a)
        return m_a, m_b

    def predict(self, history, noisy, t, state, dropped=None):
        boundary = history[:, :, -1:].half().to(history.dtype).expand_as(history)
        if dropped is not None:
            state = width.token.apply_dropout(state, dropped)
            boundary = torch.where(dropped[:, None, None, None, None],
                                   torch.zeros_like(boundary), boundary)
        return self.denoiser(boundary, noisy, t, state)


def paired_models(seed, device='cuda'):
    torch.manual_seed(seed)
    denoiser = width.token.Denoiser().to(device)
    torch.manual_seed(seed+300)
    encoder = width.token.SemanticHead().to(device)
    # Reset the cell seed explicitly so its initialization is reproducible and
    # independent of future encoder implementation details.
    torch.manual_seed(seed+500)
    base = RecurrentModel('recent', denoiser, encoder)
    models = {name: copy.deepcopy(base) for name in ('recent', 'recurrent')}
    for name, model in models.items():
        model.kind = name
    return models
