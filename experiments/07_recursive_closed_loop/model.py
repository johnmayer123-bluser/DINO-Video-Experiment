"""Q7: fixed, latest-chunk and recursive state with identical Q6 parameters."""
import torch

from common import Q6_ROOT, WORK_ROOT, load_source

q6 = load_source('recursive_q6_model', Q6_ROOT / 'model.py')
width, precision = q6.width, q6.precision
ARMS = ('F', 'L', 'R')


class RecurrentModel(q6.ConditionModel):
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__()
        self.arm = arm

    def observe(self, chunk, old=None):
        """U(0,E) initially; subsequent behavior is the only arm difference."""
        if chunk.shape[1:] != (16, 4, 24, 24):
            raise ValueError('chunk must be [B,16,4,24,24]')
        if old is not None and old.shape != (len(chunk), 16, 256):
            raise ValueError('old must be [B,16,256]')
        if self.arm == 'F' and old is not None:
            return old
        encoded = self.encode(chunk)
        previous = (old.float() if self.arm == 'R' and old is not None
                    else torch.zeros_like(encoded))
        gate = torch.sigmoid(self.update_gate(torch.cat((previous, encoded), -1)))
        return ((1-gate)*previous + gate*encoded).half()


def load_model(arm, device='cuda', step=6000, seed=42):
    if arm not in ARMS or seed != 42:
        raise ValueError('Expected arm F/L/R and seed=42')
    name = f'final_{arm}.pt' if step == 6000 else f'step{step:05d}_{arm}.pt'
    checkpoint = torch.load(WORK_ROOT / 'runs/seed42' / name,
                            map_location='cpu', weights_only=False)
    if checkpoint['step'] != step or checkpoint['config']['seed'] != seed:
        raise ValueError('Checkpoint step/seed mismatch')
    if checkpoint['arm'] != arm:
        raise ValueError('Checkpoint arm mismatch')
    model = RecurrentModel(arm)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval()
