"""SparseAttention: sparse Q/K/V/O attention (torch backend, CPU/GPU).

Replaces the Transformer's attention, but each of the four projections keeps only
keep_frac of its connections and is trained from scratch. Only the float tensor
computation here uses PyTorch (as the GPU engine); all other layers are discrete.
"""
import numpy as np
from .module import Module


class SparseAttention(Module):
    def __init__(self, d, keep_frac=0.15, seed=0):
        super().__init__()
        import torch
        self.torch = torch
        self.d = d
        self.keep_frac = keep_frac
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d)
        for name in ('WQ', 'WK', 'WV', 'WO'):
            setattr(self, name, torch.tensor(
                rng.standard_normal((d, d)) * scale, dtype=torch.float32,
                requires_grad=True))
        # fixed mask
        mask = np.zeros((d, d), dtype=bool)
        k = max(1, int(d * keep_frac))
        for j in range(d):
            mask[j, rng.choice(d, size=k, replace=False)] = True
        self.mask = torch.tensor(mask)
        for name in ('WQ', 'WK', 'WV', 'WO'):
            getattr(self, name).data *= self.mask   # scale data in-place, keep leaf
        self._sparse_params = {
            name: (getattr(self, name), self.mask) for name in ('WQ', 'WK', 'WV', 'WO')
        }

    def to(self, device):
        super().to(device)
        target = self.torch.device('cuda' if device == 'cuda' else 'cpu')
        for name in ('WQ', 'WK', 'WV', 'WO'):
            setattr(self, name, getattr(self, name).to(target))
        self.mask = self.mask.to(target)
        self._sparse_params = {
            name: (getattr(self, name), self.mask) for name in ('WQ', 'WK', 'WV', 'WO')
        }
        return self

    def forward(self, X):
        """X: (T, d) torch tensor -> context representation (T, d)."""
        Q = X @ self.WQ.T
        K = X @ self.WK.T
        V = X @ self.WV.T
        scores = Q @ K.T / np.sqrt(self.d)
        probs = self.torch.softmax(scores, dim=-1)
        ctx = probs @ V
        return ctx @ self.WO.T

    def nonzeros(self):
        return int(self.mask.sum().item())
