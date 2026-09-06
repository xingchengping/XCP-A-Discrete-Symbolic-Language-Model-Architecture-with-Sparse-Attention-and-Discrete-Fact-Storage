"""The Module base class of the XCP library, mirroring torch.nn.Module.

Sub-modules are auto-registered; provides to(device), train/eval, and
state_dict/load_state_dict. XCP has two kinds of parameters: discrete parameters
(integer tables, CPU-only) and sparse parameters (float matrices with masks).
"""
import os
import json
import pickle
import zipfile
import numpy as np


class Module:
    """Base class for all XCP layers, matching the common torch.nn.Module subset."""

    def __init__(self):
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_training', True)
        object.__setattr__(self, '_device', 'cpu')

    # ---- auto-register sub-modules (like PyTorch) ----
    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # ---- traversal ----
    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def named_modules(self, prefix=''):
        yield prefix, self
        for name, m in self._modules.items():
            yield from m.named_modules(prefix + name + '.')

    # ---- device / training mode ----
    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = device
        for m in self._modules.values():
            m.to(device)
        return self

    def train(self, mode=True):
        self._training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def training(self):
        return self._training

    # ---- sparse float parameters (for the optimizer) ----
    def sparse_parameters(self):
        """Recursively collect all sparse float parameters (as {'WQ': (values, mask)})."""
        for m in self.modules():
            for name, (values, mask) in getattr(m, '_sparse_params', {}).items():
                yield m, name, values, mask

    # ---- state (for .xcp saving) ----
    def state_dict(self):
        """Return a serializable state: discrete tables + sparse matrices."""
        sd = {'_discrete': {}, '_sparse': {}, '_meta': {}}
        for name, m in self.named_modules():
            if getattr(m, '_discrete', None):
                sd['_discrete'][name] = m._discrete
            if getattr(m, '_sparse_params', None):
                sd['_sparse'][name] = {
                    k: (np.asarray(v), np.asarray(mask))
                    for k, (v, mask) in m._sparse_params.items()
                }
        return sd

    def load_state_dict(self, sd):
        for name, m in self.named_modules():
            if name in sd.get('_discrete', {}):
                m._discrete = sd['_discrete'][name]
            if name in sd.get('_sparse', {}):
                for k, (v, mask) in sd['_sparse'][name].items():
                    m._sparse_params[k] = (np.asarray(v), np.asarray(mask))
        return self
