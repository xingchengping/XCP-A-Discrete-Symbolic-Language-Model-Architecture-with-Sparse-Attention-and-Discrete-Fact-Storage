"""XCP: a discrete-symbolic + sparse language model library.

The calling convention mirrors the common subset of PyTorch:
    import xcp
    import xcp.nn as nn
    import xcp.optim as optim

    model = nn.XCPModel(n_classes=50, m=4, use_attention=True, d=64)
    model.build(tokens)                    # discrete counting (one pass)
    model.fit_attention(epochs=300)        # sparse attention gradient (GPU optional)
    model.to('cuda')                       # move sparse layers to GPU
    acc, miss = model.evaluate(ids, split, len(ids))

    model = xcp.load('model.xcp')          # load a .xcp model
    xcp.save(model, 'model.xcp')
"""
from . import nn
from . import optim
from .io import save, load

__all__ = ['nn', 'optim', 'save', 'load']
__version__ = '0.1.0'
