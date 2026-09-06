# XCP Library

XCP — a discrete-symbolic + sparse language model library. The calling convention mirrors PyTorch's common subset, but **only the float tensor computation borrows PyTorch's GPU; everything else (API, modules, optimizer, serialization) is self-implemented**.

## Directory structure

```
XCP_library/
  ├── xcp/                  # library body
  │   ├── __init__.py       # exports nn / optim / save / load
  │   ├── io.py             # .xcp model format (zip + json + pickle + npy)
  │   ├── nn/
  │   │   ├── module.py     # Module base class (mirrors torch.nn.Module)
  │   │   ├── embedding.py  # DiscreteEmbedding (token->ID->class)
  │   │   ├── ffn.py        # DiscreteFFN (class rules + fact lookup + class fallback)
  │   │   ├── attention.py  # SparseAttention (sparse Q/K/V/O, torch backend)
  │   │   ├── relation.py   # RelationGate (discourse regulator)
  │   │   └── model.py      # XCPModel (composition)
  │   └── optim/
  │       └── sparse_adam.py# SparseAdam + CosineAnnealingLR
  └── tests/
      └── test_library.py   # smoke test
```

## Usage (mirrors PyTorch)

```python
import xcp
import xcp.nn as nn

# build a model (like torch)
model = nn.XCPModel(n_classes=50, m=4, use_attention=True, d=64, attention_depth=4)

# training: two stages
model.build(tokens)                    # stage 1: discrete counting (one pass, no gradients)
model.fit_attention(epochs=300)        # stage 2: sparse attention gradients

# move to GPU (only sparse layers)
model.to('cuda')

# evaluate
acc, miss = model.evaluate(ids, split, len(ids))

# save / load (.xcp format)
xcp.save(model, 'model.xcp')
model2 = xcp.load('model.xcp')
```

## Correspondence with PyTorch

| PyTorch | XCP |
|---|---|
| `torch.nn.Module` | `xcp.nn.Module` |
| `nn.Embedding` | `nn.DiscreteEmbedding` (integer ID, not a float vector) |
| `nn.MultiheadAttention` | `nn.SparseAttention` (85% sparse) |
| `nn.Linear` (FFN) | `nn.DiscreteFFN` (class rules + fact lookup) |
| `torch.optim.Adam` | `xcp.optim.SparseAdam` (updates only mask nonzeros) |
| `torch.save/load` (.pt) | `xcp.save/load` (.xcp) |

## Design notes

- **Two parameter kinds**: discrete parameters (integer tables, CPU-only, no gradients) + sparse parameters (float matrices + fixed masks, GPU optional).
- **Two training stages**: discrete layers use counting (closed-form optimum, one pass); sparse layers use gradients (`SparseAdam` updates only nonzeros).
- **GPU boundary**: only `SparseAttention`'s float ops go through PyTorch CUDA; discrete layers (lookup/counting) have no GPU concept.
- **.xcp format**: `meta.json` (architecture) + `discrete.pkl` (integer tables) + `sparse/*.npy` (sparse matrix values + masks).

## Run tests

```bash
cd XCP_library
python tests/test_library.py
```

## Honest boundaries

This is the first version: the core API works, CPU/GPU switching works, .xcp round-trip works. Not yet implemented: batched parallel training, a data loader, more layers (e.g. stacked multi-layer FFN). The GPU backend currently uses PyTorch; once validated it can be replaced with hand-written CUDA kernels.
