# XCP

**XCP** is a language-model **architecture** that reaches the same macroscopic
behavior as a Transformer — text in, text out — through a completely different
implementation: a **discrete-symbolic core** plus **sparse layers**, instead of
dense continuous vectors.

The insight is that XCP and the Transformer are two implementations of the same
thing, on the same evolution path: both start from "predict the next token" and
grow toward conversation. XCP is the attempt to do that work with orders of
magnitude fewer parameters and without gradient descent as the foundation.

---

## The idea in one picture

| Transformer | XCP | Technique | Compression |
|---|---|---|---|
| Tokenizer + Embedding | `DiscreteEmbedding` | token -> integer ID -> semantic class (unsupervised clustering) | ~1000x |
| FFN | `DiscreteFFN` | class->class m-gram rules + (left-word, class)->word fact lookup + class fallback | ~112x |
| Attention | `SparseAttention` | Q/K/V/O kept at 15% of connections, trained from scratch | ~85% |
| *(no equivalent)* | `RelationGate` | discourse connectives (because/but/then) as prediction-state regulators | — |

Every layer has a uniform interface: `build()` learns, `forward()` runs.

## Two-stage training

1. **Discrete counting** (`build()`) — one pass over the corpus builds the
   vocabulary, word->class clusters, class-gram rules, and fact tables. It is a
   closed-form optimum: no gradients, no iteration, ~10 seconds on a 131k-word
   novel.
2. **Sparse gradients** (`fit_attention()`) — only the sparse attention
   projections are trained, with `SparseAdam` updating only the non-zero
   entries.

Discrete layers are integer tables (CPU, no gradients); sparse layers are float
matrices with fixed masks (GPU optional).

## Key result (Pride and Prejudice)

Corpus: *Pride and Prejudice*, 131k words, 6,982 word types.

| Model | Parameters | Top-1 next-token acc | Training |
|---|---|---|---|
| XCP (`n_classes=3200, m=4`) | **468,504 integers + 1,152 sparse floats (~0.47M)** | **0.841** | ~10 s (one-pass counting) |
| PyTorch Transformer (d=512, L=3) | 16.6M floats | 0.067 | 2,000 steps (GTX 1650) |

The parameter count is a tunable knob: `n_classes=100` collapses XCP to a few
thousand integer parameters, trading accuracy for size.

> **Honesty note.** Both models are at an early stage of fluent continuation
> with imperfect grammar. The Transformer baseline above was limited to 2,000
> steps by local compute (a single GTX 1650), which is *not* its capacity
> ceiling — so this comparison demonstrates **parameter and training
> efficiency**, not a full surpassing of language understanding. A converged,
> benchmark-fair comparison (e.g. WikiText) is planned future work.

---

## Repository structure

```
XCP/
  ├── XCP_library/       # the library: a PyTorch-mirroring API (self-implemented)
  │   ├── xcp/           #   Module base, nn layers, SparseAdam, .xcp serialization
  │   ├── train.py       #   interactive train -> save .xcp -> chat
  │   ├── tests/         #   smoke test (build -> evaluate -> attention -> save/load)
  │   └── README.md
  ├── XCP_Full Model/    # the standalone architecture implementation + tooling
  │   ├── layers.py      #   DiscreteEmbedding / DiscreteFFN / SparseAttention / RelationGate
  │   ├── model.py       #   XCPModel (composition + two-stage training)
  │   ├── run.py         #   run entrypoint (pure discrete or + attention)
  │   ├── generators/    #   synthetic corpus generators (template / semantic / held-out)
  │   ├── scripts/       #   compare vs Transformer, config sweep, figures
  │   ├── docs/          #   papers, emails, figures
  │   └── README.md
  ├── tests/             # early validation experiments (compression, capacity, attention)
  └── Data/              # corpus files
```

Two implementations of the same architecture:

- **`XCP_library/`** is the *library*: the calling convention mirrors PyTorch's
  common subset (`Module`, `nn.*`, `optim`, `save/load`), but only the sparse
  attention float ops borrow PyTorch's GPU — everything else (API, modules,
  optimizer, `.xcp` serialization) is self-implemented.
- **`XCP_Full Model/`** is the *reference implementation* in plain NumPy, with
  the corpus generators, the Transformer comparison harness, and the
  paper/figure tooling.

## Quick start

```bash
# 1. Train a model and chat with it (interactive prompts)
cd XCP_library
python train.py

# 2. Smoke test (build -> evaluate -> fit_attention -> save/load round-trip)
python tests/test_library.py

# 3. Run the reference implementation
cd "../XCP_Full Model"
python run.py            # pure discrete chain
python run.py --attn     # discrete chain + sparse attention

# 4. Compare against a PyTorch Transformer (needs GPU)
python scripts/compare.py

# 5. Sweep n_classes x m for the best accuracy
python scripts/scan.py
```

## Library usage (mirrors PyTorch)

```python
import xcp
import xcp.nn as nn

model = nn.XCPModel(n_classes=3200, m=4, use_attention=True, d=64, attention_depth=4)
model.build(tokens)                 # stage 1: discrete counting, one pass
model.fit_attention(epochs=300)     # stage 2: sparse attention gradients

model.to('cuda')                    # move sparse layers to GPU (optional)
acc, miss = model.evaluate(ids, split, len(ids))

xcp.save(model, 'model.xcp')
model2 = xcp.load('model.xcp')
```

The `.xcp` format is a zip archive: `meta.json` (architecture) + `discrete.pkl`
(integer tables) + `sparse/*.npy` (masked float matrices).

## Honest boundaries

- XCP's discrete core trades continuous smoothness for memory; its
  generalization comes from class-level fallback, not continuous interpolation.
- The head-to-head Transformer comparison is **preliminary**: the baseline is
  under-trained relative to its capacity, and no public benchmark has been run
  yet.
- The GPU backend currently uses PyTorch; once validated it can be replaced
  with hand-written CUDA kernels.

Author: **Chengping Xing** (邢承平).
