# XCP Full Model

The complete "discrete-symbolic + sparse" technical stack, mapping to the
Transformer's end-to-end architecture, but with **nowhere requiring a dense
float matrix**.

## Architecture mapping

| Transformer | XCP | Technique | Compression |
|---|---|---|---|
| Tokenizer / Embedding | `DiscreteEmbedding` | word -> integer ID -> class (neighbor clustering) | ~1000x |
| FFN | `DiscreteFFN` | structure rules + fact lookup + class fallback | ~112x |
| Attention (multi-layer) | `SparseAttention` x L | sparse Q/K/V/O 85%, refined layer by layer | 85% |
| (no equivalent) | `RelationGate` | cause/contrast regulator | 6.8x |

## End-to-end data flow

```
text -> word ID -> class ID
     -> DiscreteFFN: class->class rules predict the next class
                   -> (left-word, predicted class) -> word lookup + class fallback
     -> [SparseAttention long-range mixing, optional]
     -> [RelationGate relation regulation]
     -> output word
```

## Two-stage training

1. **Discrete (no gradient, fast)**: `build()` scans the corpus and counts (vocab, class rules, fact table, fallback table).
2. **Sparse (gradient)**: `fit_attention()` trains the sparse Q/K/V/O (only the 15% nonzero entries).

## Directory structure

```
XCP_Full Model/
  ├── layers.py       # core: layer classes (DiscreteEmbedding / DiscreteFFN / SparseAttention / RelationGate)
  ├── model.py        # core: XCPModel (composes the layers)
  ├── run.py          # core: run entrypoint
  ├── README.md
  ├── generators/     # corpus generators (generate_corpus*.py, 5 of them, output to ../../Data/)
  ├── scripts/        # tool scripts (compare / scan / collect_data / make_figures / generate_docs)
  └── docs/           # external deliverables (paper, email docx + charts.html + figure PNGs)
```

## Running

```bash
cd "XCP_Full Model"
python run.py                          # pure discrete chain
python run.py --attn                   # discrete chain + sparse attention
python scripts/scan.py                 # sweep configurations
python scripts/compare.py              # end-to-end comparison against Transformer
```

## Current status (honest)

- [x] The discrete chain (Embedding + FFN) is composed and can run next-token prediction.
- [x] Sparse attention is wired into prediction (`fit_attention` + long-range fallback).
- [x] Multi-layer attention (x L, default 2 layers, set with `--depth N`) is implemented.
- [x] The relation gate is wired into the forward pass (relation words take top prediction priority).
- [x] Real-corpus comparison (Pride and Prejudice): XCP acc 0.841 vs Transformer 0.067 (under-trained).

## End-to-end data flow (predict_next priority)

```
predict_next(t):
  0) relation gate: left word is a relation word -> return its next word directly
  1) short-range class rule: class context (m-gram) -> predict next class
  2) long-range attention fallback: class rule miss -> attention's next-class
  3) fact lookup: (left word, predicted class) -> right word; miss -> class fallback (most frequent word in class)
```

## Honesty boundary

Each layer was validated independently; the full chain was validated on a
template corpus plus one real novel. The Transformer comparison baseline is
under-trained; a fair (converged) comparison and public benchmarks have not
been done yet.
