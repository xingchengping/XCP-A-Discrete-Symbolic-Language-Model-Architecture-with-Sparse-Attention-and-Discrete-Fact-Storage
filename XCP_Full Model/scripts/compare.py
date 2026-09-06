"""End-to-end comparison: Transformer (PyTorch+GPU) vs XCP (CPU, discrete+sparse).

Transformer is the "reference yardstick", trained with PyTorch on GPU; XCP is
our architecture, on CPU. Both use the same corpus (gen_corpus.txt) and are
compared on next-token accuracy and parameter count.

Run (from the XCP_Full Model directory):
    python compare.py
"""
import os
import sys
import re
import math
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # XCP_Full Model
sys.path.insert(0, ROOT)
from model import XCPModel

CORPUS = os.path.join(os.path.dirname(os.path.dirname(ROOT)), 'Data',
                      'real', 'pride_and_prejudice.txt')


def load_words(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    words = re.findall(r"[a-z']+", text.lower())
    vocab = sorted(set(words))
    w2i = {w: i for i, w in enumerate(vocab)}
    return words, [w2i[w] for w in words], len(vocab), w2i


# ---------------------------------------------------------------------------
# PyTorch GPT-style Transformer (reference yardstick, GPU)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head=8):
        super().__init__()
        self.d, self.n_head = d, n_head
        self.dk = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(self.d, dim=2)
        q = q.view(B, T, self.n_head, self.dk).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.dk).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.o(y)


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, V, d, L, T, n_head=8):
        super().__init__()
        self.T = T
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)


def make_batches(ids, T, B):
    data = torch.tensor(ids, dtype=torch.long)
    n = (len(ids) - 1) // T
    x = data[:n * T].view(n, T)
    y = data[1:n * T + 1].view(n, T)
    return x, y


def train_gpt(model, x_tr, y_tr, x_te, y_te, steps=2000, lr=3e-4, B=32):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(dev)
    x_tr, y_tr = x_tr.to(dev), y_tr.to(dev)
    x_te, y_te = x_te.to(dev), y_te.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = len(x_tr)
    for s in range(steps):
        idx = torch.randint(0, n, (B,), device=dev)
        xb, yb = x_tr[idx], y_tr[idx]
        logits = model(xb)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    accs = []
    with torch.no_grad():
        for i in range(0, len(x_te), B):
            xb = x_te[i:i + B]
            yb = y_te[i:i + B]
            logits = model(xb)
            accs.append((logits.argmax(-1) == yb).float().mean().item())
    return float(np.mean(accs))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print('=' * 70)
    print('End-to-end comparison: Transformer(PyTorch+GPU) vs XCP(CPU)')
    print('=' * 70)

    words, ids, V, w2i = load_words(CORPUS)
    n = len(ids)
    split = int(n * 0.9)
    print(f'\ncorpus {n} words, vocab V={V}\n')

    # ---------- Transformer (~9.5M params, GPU) ----------
    d, L, T = 512, 3, 64
    print(f'[1] Transformer: d={d}, L={L}, GPU training 2000 steps...')
    gpt = GPT(V, d, L, T)
    p_gpt = count_params(gpt)
    x_all, y_all = make_batches(ids, T, 1)
    x_tr, y_tr = x_all[:int(len(x_all) * 0.9)], y_all[:int(len(y_all) * 0.9)]
    x_te, y_te = x_all[int(len(x_all) * 0.9):], y_all[int(len(y_all) * 0.9):]
    acc_tf = train_gpt(gpt, x_tr, y_tr, x_te, y_te, steps=2000)
    print(f'    test acc={acc_tf:.3f}, params={p_gpt:,}')

    # ---------- XCP (CPU) ----------
    print(f'\n[2] XCP: discrete counting + 4 sparse attention layers (CPU)...')
    model = XCPModel(n_classes=50, m=2, use_attention=True, d=64,
                     attention_depth=4)
    model.build(words)
    model.fit_attention(epochs=200, lr=0.05)
    tr_acc, _ = model.evaluate(1, split)
    te_acc, _ = model.evaluate(split, n)
    p = model.n_params()
    print(f'    train acc={tr_acc:.3f}, test acc={te_acc:.3f}')
    print(f'    params: discrete {p["int"]:,} ints + sparse {p["float"]:,} floats')

    print()
    print('=' * 70)
    print('Comparison (test-set next-token acc):')
    print(f'  Transformer : acc={acc_tf:.3f}, params={p_gpt:,} (float)')
    print(f'  XCP         : acc={te_acc:.3f}, params={p["int"]:,} ints + '
          f'{p["float"]:,} floats')
    print('=' * 70)


if __name__ == '__main__':
    main()
