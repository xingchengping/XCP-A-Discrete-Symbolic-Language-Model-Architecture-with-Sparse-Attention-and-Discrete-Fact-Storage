"""Collect chart data: training-loss trajectory + generalization gain across class counts, output JSON.

Curve 1: Transformer training loss vs steps (GPU)
Curve 4: XCP test accuracy vs class count R (within-class generalization gain, v5 corpus)

Run: python collect_data.py > chart_data.json
"""
import os
import sys
import re
import math
import json
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # XCP_Full Model
sys.path.insert(0, ROOT)
from compare import GPT, make_batches, load_words
from model import XCPModel

CORPUS = os.path.join(os.path.dirname(os.path.dirname(ROOT)), 'Data',
                      'gen_corpus_sem2.txt')


def collect_loss():
    words, ids, V, w2i = load_words(CORPUS)
    d, L, T = 512, 3, 64
    gpt = GPT(V, d, L, T).cuda()
    x, y = make_batches(ids, T, 1)
    n = len(x)
    x_tr, y_tr = x[:int(n * 0.9)].cuda(), y[:int(n * 0.9)].cuda()
    opt = torch.optim.AdamW(gpt.parameters(), lr=3e-4, weight_decay=0.01)
    losses = []
    for s in range(2000):
        idx = torch.randint(0, len(x_tr), (32,), device='cuda')
        xb, yb = x_tr[idx], y_tr[idx]
        logits = gpt(xb)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                           yb.view(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            losses.append([s, round(loss.item(), 4)])
    return losses


def collect_gain():
    words, ids, V, w2i = load_words(CORPUS)
    n = len(ids)
    split = int(n * 0.9)
    gains = []
    for R in (3, 5, 8, 12, 16, 24, 32):
        m = XCPModel(n_classes=R, m=2, use_attention=False)
        m.build(words)
        te_acc, _ = m.evaluate(split, n)
        gains.append([R, round(te_acc, 4)])
    return gains


def collect_xcp_loss():
    words, ids, V, w2i = load_words(CORPUS)
    m = XCPModel(n_classes=10, m=2, use_attention=True, d=64, attention_depth=4)
    m.build(words)
    m.fit_attention(epochs=200, lr=0.05)
    return m.train_losses


if __name__ == '__main__':
    losses = collect_loss()
    xcp_losses = collect_xcp_loss()
    gains = collect_gain()
    data = {
        'loss_tf': losses,
        'loss_xcp': xcp_losses,
        'gain': gains,
        'acc_comparison': [
            {'corpus': 'simple(64w)', 'transformer': 0.381, 'xcp': 0.332},
            {'corpus': 'hard(447w)', 'transformer': 0.041, 'xcp': 0.046},
            {'corpus': 'semantic(clean)', 'transformer': 0.096, 'xcp': 0.107},
        ],
        'param_scaling': [
            {'d': 64, 'transformer': 98304, 'xcp': 8192},
            {'d': 128, 'transformer': 786432, 'xcp': 32768},
            {'d': 256, 'transformer': 4718592, 'xcp': 157286},
            {'d': 512, 'transformer': 25165824, 'xcp': 629146},
            {'d': 1024, 'transformer': 150994944, 'xcp': 3145728},
        ],
    }
    print(json.dumps(data))
