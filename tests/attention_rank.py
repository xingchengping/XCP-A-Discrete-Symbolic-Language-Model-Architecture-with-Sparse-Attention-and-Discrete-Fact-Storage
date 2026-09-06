"""实测 rank(q_l @ k_l.T) —— 决定注意力层压缩上限的那个数。

理论：注意力分数 scores[i,j] = hn[i]·(q_l @ k_l.T)·hn[j]，所以 P = q_l @ k_l.T
决定注意力的表达力。稀疏压缩要求 s ≥ rank(P)/d。旧测量只测了单个 q/k 矩阵的
秩（~0.2d），没测 P = q_l@k_l.T 的秩。本脚本训练参考 Transformer 后直接测。

复用 history/ 的训练框架（verify_direction_ssm.Transformer + train_min）。

Run:  python tests/attention_rank.py
"""
import os
import sys
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HIST = os.path.join(ROOT, 'history')
sys.path.insert(0, ROOT)
sys.path.insert(0, HIST)

from verify_direction_ssm import load_corpus, make_batch, softmax, Transformer


class AdamDecay:
    def __init__(self, lr=3e-3, steps=400):
        self.lr0, self.steps = lr, steps
        self.b1, self.b2, self.eps = 0.9, 0.95, 1e-8
        self.t, self.m, self.v = 0, {}, {}

    def lr(self):
        return self.lr0 * 0.5 * (1.0 + math.cos(math.pi * self.t / self.steps))

    def step(self, params, grads, clip=1.0):
        self.t += 1
        lr = self.lr()
        gnorm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        scale = min(1.0, clip / max(gnorm, clip))
        for k in params:
            if k not in grads:
                continue
            g = grads[k] * scale
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)


def train_min(model, data, B, T, steps, V):
    opt = AdamDecay(lr=3e-3, steps=steps)
    rng_t = np.random.default_rng(7)
    best = 1e9
    for s in range(steps):
        x, y = make_batch(data, B, T, rng_t)
        logits, cache = model.forward(x)
        logp = softmax(logits.reshape(-1, V))
        loss = -np.mean(np.log(np.maximum(logp[np.arange(B * T), y.reshape(-1)], 1e-12)))
        dlogits = (logp - np.eye(V)[y.reshape(-1)]).reshape(B, T, V) / (B * T)
        grads = model.backward(cache, dlogits)
        opt.step(model.p, grads)
        best = min(best, float(loss))
    return best


def eff_rank(M, thresh=0.9):
    s = np.linalg.svd(np.asarray(M), compute_uv=False)
    total = np.sum(s ** 2)
    if total <= 0:
        return 0
    acc = 0.0
    for i in range(1, len(s) + 1):
        acc += s[i - 1] ** 2
        if acc / total >= thresh:
            return i
    return len(s)


def qk_rank(tf):
    """每层 P = q_l @ k_l.T 的有效秩，返回均值与列表。"""
    rs = [eff_rank(tf.p[f'q_{l}'] @ tf.p[f'k_{l}'].T) for l in range(tf.L)]
    return float(np.mean(rs)), rs


def main():
    print('=' * 76)
    print('实测 rank(q_l @ k_l.T)：决定注意力层压缩上限')
    print('=' * 76)

    data, stoi = load_corpus()
    V = len(stoi)
    T, B, steps, L = 32, 16, 400, 2

    print(f'\n语料 V={V}，T={T}，steps={steps}（cosine 衰减），L={L} 层')
    print(f'（测小 d：对应多头每头 d_k=d/h。若小 d 时 rank(P)≈d=满秩，多头无压缩空间；')
    print(f'  若 rank(P) 饱和≈常数，多头每头有压缩空间）\n')
    print(f'{"d":>4} {"loss":>7} | {"rank(P)":>8} {"r/d":>7}')
    print('-' * 40)

    for d in (16, 32):
        tf = Transformer(V, d, L, T, np.random.RandomState(2))
        loss = train_min(tf, data, B, T, steps, V)
        r, rs = qk_rank(tf)
        r_d = r / d
        print(f'{d:>4} {loss:>7.3f} | {r:>8.1f} {r_d:>7.3f}')

    print()
    print('=' * 76)
    print('读法：r/d 决定稀疏压缩上限（s ≥ r/d）。')
    print('  ✓ = 该压缩率下理论可行；✗ = 不够。')
    print('  注意：这是单头 d≤128 小模型；大模型多头结论需进一步确认。')
    print('=' * 76)


if __name__ == '__main__':
    main()
