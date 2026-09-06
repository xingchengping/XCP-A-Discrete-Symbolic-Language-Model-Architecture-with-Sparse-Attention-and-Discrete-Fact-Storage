"""端到端验证：全层压缩（attention 85% + FFN 95%）后整体 loss 是否不涨。

这是从「逐层省参」到「完整模型 12x」的最后一步。复用 history/ 的完整
Transformer（embedding + L 层 [attention + FFN] + 输出头），对比：
  [满]     全可学习
  [压缩]    attention Q/K/V/O 稀疏 85% + FFN ff1 稀疏 90% + ff2 冻结随机基

关键看：压缩模型的 loss 是否 ≈ 满模型（不涨 = 12x 成立）。

Run:  python tests/end2end.py
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
    def __init__(self, lr=3e-3, steps=600):
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


def apply_compression(tf, attn_keep, ffn_keep, emb_keep=None, seed=0):
    """稀疏 attention + 稀疏/冻结 FFN + 可选稀疏 embedding，返回 (masks, frozen)。"""
    rng = np.random.default_rng(seed)
    d = tf.d
    masks = {}
    frozen = set()
    for l in range(tf.L):
        for name in ('q', 'k', 'v', 'o'):
            m = np.zeros((d, d), dtype=bool)
            k = max(1, int(d * attn_keep))
            for j in range(d):
                m[j, rng.choice(d, size=k, replace=False)] = True
            masks[f'{name}_{l}'] = m
            tf.p[f'{name}_{l}'] = tf.p[f'{name}_{l}'] * m
        # ff1 稀疏
        m1 = np.zeros((d, 4 * d), dtype=bool)
        k1 = max(1, int(4 * d * ffn_keep))
        for j in range(d):
            m1[j, rng.choice(4 * d, size=k1, replace=False)] = True
        masks[f'ff1_{l}'] = m1
        tf.p[f'ff1_{l}'] = tf.p[f'ff1_{l}'] * m1
        # ff2 冻结随机基
        B = rng.standard_normal((4 * d, d))
        B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
        tf.p[f'ff2_{l}'] = B
        frozen.add(f'ff2_{l}')
    # embedding / 输出头 稀疏（可选）
    if emb_keep is not None:
        V, d = tf.p['emb'].shape
        m_emb = np.zeros((V, d), dtype=bool)
        for j in range(V):
            m_emb[j, rng.choice(d, size=max(1, int(d * emb_keep)), replace=False)] = True
        masks['emb'] = m_emb
        tf.p['emb'] = tf.p['emb'] * m_emb
        m_out = np.zeros((d, V), dtype=bool)
        for j in range(d):
            m_out[j, rng.choice(V, size=max(1, int(V * emb_keep)), replace=False)] = True
        masks['out'] = m_out
        tf.p['out'] = tf.p['out'] * m_out
    return masks, frozen


def train(tf, data, B, T, steps, V, masks=None, frozen=None):
    opt = AdamDecay(lr=3e-3, steps=steps)
    rng_t = np.random.default_rng(7)
    best = 1e9
    for s in range(steps):
        x, y = make_batch(data, B, T, rng_t)
        logits, cache = tf.forward(x)
        logp = softmax(logits.reshape(-1, V))
        loss = -np.mean(np.log(np.maximum(logp[np.arange(B * T), y.reshape(-1)], 1e-12)))
        dlogits = (logp - np.eye(V)[y.reshape(-1)]).reshape(B, T, V) / (B * T)
        grads = tf.backward(cache, dlogits)
        if masks:
            for k in masks:
                if k in grads:
                    grads[k] = grads[k] * masks[k]
        if frozen:
            for k in frozen:
                grads.pop(k, None)
        opt.step(tf.p, grads)
        best = min(best, float(loss))
    return best


def n_params(tf, masks=None, frozen=None):
    """真实可训练参数（float 数）。"""
    if masks is None:
        return sum(int(np.size(v)) for v in tf.p.values())
    n = 0
    for k, v in tf.p.items():
        if frozen and k in frozen:
            continue
        if k in masks:
            n += int(masks[k].sum())
        else:
            n += int(np.size(v))
    return n


def main():
    print('=' * 74)
    print('端到端验证：全层压缩（attention 85% + FFN 95%）loss 是否不涨')
    print('=' * 74)

    data, stoi = load_corpus()
    V = len(stoi)
    T, B, steps, L, d = 32, 16, 400, 4, 256

    print(f'\n语料 V={V}，T={T}，L={L}，d={d}，steps={steps}\n')

    # 满模型
    tf_full = Transformer(V, d, L, T, np.random.RandomState(2))
    loss_full = train(tf_full, data, B, T, steps, V)
    p_full = n_params(tf_full)
    print(f'满模型      : loss={loss_full:.3f}, 参数={p_full:,}')

    # 压缩模型（attention + FFN）
    tf_c = Transformer(V, d, L, T, np.random.RandomState(2))
    masks, frozen = apply_compression(tf_c, attn_keep=0.15, ffn_keep=0.10, seed=0)
    loss_c = train(tf_c, data, B, T, steps, V, masks=masks, frozen=frozen)
    p_c = n_params(tf_c, masks, frozen)
    print(f'压缩(attn+FFN): loss={loss_c:.3f}, 参数={p_c:,} ({p_c/p_full:.1%})')

    # 全压缩模型（attention + FFN + embedding 85%）
    tf_e = Transformer(V, d, L, T, np.random.RandomState(2))
    masks_e, frozen_e = apply_compression(tf_e, attn_keep=0.15, ffn_keep=0.10,
                                          emb_keep=0.15, seed=0)
    loss_e = train(tf_e, data, B, T, steps, V, masks=masks_e, frozen=frozen_e)
    p_e = n_params(tf_e, masks_e, frozen_e)
    print(f'压缩(全层含emb): loss={loss_e:.3f}, 参数={p_e:,} ({p_e/p_full:.1%})')

    print()
    print('=' * 74)
    print(f'attn+FFN 压缩: loss 涨 {loss_c - loss_full:+.3f}, 省 {(1-p_c/p_full)*100:.1f}%')
    print(f'全层压缩      : loss 涨 {loss_e - loss_full:+.3f}, 省 {(1-p_e/p_full)*100:.1f}%')
    print('=' * 74)


if __name__ == '__main__':
    main()
