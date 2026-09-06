"""注意力稀疏实验：稀疏 85/90% 的 Q/K/V/O 从头训练，检索能力是否等价。

注意力的核心功能 = 检索：给 query，从 key-value 里找到匹配的 key、输出对应
value。任务用 associative recall（联想检索）：
  序列 = [k1 v1 k2 v2 k3 v3 k4 v4 q]，q 随机等于某个 ki，目标输出 vi。

模型 = 单头注意力（query 位置），Q/K/V/O 投影可稀疏。
对比：满 vs 稀疏 85% vs 稀疏 90%，看检索准确率掉多少。

Run:  python tests/attention_sparse.py
"""
import numpy as np


def make_batch(V, d, N, B, rng):
    """生成 B 个序列：[k1 v1 ... kN vN q]，q=某 ki，目标=vi。返回 idx 和 targets。"""
    seqs = np.zeros((B, 2 * N + 1), dtype=int)
    targets = np.zeros(B, dtype=int)
    for b in range(B):
        keys = rng.integers(0, V, size=N)
        vals = rng.integers(0, V, size=N)
        for i in range(N):
            seqs[b, 2 * i] = keys[i]
            seqs[b, 2 * i + 1] = vals[i]
        qi = rng.integers(0, N)
        seqs[b, -1] = keys[qi]
        targets[b] = vals[qi]
    return seqs, targets


def make_mask(d, keep_frac, rng):
    mask = np.zeros((d, d), dtype=bool)
    k = max(1, int(d * keep_frac))
    for j in range(d):
        mask[j, rng.choice(d, size=k, replace=False)] = True
    return mask


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def train(V, d, N, B, sparse_frac, epochs=400, lr=0.01, seed=0):
    T = 2 * N + 1
    q_pos = T - 1
    scale = 1.0 / np.sqrt(d)
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((V, d)) * scale
    P = rng.standard_normal((T, d)) * scale
    WQ = rng.standard_normal((d, d)) * scale
    WK = rng.standard_normal((d, d)) * scale
    WV = rng.standard_normal((d, d)) * scale
    WO = rng.standard_normal((d, d)) * scale
    masks = {}
    if sparse_frac > 0:
        keep = 1 - sparse_frac
        masks['WQ'] = make_mask(d, keep, rng)
        masks['WK'] = make_mask(d, keep, rng)
        masks['WV'] = make_mask(d, keep, rng)
        masks['WO'] = make_mask(d, keep, rng)
        WQ = WQ * masks['WQ']
        WK = WK * masks['WK']
        WV = WV * masks['WV']
        WO = WO * masks['WO']

    params = {'E': E, 'P': P, 'WQ': WQ, 'WK': WK, 'WV': WV, 'WO': WO}
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for _ in range(epochs):
        seqs, targets = make_batch(V, d, N, B, rng)
        X = E[seqs] + P[None, :, :]     # (B, T, d) 加位置编码
        Q = X @ WQ.T                   # (B, T, d)
        K = X @ WK.T
        Vv = X @ WV.T
        q = Q[:, q_pos]                # (B, d)
        scores = (q[:, None, :] @ K.transpose(0, 2, 1)).squeeze(1) / np.sqrt(d)  # (B, T)
        probs = softmax(scores)        # (B, T)
        ctx = probs[:, :, None] * Vv   # (B, T, d)
        ctx = ctx.sum(1)               # (B, d)
        out = ctx @ WO.T               # (B, d)
        logits = out @ E.T             # (B, V)

        ex = np.exp(logits - logits.max(1, keepdims=True))
        pr = ex / ex.sum(1, keepdims=True)
        if (sparse_frac == 0.0) and (_ % 50 == 0 or _ == epochs - 1):
            loss = -np.log(pr[np.arange(B), targets] + 1e-12).mean()
            print(f'    ep={_:3d} loss={loss:.3f}', flush=True)
        g = pr
        g[np.arange(B), targets] -= 1
        g /= B

        dout = g @ E                   # (B, d)
        dctx = dout @ WO               # (B, d)
        dWO = dout.T @ ctx             # (d, d)
        dVv = probs[:, :, None] * dctx[:, None, :]   # (B, T, d)
        dprobs = (dctx[:, None, :] * Vv).sum(2)      # (B, T)
        dscores = probs * (dprobs - (dprobs * probs).sum(1, keepdims=True))  # (B, T)
        dq = np.einsum('bt,btd->bd', dscores, K) / np.sqrt(d)   # (B, d)
        dK = dscores[:, :, None] * q[:, None, :] / np.sqrt(d)   # (B, T, d)
        dQ = np.zeros_like(Q)
        dQ[:, q_pos] = dq
        dQ = dQ
        dWQ = np.einsum('btd,bte->de', X, dQ)
        dWK = np.einsum('btd,bte->de', X, dK)
        dWV = np.einsum('btd,bte->de', X, dVv)

        dX = (dQ @ WQ + dK @ WK + dVv @ WV)  # (B,T,d)
        dE = np.zeros_like(E)
        np.add.at(dE, seqs, dX)
        dP = dX.sum(0)                     # 位置编码梯度

        grads = {'E': dE, 'P': dP, 'WQ': dWQ, 'WK': dWK, 'WV': dWV, 'WO': dWO}
        for k in masks:
            grads[k] = grads[k] * masks[k]

        t += 1
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / epochs))  # cosine 衰减
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            mh = m[k] / (1 - 0.9 ** t)
            vh = v[k] / (1 - 0.999 ** t)
            params[k] -= lr_t * mh / (np.sqrt(vh) + 1e-8)

    # 评估：新 batch 的检索 acc
    seqs, targets = make_batch(V, d, N, B, rng)
    X = E[seqs] + P[None, :, :]
    Q = X @ WQ.T
    K = X @ WK.T
    Vv = X @ WV.T
    q = Q[:, q_pos]
    scores = (q[:, None, :] @ K.transpose(0, 2, 1)).squeeze(1) / np.sqrt(d)
    probs = softmax(scores)
    ctx = (probs[:, :, None] * Vv).sum(1)
    out = ctx @ WO.T
    logits = out @ E.T
    return (logits.argmax(1) == targets).mean()


def main():
    print('=' * 74)
    print('注意力稀疏实验：稀疏 Q/K/V/O 后检索能力是否等价')
    print('=' * 74)

    V, N, B = 64, 2, 128
    print(f'\n任务：associative recall（V={V}, N={N} 对 key-value, B={B}）\n')

    print(f'{"d":>4} | {"满acc":>7} {"稀疏85%acc":>10} {"Δacc":>7}')
    print('-' * 42)

    for d in (64, 128, 256):
        acc_full = train(V, d, N, B, 0.0, epochs=1500, seed=0)
        acc_sparse = train(V, d, N, B, 0.85, epochs=1500, seed=0)
        print(f'{d:>4} | {acc_full:>7.3f} {acc_sparse:>10.3f} {acc_sparse-acc_full:>+7.3f}')

    print()
    print('=' * 74)
    print('读法：Δacc（稀疏85% - 满）若随 d 增大保持 ≈0，则稀疏无损在大 d 下也成立；')
    print('  若随 d 增大转负（稀疏掉 acc），则稀疏在大 d 失效。')
    print('=' * 74)


if __name__ == '__main__':
    main()
