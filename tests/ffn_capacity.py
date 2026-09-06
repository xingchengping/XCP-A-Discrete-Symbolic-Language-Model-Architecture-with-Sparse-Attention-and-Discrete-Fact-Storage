"""决定性实验：FFN 知识层的「记忆容量 vs 参数量」。

问题：FFN 参数量砍到 10%（稀疏 90% + 冻结 W2），能存的事实是不是也只剩 10%？

用最干净的「记忆随机映射」任务：
  随机 N 个 one-hot 输入 x → 随机 one-hot 输出 y（没有任何规律，纯事实）。
  模型记住的映射数 = 它的「事实容量」。这是知识层的本质能力。

对比四个方案（同一任务、同一训练预算）：
  [满]     W1、W2 稠密可训练（基准容量）
  [冻结W2] W2 = 固定随机基（参数减半）
  [稀疏W1] W1 保留 10%（90% 掩码）
  [结合]   冻结 W2 + 稀疏 W1（用户方案，参数 ~5%）

若「结合」容量 ≈ 满的 5%  → 参数∝容量，90% 压缩 = 90% 残废（坐实不可压）；
若「结合」容量 ≈ 满的 100% → 满 FFN 有 90% 冗余（用户的 90% 压缩成立）。

Run:  python tests/ffn_capacity.py
"""
import numpy as np


def make_task(V, N, seed=0, dense=True):
    rng = np.random.default_rng(seed)
    if dense:
        # 稠密随机输入（真实 FFN 的输入是 embedding/attention 的稠密向量）
        X = rng.standard_normal((N, V)).astype(np.float32)
        y = rng.integers(0, V, size=N)   # 随机输出类别
    else:
        # one-hot 输入（最弱任务，每个维度独立）
        x_idx = rng.choice(V, size=N, replace=False)
        y = rng.choice(V, size=N, replace=True)
        X = np.zeros((N, V), dtype=np.float32)
        X[np.arange(N), x_idx] = 1.0
    return X, y


def make_random_B(d, d_ff, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d, d_ff))
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return B


def make_mask(d_ff, V, keep_frac, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((d_ff, V), dtype=bool)
    k = max(1, int(V * keep_frac))
    for j in range(d_ff):
        mask[j, rng.choice(V, size=k, replace=False)] = True
    return mask


def train(X, y, V, d_ff, freeze_W2=False, sparse_W1=False, epochs=800, lr=0.1,
          seed=0):
    N = len(y)
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d_ff, V)) * 0.1
    b1 = np.zeros(d_ff)
    if freeze_W2:
        W2 = make_random_B(V, d_ff, seed + 1)
    else:
        W2 = rng.standard_normal((V, d_ff)) * 0.1
    b2 = np.zeros(V)
    mask = make_mask(d_ff, V, 0.1, seed + 2) if sparse_W1 else None
    if mask is not None:
        W1 = W1 * mask

    # Adam 状态
    def adam_init(p):
        return np.zeros_like(p), np.zeros_like(p)
    params = {}
    if not freeze_W2:
        params['W2'] = W2
    params['W1'] = W1
    params['b1'] = b1
    params['b2'] = b2
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for ep in range(epochs):
        z = X @ W1.T + b1
        a = np.maximum(z, 0)
        logits = a @ W2.T + b2
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)
        g = probs
        g[np.arange(N), y] -= 1
        g /= N
        # 梯度
        db2 = g.sum(0)
        dW2 = g.T @ a
        da = g @ W2
        dz = da * (z > 0)
        db1 = dz.sum(0)
        dW1 = dz.T @ X
        if mask is not None:
            dW1 = dW1 * mask
        grads = {'b2': db2, 'b1': db1, 'W1': dW1}
        if not freeze_W2:
            grads['W2'] = dW2
        t += 1
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            mh = m[k] / (1 - 0.9 ** t)
            vh = v[k] / (1 - 0.999 ** t)
            params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    # 最终 acc
    z = X @ W1.T + b1
    a = np.maximum(z, 0)
    logits = a @ W2.T + b2
    pred = logits.argmax(1)
    return (pred == y).mean()


def main():
    print('=' * 76)
    print('决定性实验：FFN 记忆容量 vs 参数量（稠密随机输入，贴近真实 FFN）')
    print('=' * 76)

    V, d_ff = 256, 256
    ffn_full = 2 * V * d_ff
    print(f'\nV={V}, d_ff={d_ff}。满 FFN 参数 = {ffn_full:,}（容量极限 ≈ d_ff={d_ff}）')

    print(f'\n{"方案":<24} {"FFN参数":>10} {"压缩率":>7} {"记忆acc":>8}')
    print('-' * 56)

    def report(name, params, acc):
        print(f'{name:<24} {params:>10,} {1-params/ffn_full:>6.1%} {acc:>8.3f}')

    for N in (128, 256):
        print(f'\n--- 记忆 N={N} 个随机映射 ---')
        X, y = make_task(V, N)
        a1 = train(X, y, V, d_ff, freeze_W2=False, sparse_W1=False)
        report(f'满(全可学习) N={N}', ffn_full, a1)
        a2 = train(X, y, V, d_ff, freeze_W2=True, sparse_W1=False)
        report(f'冻结W2 N={N}', V * d_ff, a2)
        a3 = train(X, y, V, d_ff, freeze_W2=False, sparse_W1=True)
        report(f'稀疏W1(10%) N={N}', int(V * d_ff * 0.1) + V * d_ff, a3)
        a4 = train(X, y, V, d_ff, freeze_W2=True, sparse_W1=True)
        report(f'冻结W2+稀疏W1 N={N}', int(V * d_ff * 0.1), a4)

    print()
    print('=' * 76)
    print('读法：')
    print('  满模型 acc≈1.0 的 N 就是满 FFN 的容量；各压缩方案在同一 N 下的 acc')
    print('  掉多少，就是「参数量砍到 X% 后容量剩多少」。')
    print('=' * 76)


if __name__ == '__main__':
    main()
