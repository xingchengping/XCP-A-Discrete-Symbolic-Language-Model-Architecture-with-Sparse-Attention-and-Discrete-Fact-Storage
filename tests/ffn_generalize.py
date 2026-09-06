"""FFN 泛化决定性测试（回归版）：稀疏90%+冻结W2 能否表达低秩结构。

用回归任务，天花板明确 = MSE→0（完美学到 W_true）。没有分类的过拟合混淆。
核心问题：稀疏 90% 后，d_ff 个稀疏神经元（每个只看 10% 维度）合起来，
能否表达 rank-r 的满维度结构？

任务：y = x · W_true，W_true = U·V（rank r）。x ∈ R^V，y ∈ R^d_out。
Run:  python tests/ffn_generalize.py
"""
import numpy as np


def make_task(V, d_out, r, N_train, N_test, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((V, r))
    Vm = rng.standard_normal((r, d_out))
    W_true = U @ Vm                          # (V, d_out) rank=r
    X_tr = rng.standard_normal((N_train, V)).astype(np.float32)
    Y_tr = X_tr @ W_true
    X_te = rng.standard_normal((N_test, V)).astype(np.float32)
    Y_te = X_te @ W_true
    return X_tr, Y_tr, X_te, Y_te


def make_random_B(d, d_ff, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d, d_ff))
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def make_mask(d_ff, V, keep_frac, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((d_ff, V), dtype=bool)
    k = max(1, int(V * keep_frac))
    for j in range(d_ff):
        mask[j, rng.choice(V, size=k, replace=False)] = True
    return mask


def train(X_tr, Y_tr, X_te, Y_te, V, d_out, d_ff, freeze_W2=False,
          sparse_W1=False, epochs=500, lr=0.05, wd=0.0, seed=0):
    N = len(Y_tr)
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((d_ff, V)) * 0.1
    b1 = np.zeros(d_ff)
    W2 = make_random_B(d_out, d_ff, seed + 1) if freeze_W2 else \
        rng.standard_normal((d_out, d_ff)) * 0.1
    b2 = np.zeros(d_out)
    mask = make_mask(d_ff, V, 0.1, seed + 2) if sparse_W1 else None
    if mask is not None:
        W1 = W1 * mask

    params = {'W1': W1, 'b1': b1, 'b2': b2}
    if not freeze_W2:
        params['W2'] = W2
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for _ in range(epochs):
        z = X_tr @ W1.T + b1
        a = np.maximum(z, 0)
        out = a @ W2.T + b2
        err = out - Y_tr
        dout = 2 * err / N
        db2 = dout.sum(0)
        dW2 = dout.T @ a
        da = dout @ W2
        dz = da * (z > 0)
        db1 = dz.sum(0)
        dW1 = dz.T @ X_tr
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
            params[k] *= (1 - lr * wd)   # weight decay：逼学低秩结构而非死记

    def mse(X, Y):
        z = X @ W1.T + b1
        a = np.maximum(z, 0)
        out = a @ W2.T + b2
        return ((out - Y) ** 2).mean()

    return mse(X_tr, Y_tr), mse(X_te, Y_te)


def main():
    print('=' * 76)
    print('FFN 泛化决定性测试（回归）：稀疏90%能否表达低秩结构')
    print('=' * 76)

    V, d_out, d_ff = 256, 4, 64
    ffn_full = V * d_ff + d_ff * d_out
    print(f'\nV={V}, d_out={d_out}, d_ff={d_ff}。满 FFN 参数 = {ffn_full:,}')
    print(f'（稀疏 90% 后每个神经元看 {int(V*0.1)} 维，共 {d_ff} 个神经元，')
    print(f'  合覆盖 {d_ff*int(V*0.1):,} 维 > V={V} → 理论上仍能表达满维度）\n')

    for r in (2, 4, 8):
        X_tr, Y_tr, X_te, Y_te = make_task(V, d_out, r, 5000, 2000, seed=0)
        print(f'--- rank r={r}（结构复杂度）---')
        print(f'{"方案":<20} {"压缩率":>6} {"训练MSE":>10} {"测试MSE":>10}')
        print('-' * 52)

        def run(name, freeze, sparse, params):
            tr, te = train(X_tr, Y_tr, X_te, Y_te, V, d_out, d_ff,
                           freeze_W2=freeze, sparse_W1=sparse)
            print(f'{name:<20} {1-params/ffn_full:>5.1%} {tr:>10.4f} {te:>10.4f}')

        run('满(全可学习)', False, False, ffn_full)
        run('冻结W2', True, False, V * d_ff)
        run('稀疏W1(10%)', False, True, int(V * d_ff * 0.1) + d_ff * d_out)
        run('冻结W2+稀疏W1', True, True, int(V * d_ff * 0.1))

    print()
    print('=' * 76)
    print('读法：测试 MSE 越接近 0 = 越完美学到低秩结构（泛化好）。')
    print('  满模型应 MSE→0；稀疏/冻结的 MSE 涨多少 = 压缩的真实代价。')
    print('=' * 76)


if __name__ == '__main__':
    main()
