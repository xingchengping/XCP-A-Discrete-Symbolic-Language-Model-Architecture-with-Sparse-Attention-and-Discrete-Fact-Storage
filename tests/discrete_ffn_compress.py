"""实测：重构 FFN 的「冻结 W2 用随机基 + W1 稀疏 90%」到底掉多少 loss。

用户的方案（discrete_ffn 重构）：
    FFN(x) = B_fixed · σ( (W1 ⊙ M_fixed) · x )
  - B_fixed 随机基替换可训练 W2（参数→0）
  - M_fixed 90% 稀疏掩码（W1 只训 10%）

本脚本在真实 next-word 任务上，把方案拆成四个变体逐一实测 loss：
  [基准]    W1、W2 都可学习（稠密）
  [冻结W2]  W2 = 固定随机基 B（近似正交 / 精确正交两种），只训 W1
  [稀疏W1]  W1 保留 10%（90% 掩码），W2 正常训练
  [结合]    冻结 W2 + 稀疏 W1（用户方案的完整形态）

任务：词级 next-word 预测（english_data.txt，K=895 词）。模型：
  x = E_in[id] ; z = W1·x + b1 ; a = ReLU(z) ; y = W2·a + b2 ; logits = y·Oᵀ
  E_in、O 是 embedding（不在本方案压缩范围内，正常训练）。

只报告 FFN（W1+W2）的压缩率和对应的 loss 变化 —— 这是用户方案的核心。

Run:  python tests/discrete_ffn_compress.py
"""
import re
import os
import numpy as np

CORPUS = os.path.join(os.path.dirname(__file__), '..', 'Data', 'english_data.txt')


def load(path):
    text = open(path, encoding='utf-8').read()
    words = re.findall(r"[a-z']+", text.lower())
    vocab = sorted(set(words))
    w2i = {w: i for i, w in enumerate(vocab)}
    return [w2i[w] for w in words], len(vocab)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits, targets):
    logz = logits - logits.max(axis=-1, keepdims=True)
    lp = logz - np.log(np.exp(logz).sum(axis=-1, keepdims=True))
    return -lp[np.arange(len(targets)), targets].mean()


def make_random_B(d, d_ff, orthogonal, seed):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d, d_ff))
    if orthogonal:
        # Gram-Schmidt 行正交化 → BBᵀ = I_d
        for i in range(d):
            for j in range(i):
                B[i] -= B[j] * (B[i] @ B[j])
            n = np.linalg.norm(B[i])
            B[i] /= n if n > 1e-12 else 1.0
    else:
        B = B / np.linalg.norm(B, axis=1, keepdims=True)  # 行归一（近似正交）
    return B


def make_mask(d_ff, d, keep_frac, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((d_ff, d), dtype=bool)
    k = max(1, int(d * keep_frac))
    for j in range(d_ff):
        mask[j, rng.choice(d, size=k, replace=False)] = True
    return mask


class Adam:
    def __init__(self, lr=3e-3):
        self.lr, self.t = lr, 0
        self.m, self.v = {}, {}

    def step(self, name, param, grad):
        if name not in self.m:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)
        self.t += 1
        self.m[name] = 0.9 * self.m[name] + 0.1 * grad
        self.v[name] = 0.999 * self.v[name] + 0.001 * grad ** 2
        mh = self.m[name] / (1 - 0.9 ** self.t)
        vh = self.v[name] / (1 - 0.999 ** self.t)
        param -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def train_model(ids, K, d, d_ff, freeze_W2, sparse_W1, orthogonal, epochs=200,
                lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((K, d)) * 0.1
    O = rng.standard_normal((K, d)) * 0.1
    W1 = rng.standard_normal((d_ff, d)) * 0.1
    b1 = np.zeros(d_ff)
    b2 = np.zeros(d)
    if freeze_W2:
        W2 = make_random_B(d, d_ff, orthogonal, seed + 1)
    else:
        W2 = rng.standard_normal((d, d_ff)) * 0.1
    mask = make_mask(d_ff, d, 0.1, seed + 2) if sparse_W1 else None
    if mask is not None:
        W1 = W1 * mask

    split = int(len(ids) * 0.8)
    tr_x, tr_y = np.array(ids[:split]), np.array(ids[1:split + 1])
    te_x, te_y = np.array(ids[split:-1]), np.array(ids[split + 1:])

    opt = Adam(lr)
    for ep in range(epochs):
        # ---- 前向（训练集） ----
        x = E[tr_x]                      # (N,d)
        z = x @ W1.T + b1
        a = np.maximum(z, 0)
        y = a @ W2.T + b2
        logits = y @ O.T
        # ---- 反向 ----
        probs = softmax(logits)
        dlogits = probs.copy()
        dlogits[np.arange(len(tr_y)), tr_y] -= 1
        dlogits /= len(tr_y)
        dy = dlogits @ O               # (N,d)
        dO = dlogits.T @ y             # (K,d)
        db2 = dy.sum(0)
        da = dy @ W2                   # (N,d_ff)
        dz = da * (z > 0)
        dW1 = dz.T @ x                 # (d_ff,d)
        db1 = dz.sum(0)
        dx = dz @ W1                   # (N,d)
        dE = np.zeros_like(E)
        np.add.at(dE, tr_x, dx)        # 通过输入路径 x=E[tr_x] 累积
        # W2 梯度（只有可训练时才更新）
        dW2 = dy.T @ a
        # 更新
        opt.step('E', E, dE)
        opt.step('O', O, dO)
        opt.step('b1', b1, db1)
        opt.step('b2', b2, db2)
        if mask is not None:
            dW1 = dW1 * mask
        opt.step('W1', W1, dW1)
        if not freeze_W2:
            opt.step('W2', W2, dW2)

    # ---- 评估 ----
    def loss_on(x_ids, y_ids):
        x = E[x_ids]
        z = x @ W1.T + b1
        a = np.maximum(z, 0)
        y = a @ W2.T + b2
        logits = y @ O.T
        return cross_entropy(logits, y_ids)

    return loss_on(tr_x, tr_y), loss_on(te_x, te_y)


def main():
    print('=' * 84)
    print('实测：FFN 重构（冻结W2随机基 + W1稀疏90%）掉多少 loss')
    print('=' * 84)

    ids, K = load(CORPUS)
    d, d_ff = 64, 256
    ffn_base = 2 * d * d_ff
    print(f'\n语料 {len(ids)} 词，K={K}。d={d}, d_ff={d_ff}。')
    print(f'FFN 基准参数 = 2·d·d_ff = {ffn_base:,}\n')

    print(f'{"方案":<26} {"FFN参数":>9} {"压缩率":>7} {"训练loss":>9} {"测试loss":>9}')
    print('-' * 66)

    def report(name, ffn_params, tr, te):
        ratio = 1 - ffn_params / ffn_base
        print(f'{name:<26} {ffn_params:>9,} {ratio:>6.1%} {tr:>9.3f} {te:>9.3f}')

    # 基准
    tr, te = train_model(ids, K, d, d_ff, freeze_W2=False, sparse_W1=False,
                         orthogonal=False)
    report('基准(全可学习)', ffn_base, tr, te)

    # 冻结 W2（近似正交）
    tr, te = train_model(ids, K, d, d_ff, freeze_W2=True, sparse_W1=False,
                         orthogonal=False)
    report('冻结W2(随机基,近似正交)', d * d_ff, tr, te)

    # 冻结 W2（精确正交）
    tr, te = train_model(ids, K, d, d_ff, freeze_W2=True, sparse_W1=False,
                         orthogonal=True)
    report('冻结W2(随机基,QR正交)', d * d_ff, tr, te)

    # 稀疏 W1（90%）
    tr, te = train_model(ids, K, d, d_ff, freeze_W2=False, sparse_W1=True,
                         orthogonal=False)
    report('稀疏W1(留10%)', int(d * d_ff * 0.1) + d * d_ff, tr, te)

    # 结合（用户方案完整形态）
    tr, te = train_model(ids, K, d, d_ff, freeze_W2=True, sparse_W1=True,
                         orthogonal=True)
    report('冻结W2+稀疏W1(用户方案)', int(d * d_ff * 0.1), tr, te)

    print()
    print('=' * 84)
    print('读法：压缩率越高、loss 涨得越少，方案越好。')
    print('关键看「结合」相对「基准」：loss 涨多少 = 90% 压缩的真实代价。')
    print('=' * 84)


if __name__ == '__main__':
    main()
