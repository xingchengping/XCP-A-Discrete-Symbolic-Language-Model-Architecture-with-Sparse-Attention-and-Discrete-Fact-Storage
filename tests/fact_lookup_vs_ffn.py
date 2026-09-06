"""事实侧：离散 ID 查表 vs 稀疏 FFN（公平对比）。

用户方案：FFN 事实侧（稀疏 W1 矩阵）替换成「离散 ID 查表」——
  dict: (左词ID, 类ID) -> 右词ID（最高频）。
对比稀疏 FFN 95%，看 acc 是否等价、参数省多少。

关键：同一数据、同一任务（(w1,c2)->w2 的 top1 预测），才公平。

Run:  python tests/fact_lookup_vs_ffn.py
"""
import re
import os
import numpy as np
from collections import Counter, defaultdict

CORPUS = os.path.join(os.path.dirname(__file__), '..', 'Data', 'english_data.txt')


def load(path):
    text = open(path, encoding='utf-8').read()
    words = re.findall(r"[a-z']+", text.lower())
    vocab = sorted(set(words))
    w2i = {w: i for i, w in enumerate(vocab)}
    return words, [w2i[w] for w in words], len(vocab), w2i, vocab


DETERMINERS = {'the', 'a', 'an', 'my', 'his', 'her', 'our', 'their', 'its'}


def tag_pos(words):
    classes = {}
    for i, w in enumerate(words):
        if w in DETERMINERS:
            continue
        if i > 0 and words[i - 1] in DETERMINERS:
            if i > 1 and words[i - 2] in DETERMINERS:
                classes.setdefault(words[i - 1], 'ADJ')
                classes.setdefault(w, 'NOUN')
            else:
                classes.setdefault(w, 'NOUN')
            continue
    for i, w in enumerate(words):
        if i > 0 and classes.get(words[i - 1]) == 'NOUN' and w not in classes:
            classes.setdefault(w, 'VERB')
    for w in words:
        classes.setdefault(w, 'OTHER')
    return classes


# ---------------- 查表方案 ----------------
def build_lookup(ids, word_cls, split):
    """训练集上构建 (w1, c2) -> 最高频 w2。"""
    table = defaultdict(Counter)
    for t in range(1, split):
        table[(ids[t - 1], word_cls[t])][ids[t]] += 1
    lookup = {k: v.most_common(1)[0][0] for k, v in table.items()}
    return lookup


def lookup_acc(lookup, ids, word_cls, start, end):
    hits = total = miss = 0
    for t in range(start, end):
        key = (ids[t - 1], word_cls[t])
        pred = lookup.get(key)
        if pred is None:
            miss += 1
            continue
        total += 1
        if pred == ids[t]:
            hits += 1
    return hits / total if total else 0.0, miss


# ---------------- 稀疏 FFN（复用 ffn_fact_real 的结构） ----------------
def train_sparse_ffn(X, y, K, d_ff, epochs=400, lr=0.1, seed=0):
    rng = np.random.default_rng(seed)
    V = X.shape[1]
    W1 = rng.standard_normal((d_ff, V)) * 0.1
    b1 = np.zeros(d_ff)
    B = rng.standard_normal((K, d_ff))
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    W2 = B  # 冻结
    b2 = np.zeros(K)
    mask = np.zeros((d_ff, V), dtype=bool)
    k = max(1, int(V * 0.1))
    for j in range(d_ff):
        mask[j, rng.choice(V, size=k, replace=False)] = True
    W1 = W1 * mask

    params = {'W1': W1, 'b1': b1, 'b2': b2}
    m = {kk: np.zeros_like(vv) for kk, vv in params.items()}
    v = {kk: np.zeros_like(vv) for kk, vv in params.items()}
    t = 0
    N = len(y)
    for _ in range(epochs):
        z = X @ W1.T + b1
        a = np.maximum(z, 0)
        logits = a @ W2.T + b2
        ex = np.exp(logits - logits.max(1, keepdims=True))
        pr = ex / ex.sum(1, keepdims=True)
        g = pr
        g[np.arange(N), y] -= 1
        g /= N
        db2 = g.sum(0)
        da = g @ W2
        dz = da * (z > 0)
        db1 = dz.sum(0)
        dW1 = dz.T @ X * mask
        grads = {'W1': dW1, 'b1': db1, 'b2': db2}
        t += 1
        for kk in params:
            m[kk] = 0.9 * m[kk] + 0.1 * grads[kk]
            v[kk] = 0.999 * v[kk] + 0.001 * grads[kk] ** 2
            mh = m[kk] / (1 - 0.9 ** t)
            vh = v[kk] / (1 - 0.999 ** t)
            params[kk] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def acc(Xx, yy):
        z = Xx @ W1.T + b1
        a = np.maximum(z, 0)
        logits = a @ W2.T + b2
        return (logits.argmax(1) == yy).mean()

    return acc


def main():
    print('=' * 76)
    print('事实侧：离散 ID 查表 vs 稀疏 FFN（公平对比）')
    print('=' * 76)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    ref_cls = sorted(set(pos.values()))
    cls2id = {c: i for i, c in enumerate(ref_cls)}
    R = len(ref_cls)
    word_cls = [cls2id[pos[w]] for w in words]

    N = len(ids)
    split = int(N * 0.8)

    # 查表
    lookup = build_lookup(ids, word_cls, split)
    tr_acc, _ = lookup_acc(lookup, ids, word_cls, 1, split)
    te_acc, te_miss = lookup_acc(lookup, ids, word_cls, split, N)
    n_facts = len(lookup)
    print(f'\n查表：训练集 acc={tr_acc:.3f}, 测试集 acc={te_acc:.3f}, '
          f'事实条数 F={n_facts}')

    # 稀疏 FFN（同一数据）
    V = K + R
    X = np.zeros((N, V), dtype=np.float32)
    y = np.zeros(N, dtype=int)
    for t in range(1, N):
        X[t - 1, ids[t - 1]] = 1.0
        X[t - 1, K + word_cls[t]] = 1.0
        y[t - 1] = ids[t]
    X = X[:N - 1]
    y = y[:N - 1]
    d_ff = 1024
    acc_fn = train_sparse_ffn(X[:split], y[:split], K, d_ff)
    ffn_tr = acc_fn(X[:split], y[:split])
    ffn_te = acc_fn(X[split:], y[split:])

    print(f'稀疏FFN：训练集 acc={ffn_tr:.3f}, 测试集 acc={ffn_te:.3f}')

    # 参数对比
    ffn_params = int(V * d_ff * 0.1)   # W1 稀疏 90%（W2 冻结）
    lookup_params = n_facts * 2          # 每条事实 2 个整数
    print(f'\n参数：稀疏FFN = {ffn_params} float（{ffn_params*4/1024:.1f}KB），')
    print(f'      查表 = {lookup_params} 整数（{lookup_params*2/1024:.1f}KB），')
    print(f'      省 {(ffn_params*4)/(lookup_params*2):.1f}x')

    print()
    print('=' * 76)
    print('读法：若查表 acc ≈ 稀疏FFN acc（差 <2%），则离散查表可无损替代；')
    print('  参数省多少看「省 X 倍」。')
    print('=' * 76)


if __name__ == '__main__':
    main()
