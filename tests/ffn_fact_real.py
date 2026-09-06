"""事实侧稀疏 FFN 在真实语料上的压缩验证。

之前 ffn_capacity.py 用「随机映射」证明稀疏 90% 记忆无损，但那是合成任务。
本脚本用真实语料的「类内词搭配」再验一次，看 95% 压缩是否仍记忆无损。

任务（事实侧 = 类内选词）：
  给定 (上一个词 w1, 当前类 c2)，预测具体词 w2。
  —— 这就是「事实」：the NOUN 里 NOUN 具体是 dog 还是 cat。

模型 = 稀疏 FFN：
  输入 = w1 one-hot(K) + c2 one-hot(R) 拼接；
  W1 稀疏 90%（每神经元只连 10% 输入）+ W2 冻结（随机基）；
  输出 = w2 的 K 维 logits。

对比：满 FFN vs 冻结W2 vs 稀疏W1 vs 冻结+稀疏，看训练 acc（记忆）掉多少。

Run:  python tests/ffn_fact_real.py
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


def main():
    print('=' * 78)
    print('事实侧稀疏 FFN：真实语料类内选词，95% 压缩是否记忆无损')
    print('=' * 78)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    ref_cls = sorted(set(pos.values()))
    cls2id = {c: i for i, c in enumerate(ref_cls)}
    R = len(ref_cls)
    word_cls = [cls2id[pos[w]] for w in words]

    # 构建样本：输入 (w1 one-hot + c2 one-hot)，输出 w2
    V = K + R
    N = len(ids) - 1
    X = np.zeros((N, V), dtype=np.float32)
    y = np.zeros(N, dtype=int)
    for t in range(1, len(ids)):
        w1, c2, w2 = ids[t - 1], word_cls[t], ids[t]
        X[t - 1, w1] = 1.0
        X[t - 1, K + c2] = 1.0
        y[t - 1] = w2

    # 用前 800 个样本（子集），d_ff=1024 够记，加速
    N = 800
    X, y = X[:N], y[:N]

    d_ff = 1024
    ffn_full = V * d_ff + d_ff * K
    print(f'\n语料 {len(ids)} 词，K={K}，R={R} 类。用前 N={N} 个词对。')
    print(f'输入维度 V=K+R={V}，输出 K={K}，d_ff={d_ff}。')
    print(f'满 FFN 参数 = {ffn_full:,}\n')

    print(f'{"方案":<22} {"压缩率":>7} {"训练acc(记忆)":>13} {"测试acc":>9}')
    print('-' * 56)

    # 80/20 切分
    split = int(N * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    def train(freeze_W2, sparse_W1, epochs=300, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        W1 = rng.standard_normal((d_ff, V)) * 0.1
        b1 = np.zeros(d_ff)
        if freeze_W2:
            B = rng.standard_normal((K, d_ff))
            B = B / np.linalg.norm(B, axis=1, keepdims=True)
            W2 = B
        else:
            W2 = rng.standard_normal((K, d_ff)) * 0.1
        b2 = np.zeros(K)
        mask = None
        if sparse_W1:
            mask = np.zeros((d_ff, V), dtype=bool)
            k = max(1, int(V * 0.1))
            for j in range(d_ff):
                mask[j, rng.choice(V, size=k, replace=False)] = True
            W1 = W1 * mask

        params = {'W1': W1, 'b1': b1, 'b2': b2}
        if not freeze_W2:
            params['W2'] = W2
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        tt = 0
        n_tr = len(y_tr)
        for _ in range(epochs):
            z = X_tr @ W1.T + b1
            a = np.maximum(z, 0)
            logits = a @ W2.T + b2
            ex = np.exp(logits - logits.max(1, keepdims=True))
            probs = ex / ex.sum(1, keepdims=True)
            g = probs
            g[np.arange(n_tr), y_tr] -= 1
            g /= n_tr
            db2 = g.sum(0)
            dW2 = g.T @ a
            da = g @ W2
            dz = da * (z > 0)
            db1 = dz.sum(0)
            dW1 = dz.T @ X_tr
            if mask is not None:
                dW1 = dW1 * mask
            grads = {'b2': db2, 'b1': db1, 'W1': dW1}
            if not freeze_W2:
                grads['W2'] = dW2
            tt += 1
            for kk in params:
                m[kk] = 0.9 * m[kk] + 0.1 * grads[kk]
                v[kk] = 0.999 * v[kk] + 0.001 * grads[kk] ** 2
                mh = m[kk] / (1 - 0.9 ** tt)
                vh = v[kk] / (1 - 0.999 ** tt)
                params[kk] -= lr * mh / (np.sqrt(vh) + 1e-8)

        def acc(Xx, yy):
            z = Xx @ W1.T + b1
            a = np.maximum(z, 0)
            logits = a @ W2.T + b2
            return (logits.argmax(1) == yy).mean()

        return acc(X_tr, y_tr), acc(X_te, y_te)

    def report(name, params, tr, te):
        print(f'{name:<22} {1-params/ffn_full:>6.1%} {tr:>13.3f} {te:>9.3f}')

    tr, te = train(False, False)
    report('满(全可学习)', ffn_full, tr, te)
    tr, te = train(True, True)
    report('冻结W2+稀疏W1', int(V * d_ff * 0.1), tr, te)

    print()
    print('=' * 78)
    print('读法：训练 acc = 记忆（事实侧的职责）；测试 acc = 泛化（结构侧的职责，')
    print('  本实验不要求）。关键看「冻结W2+稀疏W1」的训练 acc 是否 ≈ 满模型。')
    print('=' * 78)


if __name__ == '__main__':
    main()
