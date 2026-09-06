"""诊断 embedding 满秩：类间 vs 类内（词级语料）。

embedding 满秩是压缩死结。本脚本把 embedding 拆成「类中心 + 类内残差」，
测残差秩：若残差低秩，则 embedding 可压成「类中心(R×d) + 低秩残差」。

词级语料 english_data.txt（V=895 词），词性 NOUN/VERB/OTHER 三类。
简单 bigram embedding 训练（快），测类间/类内秩。

Run:  python tests/embed_compress.py
"""
import re
import os
import numpy as np
from collections import Counter

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


def train_emb(ids, K, d, epochs=800, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.standard_normal((K, d)) / np.sqrt(d)
    O = rng.standard_normal((d, K)) / np.sqrt(d)
    prev = np.array(ids[:-1])
    nxt = np.array(ids[1:])
    N = len(prev)
    mE = np.zeros_like(E); vE = np.zeros_like(E)
    mO = np.zeros_like(O); vO = np.zeros_like(O)
    t = 0
    for _ in range(epochs):
        logits = E[prev] @ O          # (N, K)
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p = p / p.sum(1, keepdims=True)
        g = p; g[np.arange(N), nxt] -= 1; g /= N
        dO = E[prev].T @ g
        dE_in = g @ O.T
        dE = np.zeros_like(E)
        np.add.at(dE, prev, dE_in)
        t += 1
        for name, w, gw in [('E', E, dE), ('O', O, dO)]:
            m, vv = (mE, vE) if name == 'E' else (mO, vO)
            m = 0.9 * m + 0.1 * gw
            vv = 0.999 * vv + 0.001 * gw ** 2
            mh = m / (1 - 0.9 ** t); vh = vv / (1 - 0.999 ** t)
            w -= lr * mh / (np.sqrt(vh) + 1e-8)
    return E


def kmeans_cosine(X, R, iters=60, restarts=10, seed=0):
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    K = len(Xn)
    best = None
    for _ in range(restarts):
        idx = rng.choice(K, R, replace=False)
        C = Xn[idx].copy()
        labels = np.zeros(K, dtype=int)
        for _ in range(iters):
            sim = Xn @ C.T
            labels = sim.argmax(axis=1)
            newC = np.zeros_like(C)
            for c in range(R):
                m = Xn[labels == c]
                if len(m):
                    newC[c] = m.mean(axis=0)
                else:
                    newC[c] = Xn[rng.integers(K)]
            newC /= (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-12)
            if np.allclose(C, newC):
                C = newC
                break
            C = newC
        score = (Xn @ C.T).max(axis=1).sum()
        if best is None or score > best[0]:
            best = (score, labels, C)
    return best[1]


def explained_variance(E, labels, R):
    """类间方差占比 = 1 - 类内方差/总方差（聚类解释的方差比例）。"""
    mu = E.mean(0)
    total = ((E - mu) ** 2).sum()
    within = 0.0
    for c in range(R):
        members = E[labels == c]
        if len(members):
            within += ((members - members.mean(0)) ** 2).sum()
    return 1 - within / total if total > 0 else 0.0


def main():
    print('=' * 72)
    print('embedding 类中心能解释多少方差（vs 随机对照）')
    print('=' * 72)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    ref_cls = sorted(set(pos.values()))
    cls2id = {c: i for i, c in enumerate(ref_cls)}

    d = 128
    print(f'\n语料 {len(words)} 词，V={K}，d={d}。训练 embedding...')
    E = train_emb(ids, K, d)
    print(f'embedding 总有效秩 = {eff_rank(E)}/{d}\n')

    # 邻居分布特征（左+右）
    B = np.zeros((K, 2 * K), dtype=np.float32)
    for t in range(len(ids) - 1):
        B[ids[t], ids[t + 1]] += 1
        B[ids[t + 1], K + ids[t]] += 1

    print(f'{"聚类方式":<20} {"R":>4} | {"类间方差占比":>10} {"(相对随机)":>10}')
    print('-' * 52)

    # 随机对照（多次平均）
    rng = np.random.default_rng(0)
    rand_refs = {}
    for R in (3, 10, 30):
        vals = []
        for _ in range(5):
            rl = rng.integers(0, R, size=K)
            vals.append(explained_variance(E, rl, R))
        rand_refs[R] = np.mean(vals)

    # 词性类
    cls = np.array([cls2id[pos[v]] for v in vocab])
    R = len(ref_cls)
    ev = explained_variance(E, cls, R)
    print(f'{"词性类":<20} {R:>4} | {ev:>9.1%} {ev/rand_refs[R]:>9.2f}x')

    # embedding 自身 kmeans
    for R in (3, 10, 30):
        labels = kmeans_cosine(E, R)
        ev = explained_variance(E, labels, R)
        print(f'{"embedding kmeans":<20} {R:>4} | {ev:>9.1%} {ev/rand_refs[R]:>9.2f}x')

    # 邻居分布聚类
    for R in (3, 10, 30):
        labels = kmeans_cosine(B, R)
        ev = explained_variance(E, labels, R)
        print(f'{"邻居分布聚类":<20} {R:>4} | {ev:>9.1%} {ev/rand_refs[R]:>9.2f}x')

    print()
    print('=' * 72)
    print('读法：类间方差占比 = 聚类能「解释」多少词向量差异。')
    print('  (相对随机) = 占比 / 随机分组的占比，>1 说明有真实结构，≈1 说明无结构。')
    print('=' * 72)


if __name__ == '__main__':
    main()
