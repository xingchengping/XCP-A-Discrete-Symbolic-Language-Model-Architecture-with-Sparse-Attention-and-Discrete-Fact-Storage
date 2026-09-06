"""抽象层 v2：提高类纯度（左+右邻居特征，不再只用单侧）。

v1 只用「右邻居分布」，纯度只有 ~0.5。本脚本对比三种特征，看抽象质量
能否提高：
  [右]  这个词后面常接什么（v1）
  [左]  这个词前面常是什么
  [左+右] 两侧拼接 —— 名词前是 the、后是动词，动词前是名词、后是 the，
          两侧一起才能干净分开

评估两把尺：
  1. 纯度：自动类 vs 启发式词性（NOUN/VERB/OTHER）的重合度
  2. 类级 acc：自动类的「类→类」预测命中率，对标白给词性类 0.672

本脚本只做「结构侧」（类级），不碰类内选词（事实侧，用户管）。

Run:  python tests/discrete_abstract_v2.py
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


def build_features(ids, K, mode):
    """mode: 'right' | 'left' | 'both'。返回 K×F 归一化特征矩阵。"""
    if mode == 'right':
        F = np.zeros((K, K), dtype=np.float32)
        for t in range(len(ids) - 1):
            F[ids[t], ids[t + 1]] += 1
    elif mode == 'left':
        F = np.zeros((K, K), dtype=np.float32)
        for t in range(1, len(ids)):
            F[ids[t], ids[t - 1]] += 1
    else:  # both: 左+右拼接
        R = np.zeros((K, K), dtype=np.float32)
        L = np.zeros((K, K), dtype=np.float32)
        for t in range(len(ids) - 1):
            R[ids[t], ids[t + 1]] += 1
            L[ids[t + 1], ids[t]] += 1
        F = np.concatenate([L, R], axis=1)
    F = F / (F.sum(axis=1, keepdims=True) + 1e-12)
    return F


def kmeans_cosine(X, R, iters=80, restarts=20, seed=0):
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


def purity(labels, R, pos, vocab):
    total = 0
    weighted = 0.0
    for c in range(R):
        members = [vocab[i] for i in range(len(vocab)) if labels[i] == c]
        if not members:
            continue
        cnt = Counter(pos.get(w, 'OTHER') for w in members)
        weighted += cnt.most_common(1)[0][1]
        total += len(members)
    return weighted / total if total else 0.0


def class_acc(labels, ids, K, m=2):
    seq = [labels[i] for i in ids]
    table = {}
    for t in range(m, len(seq)):
        ctx = tuple(seq[t - m:t])
        nxt = seq[t]
        table.setdefault(ctx, {}).setdefault(nxt, 0)
        table[ctx][nxt] += 1
    hits = total = 0
    for t in range(m, len(seq)):
        ctx = tuple(seq[t - m:t])
        nxt = seq[t]
        d = table.get(ctx)
        if not d:
            continue
        total += 1
        if max(d, key=d.get) == nxt:
            hits += 1
    return hits / total if total else 0.0


def main():
    print('=' * 78)
    print('抽象层 v2：左/右/双侧特征，提高类纯度')
    print('=' * 78)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    ref_cls = sorted(set(pos.values()))
    ref_labels = np.array([ref_cls.index(pos[v]) for v in vocab])

    # 白给词性类的类级 acc（标尺）
    ref_acc = class_acc(ref_labels, ids, K)
    ref_purity = purity(ref_labels, len(ref_cls), pos, vocab)
    print(f'\n白给词性类（R={len(ref_cls)}）：类级 acc={ref_acc:.3f}（这是标尺）\n')

    print(f'{"特征":<8} {"R":>3} | {"纯度":>8} {"类级acc":>8} | {"vs标尺acc":>10}')
    print('-' * 46)

    for mode in ('right', 'left', 'both'):
        F = build_features(ids, K, mode)
        for R in (3, 5):
            labels = kmeans_cosine(F, R)
            p = purity(labels, R, pos, vocab)
            a = class_acc(labels, ids, K)
            print(f'{mode:<8} {R:>3} | {p:>8.3f} {a:>8.3f} | {a/ref_acc:>9.2f}')

    print()
    print('=' * 78)
    print('结论：')
    print('  1. 纯度越高 = 自动类越接近真实词性；类级 acc 越接近 0.672 = 抽象越有用。')
    print('  2. 「both」（左+右）应比单侧强：名词前 the/后动词，动词前名词/后 the，')
    print('     两侧一起才能干净分开。')
    print('  3. 若纯度仍卡在 ~0.5，说明是语料太稀疏（每词 2.1 次）的硬约束，需更大语料。')
    print('=' * 78)


if __name__ == '__main__':
    main()
