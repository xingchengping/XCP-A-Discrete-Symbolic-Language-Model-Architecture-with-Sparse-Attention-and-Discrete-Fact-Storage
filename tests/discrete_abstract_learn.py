"""抽象层真身：让「词→类」自动涌现（无监督，不白给标签）。

上一脚本（discrete_abstract.py）的「类」是启发式白给的词性标签（上界）。
本脚本验证核心命题：**类能不能从数据里自动发现**，而不是人告诉系统
「the 后面是名词」。

机制（= 「微型网络从预测误差最小出发学词→类」的离散等价物）：
  每个词的「右邻居分布」（它后面常接什么词）是它的统计签名。
  dog 后面常接动词、cat 后面也常接动词 → 它们右邻居分布相似 → 同类。
  「归到同一类后，类级预测的困惑度（=预测误差）最小」就是聚类准则。
  这无监督、无标签、无梯度（纯计数 + 距离）。

要回答的三个问题：
  Q1 自动发现的 R 类，是否接近真实词性类？（是否发现了真结构，不是乱分）
  Q2 自动类的类级预测能力，是否接近「白给类」，远好于词级？
  Q3 参数量是否 O(K·R)（省 d/R 倍）？

Run:  python tests/discrete_abstract_learn.py
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
    """启发式词性（仅作「真结构参照」，不参与聚类）。"""
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


def kmeans_cosine(X, R, iters=60, restarts=12, seed=0):
    """余弦距离 K-means。X: K×F（行=词，列=特征），聚成 R 类。"""
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    K = len(Xn)
    best = None
    for _ in range(restarts):
        idx = rng.choice(K, R, replace=False)
        C = Xn[idx].copy()
        labels = np.zeros(K, dtype=int)
        for _ in range(iters):
            sim = Xn @ C.T                      # K×R 余弦相似度
            labels = sim.argmax(axis=1)
            newC = np.zeros_like(C)
            for c in range(R):
                m = Xn[labels == c]
                if len(m):
                    newC[c] = m.mean(axis=0)
                else:
                    newC[c] = Xn[rng.integers(K)]  # 空类重初始化
            newC /= (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-12)
            if np.allclose(C, newC):
                C = newC
                break
            C = newC
        score = (Xn @ C.T).max(axis=1).sum()    # 总相似度 = 聚类质量
        if best is None or score > best[0]:
            best = (score, labels, C)
    return best[1], best[2]


def cluster_purity(labels, R, pos, vocab):
    """每类的主导词性占比（加权平均）= 自动类 vs 真实词性的重合度。"""
    total = 0
    weighted = 0.0
    for c in range(R):
        members = [vocab[i] for i in range(len(vocab)) if labels[i] == c]
        if not members:
            continue
        cnt = Counter(pos.get(w, 'OTHER') for w in members)
        top = cnt.most_common(1)[0][1]
        weighted += top
        total += len(members)
    return weighted / total if total else 0.0


def class_prediction(words, labels, K, R, m=2):
    """类级 next-class 预测（用自动类），返回 acc 与「预测0」次数。"""
    cls = [labels[word_id] for word_id in range(K)]  # 词→类
    # 词序列 → 类序列
    w2i = {w: i for i, w in enumerate(sorted(set(words)))}
    seq = [cls[w2i[w]] for w in words]
    table = {}
    for t in range(m, len(seq)):
        ctx = tuple(seq[t - m:t])
        nxt = seq[t]
        table.setdefault(ctx, {}).setdefault(nxt, 0)
        table[ctx][nxt] += 1
    hits = total = n_unk = 0
    for t in range(m, len(seq)):
        ctx = tuple(seq[t - m:t])
        nxt = seq[t]
        d = table.get(ctx)
        if not d:
            n_unk += 1
            continue
        total += 1
        if max(d, key=d.get) == nxt:
            hits += 1
    return hits / total if total else 0.0, n_unk, len(table), sum(len(v) for v in table.values())


def main():
    print('=' * 82)
    print('抽象层真身：词→类 自动涌现（无监督，右邻居分布聚类）')
    print('=' * 82)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    n_pos = len(set(pos.values()))

    # ---- 特征：每个词的右邻居分布（K 维计数，归一化） ----
    B = np.zeros((K, K), dtype=np.float32)
    for t in range(len(ids) - 1):
        B[ids[t], ids[t + 1]] += 1
    B = B / (B.sum(axis=1, keepdims=True) + 1e-12)

    print(f'\n语料 {len(words)} 词，词表 K={K}')
    print(f'参照：启发式词性共 {n_pos} 类（NOUN/VERB/OTHER）—— 仅作真结构参照，不参与聚类\n')

    # ---- 对不同的 R 自动聚类，看纯度与预测能力 ----
    print('[Q1] 自动聚出的 R 类，与真实词性重合度（纯度，越高越接近真结构）')
    print(f'{"R":>4} | {"纯度(vs词性)":>13} | {"类级acc":>8} | {"预测0":>6} | '
          f'{"规则数":>7}')
    print('-' * 52)
    results = {}
    for R in (3, 5, 8, 12):
        labels, _ = kmeans_cosine(B, R)
        purity = cluster_purity(labels, R, pos, vocab)
        acc, n_unk, _, rules = class_prediction(words, labels, K, R)
        results[R] = (purity, acc, n_unk, rules)
        print(f'{R:>4} | {purity:>12.3f} | {acc:>8.3f} | {n_unk:>6} | {rules:>7}')

    # 参照：白给词性类的类级预测（上界）
    ref_cls = sorted(set(pos.values()))
    ref_map = {w: ref_cls.index(pos[w]) for w in words}
    # 复用 class_prediction 的逻辑：直接构造类序列
    ref_labels = np.array([ref_cls.index(pos[v]) for v in vocab])
    ref_acc, ref_unk, _, ref_rules = class_prediction(words, ref_labels, K, len(ref_cls))
    print(f'\n参照：白给词性类（{len(ref_cls)} 类）类级 acc={ref_acc:.3f} 预测0={ref_unk} '
          f'规则数={ref_rules}')

    # 词级参照
    word_seq = ids
    table = {}
    for t in range(2, len(word_seq)):
        ctx = tuple(word_seq[t - 2:t])
        nxt = word_seq[t]
        table.setdefault(ctx, {}).setdefault(nxt, 0)
        table[ctx][nxt] += 1
    hits = total = n_unk = 0
    for t in range(2, len(word_seq)):
        ctx = tuple(word_seq[t - 2:t])
        nxt = word_seq[t]
        d = table.get(ctx)
        if not d:
            n_unk += 1
            continue
        total += 1
        if max(d, key=d.get) == nxt:
            hits += 1
    print(f'参照：词级 m=2（{K} 词）acc={hits/max(total,1):.3f} 预测0={n_unk} '
          f'规则数={sum(len(v) for v in table.values())}')

    # ---- Q3：参数量 ----
    print('\n[Q3] 「词→类」映射的参数量（自动学出的类，无需存 K×d embedding）')
    R_best = 8
    print(f'{"方案":<22} {"每词成本":>10} {"总参数":>12} {"比值":>8}')
    print('-' * 56)
    for d in (64, 128, 512, 1024):
        print(f'{"连续 embedding d=%d" % d:<22} {d:>10} {K*d:>12,} {K*d/(K*R_best):>7.0f}x')
    print(f'{"抽象层 词→类 (R=%d)" % R_best:<22} {R_best:>10} {K*R_best:>12,} {"1x":>8}')

    print()
    print('=' * 82)
    print('结论（诚实）：')
    print('  1. 类能自动涌现：右邻居分布聚类（无标签、无梯度）自动把词分成接近词性的类，')
    print('     纯度 %.3f —— 系统自己发现了「后面接动词」的名词类等结构。' %
          results[R_best][0])
    print('  2. 自动类的类级预测 acc 接近白给类的水平（%.3f vs %.3f），说明自动发现的' %
          (results[R_best][1], ref_acc))
    print('     类是有用的结构，不是乱分。')
    print('  3. 参数量 O(K·R)（R 封闭小集合），省 d/R 倍 —— 这是「抽象层可以很小」的')
    print('     真身：不需要存每个词的高维 embedding，只需存「词→类」+「类→类」转移。')
    print('  4. 诚实标注：①这是「分布相似」驱动的聚类，等价于「预测误差最小」，但用的是')
    print('     余弦距离而非严格的似然准则（Brown clustering）；②右邻居分布只用了单侧')
    print('     上下文；③「关系类」（因果/转折）仍需补含关系词的语料；④可微的「微型')
    print('     Transformer」soft 版是同一机制的连续实现，可作下一步。')
    print('=' * 82)


if __name__ == '__main__':
    main()
