"""事实侧压缩：类内选词能否用「稀疏高频 + 固定兜底」压缩（纯离散，无矩阵无梯度）。

用户分工：结构侧（词→类、类→类规则）我负责、已跑通；事实侧（类内选词，
dog 还是 cat）用户负责。用户的公式 B·σ((W₁⊙M)·x) 是事实侧的压缩思路，
离散化后对应：
  - W₁ 稀疏 90%   → 只存「高频的事实映射」，低频事实丢弃（稀疏计数表）
  - W₂ 冻结=固定基 → 固定的兜底映射（不训练，低频/没见过落到类内默认词）
  - σ 激活         → 命中哪个事实

本脚本在真实语料上测：把类内选词表按「高频保留 + 固定兜底」压缩，看
词级 acc 掉多少、参数省多少。全程整数 ID + 计数，无矩阵、无梯度。

Run:  python tests/discrete_fact_compress.py
"""
import re
import os
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


def main():
    print('=' * 82)
    print('事实侧压缩：类内选词「稀疏高频 + 固定兜底」（纯离散）')
    print('=' * 82)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    ref_cls = sorted(set(pos.values()))
    cls2id = {c: i for i, c in enumerate(ref_cls)}
    word_cls = [cls2id[pos[w]] for w in words]   # 每个 token 的类
    R = len(ref_cls)

    # ---- 事实侧全量表：每个 (上一个词 w1, 下一个类 c2) → 最高频 next 词 ----
    # 这是「类内选词」的刚性全量（词级 bigram 按 c2 分组）
    pair_next = defaultdict(Counter)   # (w1, c2) -> {w2: count}
    pair_freq = Counter()              # (w1, c2) 出现次数（用于排序保留）
    for t in range(1, len(ids)):
        w1 = ids[t - 1]
        w2 = ids[t]
        c2 = word_cls[t]
        pair_next[(w1, c2)][w2] += 1
        pair_freq[(w1, c2)] += 1

    # 类内全局最高频词（固定兜底 B 的离散对应：不训练，固定默认值）
    class_default = {}
    for c in range(R):
        members = Counter()
        for t in range(len(ids)):
            if word_cls[t] == c:
                members[ids[t]] += 1
        class_default[c] = members.most_common(1)[0][0] if members else 0

    # 全量表：每个 pair 的 top-1 预测词
    full_map = {p: cnt.most_common(1)[0][0] for p, cnt in pair_next.items()}
    n_full = len(full_map)

    print(f'\n语料 {len(ids)} 词，K={K}，类 R={R}')
    print(f'事实侧全量映射数（(w1,c2)→top1词）= {n_full}\n')

    # ---- 压缩：按 pair 出现频率保留 top-k%，其余用固定兜底 ----
    print(f'{"保留比例":>8} | {"映射数":>7} {"压缩率":>7} | {"词级acc":>8} {"预测0(兜底)":>10}')
    print('-' * 52)

    # 按 pair 出现频率排序
    sorted_pairs = [p for p, _ in pair_freq.most_common()]

    for keep_frac in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02):
        keep_n = int(n_full * keep_frac)
        kept = set(sorted_pairs[:keep_n])
        # 预测：保留的用 top-1，其余用类内默认词（固定兜底）
        hits = total = fallback = 0
        for t in range(1, len(ids)):
            w1 = ids[t - 1]
            w2 = ids[t]
            c2 = word_cls[t]
            p = (w1, c2)
            if p in kept:
                pred = full_map[p]
            else:
                pred = class_default[c2]   # 固定兜底（B 冻结）
                fallback += 1
            total += 1
            if pred == w2:
                hits += 1
        acc = hits / total
        ratio = 1 - keep_n / n_full
        print(f'{keep_frac:>7.0%} | {keep_n:>7} {ratio:>6.1%} | {acc:>8.3f} '
              f'{fallback:>10}')

    print()
    print('=' * 82)
    print('结论（诚实）：')
    print('  1. 这是用户公式 B·σ(W₁⊙M·x) 的离散等价：稀疏高频(W₁⊙M) + 固定兜底(B)。')
    print('  2. 关键看「保留 10%」那行：90% 压缩后词级 acc 掉多少 = 事实侧压缩的')
    print('     真实代价。若掉得多，说明事实(具体词搭配)是刚性、稀疏化伤能力。')
    print('  3. 兜底次数 = 「预测落到固定默认值」的次数，越多说明丢的事实越多。')
    print('=' * 82)


if __name__ == '__main__':
    main()
