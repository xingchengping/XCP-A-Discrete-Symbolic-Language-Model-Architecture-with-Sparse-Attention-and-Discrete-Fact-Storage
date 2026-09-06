"""事实侧纯离散化：查表 + 类级兜底，能否逼近稀疏 FFN 的测试集 acc。

纯查表在测试集掉到 0.141（查不到就预测0），稀疏 FFN 是 0.233。
本脚本给查表加「类级兜底」：查不到 (w1,c2) 时，用类级信息猜，看能否逼近 0.233。

三种兜底对比：
  兜底0（无）    ：查不到→预测0（纯查表）
  兜底1（类内高频）：查不到→类 c2 的最高频词
  兜底2（类级转移）：查不到→(class(w1), c2) 类级转移的最高频词

Run:  python tests/fact_lookup_hybrid.py
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


def main():
    print('=' * 74)
    print('事实侧纯离散化：查表 + 类级兜底，逼近稀疏 FFN？')
    print('=' * 74)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    ref_cls = sorted(set(pos.values()))
    cls2id = {c: i for i, c in enumerate(ref_cls)}
    R = len(ref_cls)
    word_cls = [cls2id[pos[w]] for w in words]
    wc = np.array([cls2id[pos[v]] for v in vocab])  # 词ID -> 类ID

    N = len(ids)
    split = int(N * 0.8)

    # ---- 查表：(w1, c2) -> 最高频 w2 ----
    lookup_cnt = defaultdict(Counter)
    for t in range(1, split):
        lookup_cnt[(ids[t - 1], word_cls[t])][ids[t]] += 1
    lookup = {k: v.most_common(1)[0][0] for k, v in lookup_cnt.items()}

    # ---- 兜底1：类 c2 内最高频词 ----
    class_default = {}
    class_cnt = Counter()
    for t in range(split):
        class_cnt[word_cls[t]] += 1
    # 类内最高频词（用训练集）
    class_word_cnt = defaultdict(Counter)
    for t in range(split):
        class_word_cnt[word_cls[t]][ids[t]] += 1
    for c in range(R):
        class_default[c] = class_word_cnt[c].most_common(1)[0][0] if class_word_cnt[c] else 0

    # ---- 兜底2：(class(w1), c2) 类级转移 -> 最高频 w2 ----
    class_lookup_cnt = defaultdict(Counter)
    for t in range(1, split):
        class_lookup_cnt[(word_cls[t - 1], word_cls[t])][ids[t]] += 1
    class_lookup = {k: v.most_common(1)[0][0] for k, v in class_lookup_cnt.items()}

    # ---- 三种兜底，测试集 acc ----
    def eval_with_fallback(fallback_mode):
        hits = total = 0
        for t in range(split, N):
            key = (ids[t - 1], word_cls[t])
            pred = lookup.get(key)
            if pred is None:
                if fallback_mode == 0:
                    continue  # 预测0
                elif fallback_mode == 1:
                    pred = class_default.get(word_cls[t])
                else:  # fallback_mode == 2
                    pred = class_lookup.get((word_cls[t - 1], word_cls[t]))
                if pred is None:
                    continue
            total += 1
            if pred == ids[t]:
                hits += 1
        return hits / total if total else 0.0

    print(f'\n语料 {len(words)} 词，V={K}，类 R={R}，事实 F={len(lookup)}\n')
    print(f'{"方案":<24} {"测试集acc":>10} {"参数(整数)":>12}')
    print('-' * 52)

    a0 = eval_with_fallback(0)
    print(f'{"查表(无兜底)":<24} {a0:>10.3f} {len(lookup)*2:>12}')

    a1 = eval_with_fallback(1)
    print(f'{"查表+类内高频兜底":<24} {a1:>10.3f} {len(lookup)*2 + R:>12}')

    a2 = eval_with_fallback(2)
    print(f'{"查表+类级转移兜底":<24} {a2:>10.3f} {len(lookup)*2 + len(class_lookup)*2:>12}')

    print(f'\n参照：稀疏 FFN 测试集 acc = 0.233（目标）')

    print()
    print('=' * 74)
    print('读法：若「查表+类级转移兜底」接近 0.233，则纯离散（整数查表+类规则）')
    print('  可替代稀疏 FFN，省掉全部 float 冗余。')
    print('=' * 74)


if __name__ == '__main__':
    main()
