"""分层预测闭环：先定类，再类内选词（class-based 预测）。

把「抽象」和「词级预测」拼起来的最小闭环：
  P(下一个词) = P(下一个类 | 历史) × P(词 | 类)
  第一步用类级 m-gram 预测下一个类（便宜，R^(m+1) 规则）；
  第二步在类内选具体词。

这直接回答一个关键问题：抽象 + 离散 ID + 规则库(类→类) + 类内选词，
拼起来到底能不能做「词级预测」？拿多少 acc 换多少参数？

诚实对比三种：
  [词级 m-gram]  直接词→词，参数 K^(m+1)，acc 最高（参照）
  [分层-弱]      类级 m-gram + 类内全局词频（最省参，类内不看上下文）
  [分层-强]      类级 m-gram + 类内 bigram（类内也看上一个词，参数介于中间）

Run:  python tests/discrete_hierarchical.py
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


def build_table(seq, m):
    table = {}
    for t in range(m, len(seq)):
        ctx = tuple(seq[t - m:t])
        nxt = seq[t]
        table.setdefault(ctx, {}).setdefault(nxt, 0)
        table[ctx][nxt] += 1
    return table


def n_rules(table):
    return sum(len(v) for v in table.values())


def main():
    print('=' * 84)
    print('分层预测闭环：先定类 → 再类内选词')
    print('=' * 84)

    words, ids, K, w2i, vocab = load(CORPUS)
    pos = tag_pos(words)
    ref_cls = sorted(set(pos.values()))
    word_cls = np.array([ref_cls.index(pos[v]) for v in vocab])  # 词ID→类ID
    R = len(ref_cls)
    cls_seq = [word_cls[i] for i in ids]  # 词序列→类序列
    word_freq = Counter(ids)              # 词级词频（类内选词用）

    print(f'\n语料 {len(words)} 词，K={K}，白给词性类 R={R}（NOUN/VERB/OTHER）\n')

    print(f'{"方案":<16} {"m":>2} | {"词级acc":>8} {"规则数":>8} | {"参数(规则+类内)":>16}')
    print('-' * 62)

    for m in (1, 2, 3):
        # ---- 词级 m-gram（参照，最贵最准） ----
        wt = build_table(ids, m)
        # 词级 acc
        hits = total = 0
        for t in range(m, len(ids)):
            ctx = tuple(ids[t - m:t])
            nxt = ids[t]
            d = wt.get(ctx)
            if not d:
                continue
            total += 1
            if max(d, key=d.get) == nxt:
                hits += 1
        w_acc = hits / total if total else 0.0
        w_rules = n_rules(wt)

        # ---- 类级 m-gram（类→类） ----
        ct = build_table(cls_seq, m)
        c_rules = n_rules(ct)

        # ---- 分层-弱：先类（类级 m-gram），再类内全局词频 ----
        # 类内全局词频：每类最高频词
        best_in_class = {}
        for c in range(R):
            members = [i for i in range(K) if word_cls[i] == c]
            best_in_class[c] = max(members, key=lambda i: word_freq[i])
        hits = total = 0
        for t in range(m, len(ids)):
            ctx = tuple(cls_seq[t - m:t])
            nxt = ids[t]
            d = ct.get(ctx)
            if not d:
                continue
            pred_c = max(d, key=d.get)
            pred_w = best_in_class[pred_c]
            total += 1
            if pred_w == nxt:
                hits += 1
        weak_acc = hits / total if total else 0.0

        # ---- 分层-强：先类，再类内 bigram（类内也看上一个词） ----
        # 对每个类 c，用「上一个词 w_{t-1} → 类内词 w_t」做类内 bigram
        within = {}  # c -> {(prev_word, next_word): count}
        for t in range(1, len(ids)):
            c = word_cls[ids[t]]
            within.setdefault(c, Counter())[(ids[t - 1], ids[t])] += 1
        hits = total = 0
        for t in range(m, len(ids)):
            ctx = tuple(cls_seq[t - m:t])
            nxt = ids[t]
            d = ct.get(ctx)
            if not d:
                continue
            pred_c = max(d, key=d.get)
            wd = within.get(pred_c, {})
            # 类内 bigram：给上一个词，预测类内下一个词
            key_prev = ids[t - 1]
            candidates = {nw: c for (pw, nw), c in wd.items() if pw == key_prev}
            if not candidates:
                # 退化到类内全局词频
                pred_w = best_in_class[pred_c]
            else:
                pred_w = max(candidates, key=candidates.get)
            total += 1
            if pred_w == nxt:
                hits += 1
        strong_acc = hits / total if total else 0.0
        within_rules = sum(len(v) for v in within.values())

        print(f'{"词级":<16} {m:>2} | {w_acc:>8.3f} {w_rules:>8} | {w_rules:>16}')
        print(f'{"分层-弱":<16} {m:>2} | {weak_acc:>8.3f} {c_rules:>8} | '
              f'{c_rules + K:>16}')
        print(f'{"分层-强":<16} {m:>2} | {strong_acc:>8.3f} {c_rules:>8} | '
              f'{c_rules + within_rules:>16}')

    print()
    print('=' * 84)
    print('结论（诚实）：')
    print('  · 抽象把「先定类」这步的规则数从 K^(m+1) 压到 R^(m+1)，省得极狠。')
    print('  · 但「类内选词」才是词级预测的主体成本，抽象省不掉它——')
    print('    分层-弱（类内只看全局词频）词级 acc 很低，因为它退化成「每类猜一个')
    print('    最高频词」，完全没看上下文。')
    print('  · 分层-强（类内也看上一个词）能找回大部分词级 acc，但类内规则数又上来了')
    print('    —— 说明「词级区分」是刚性的，抽象只能把它「分层」，不能把它「消掉」。')
    print('  · 所以「抽象 + 离散 ID + 规则库/选择器」拼起来，能跑通一个省参的离散 LM，')
    print('    但省的是「结构层」的钱；词级区分的钱，靠的是「分层后每层都稀疏」。')
    print('=' * 84)


if __name__ == '__main__':
    main()
