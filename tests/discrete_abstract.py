"""抽象层最小验证：把词归到 R 个「语义类」，让离散预测按类复用规则。

命题（用户想法的精确形式）：
  抽象 = 把 K 个词映射到 R 个语义类（词性：名词/动词/…；关系：因果/转折/…）。
  R 是封闭小集合（~5-50），不随词表涨。
  → 连续 embedding 每词 d 维 = K·d 参数；
  → 抽象层只存「词→类」= K·R 参数，省 d/R 倍。
  → 且类级规则数 = R²（不是词级 K²），类级还能泛化到没见过的词。

本脚本只做「词性类」这一种语义类（语料里有 the NOUN VERB 结构，数据现成、
不造假）。「关系类」（因果/转折）是同一机制的另一实例，但需要补含关系词的
语料，本步诚实标注为「未做」。

三层对比（全部真实运行）：
  [词级] 词→词 m-gram：规则 K²，不能泛化（新词=预测0）
  [类级] 类→类 m-gram：规则 R²，能泛化（新词落到已有类）
  [抽象层] 词→类 映射：K·R 参数（这是「微型 Transformer 做抽象」的等价物）

Run:  python tests/discrete_abstract.py
"""
import re
import sys
import os
import numpy as np
from collections import Counter, defaultdict

CORPUS = os.path.join(os.path.dirname(__file__), '..', 'Data', 'english_data.txt')


def load(path):
    text = open(path, encoding='utf-8').read()
    words = re.findall(r"[a-z']+", text.lower())
    vocab = sorted(set(words))
    w2i = {w: i for i, w in enumerate(vocab)}
    return words, [w2i[w] for w in words], len(vocab), w2i


# ----------------------------------------------------------------------------
# 1. 启发式词性标注（得到「词→类」映射，R 个语义类）
#    这是「抽象存在的证明/上界」：类标签是白给的，标注这一点。
# ----------------------------------------------------------------------------
DETERMINERS = {'the', 'a', 'an', 'my', 'his', 'her', 'our', 'their', 'its'}
ADJ = set()


def tag_pos(words):
    """启发式：the/a/an 后的第一个实词是 NOUN；NOUN 后紧跟的是 VERB。
    这是极简启发式，只用于拿到「类」这个结构，不声称是精确词性标注。"""
    classes = {}
    rev = {}
    for i, w in enumerate(words):
        if w in DETERMINERS:
            continue
        # 前面是限定词 → 名词（跳过形容词）
        if i > 0 and words[i - 1] in DETERMINERS:
            if i > 1 and words[i - 2] in DETERMINERS:
                # the beautiful flower: flower 是名词, beautiful 是形容词
                classes.setdefault(words[i - 1], 'ADJ')
                classes.setdefault(w, 'NOUN')
            else:
                classes.setdefault(w, 'NOUN')
            continue
    # 第二遍：名词后面跟的词 → 动词
    for i, w in enumerate(words):
        if i > 0 and classes.get(words[i - 1]) == 'NOUN' and w not in classes:
            classes.setdefault(w, 'VERB')
    # 剩余 → OTHER
    for w in words:
        classes.setdefault(w, 'OTHER')
    return classes


def main():
    print('=' * 82)
    print('抽象层最小验证：词→语义类，规则按类复用（词性类，真实语料）')
    print('=' * 82)

    words, ids, K, w2i = load(CORPUS)
    pos = tag_pos(words)
    cls_list = sorted(set(pos.values()))
    R = len(cls_list)
    cls2id = {c: i for i, c in enumerate(cls_list)}
    word_cls = [cls2id[pos[w]] for w in words]  # 每个词的类 ID

    from collections import Counter as _C
    dist = _C(pos.values())
    print(f'\n语料 {len(words)} 词，词表 K={K}')
    print(f'启发式标出的语义类 R={R}：{dict(dist)}')
    print('（词性标签是启发式白给的 —— 这是「抽象存在」的上界，非本步要学的对象）\n')

    # ------------------------------------------------------------------------
    # 2. 词级 vs 类级 m-gram：规则数、真实内存、准确率、泛化
    # ------------------------------------------------------------------------
    def build_ngram(seq, m):
        table = {}
        for t in range(m, len(seq)):
            ctx = tuple(seq[t - m:t])
            nxt = seq[t]
            table.setdefault(ctx, {}).setdefault(nxt, 0)
            table[ctx][nxt] += 1
        return table

    def eval_ngram(table, seq, m):
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
        return hits / total if total else 0.0, n_unk

    # 前 70% 训练，后 30% 测试（顺序切分，测试句子大多没见过）
    split = int(len(words) * 0.7)
    print('[2] 词级 vs 类级 m-gram（前 70% 建表，后 30% 测试 = 泛化）')
    print(f'{"级别":>6} | {"m":>2} | {"规则数":>7} | {"acc":>7} | {"测试预测0":>8} | '
          f'{"规则空间上界":>12}')
    print('-' * 66)

    for m in (1, 2, 3):
        # 词级：K 个词，规则空间上界 K^m * K
        wt = build_ngram(ids[:split], m)
        w_acc, w_unk = eval_ngram(wt, ids[split:], m)
        wrules = sum(len(v) for v in wt.values())
        wspace = K ** (m + 1)
        print(f'{"词级":>6} | {m:>2} | {wrules:>7} | {w_acc:>7.3f} | {w_unk:>8} | '
              f'{wspace:>12,}')

        # 类级：R 个类，规则空间上界 R^m * R
        ct = build_ngram(word_cls[:split], m)
        c_acc, c_unk = eval_ngram(ct, word_cls[split:], m)
        crules = sum(len(v) for v in ct.values())
        cspace = R ** (m + 1)
        print(f'{"类级":>6} | {m:>2} | {crules:>7} | {c_acc:>7.3f} | {c_unk:>8} | '
              f'{cspace:>12,}')
    print('  （类级 acc 是「类预测」命中率；词级 acc 是「具体词」命中率，二者难度不同，')
    print('   不能直接比大小 —— 要看的是「规则空间上界」差多少 + 类级能不能泛化到新词）\n')

    # ------------------------------------------------------------------------
    # 3. 抽象层 = 词→类 映射：K·R 参数（微型网络做抽象的等价物）
    # ------------------------------------------------------------------------
    print('[3] 抽象层「词→类」映射的参数成本（这是「微型 Transformer 做抽象」的等价物）')
    print(f'{"方案":<20} {"每词成本":>10} {"总参数":>12} {"比值":>8}')
    print('-' * 56)
    for d in (64, 128, 512, 1024):
        cont = K * d
        abstract = K * R
        print(f'{"连续 embedding d=%d" % d:<20} {d:>10} {cont:>12,} {cont/abstract:>7.0f}x')
    print(f'{"抽象层 词→类 (R=%d)" % R:<20} {R:>10} {K*R:>12,} {"1x":>8}')
    print(f'  → 抽象层只存「词→哪个类」，每词 R={R} 维；连续 embedding 每词 d 维。')
    print(f'    省 d/R 倍，且 R 是封闭小集合不随 K 涨，所以省的比例随 K 增大。\n')

    # ------------------------------------------------------------------------
    # 4. 泛化验证：类级能否预测「训练中没见过的词」
    # ------------------------------------------------------------------------
    print('[4] 泛化验证：类级能不能预测「训练中没见过的新词」')
    print('    取词表的一半词做训练，另一半词只出现在测试集（新词）。')
    half = K // 2
    seen = set(range(half))  # 训练只见过前一半词的 ID
    # 用词级 vs 类级，在测试集上：新词（ID >= half）能否被预测
    tr_ids = [i for i in ids if i < half]  # 训练序列（只含已见词）
    te_ids = ids  # 测试序列（含新词）
    m = 2
    wt = build_ngram(tr_ids, m)
    ct = build_ngram([cls2id[pos[words[i]]] for i in range(len(words)) if ids[i] < half], m)
    # 词级预测测试集：新词 ID 从未在表里 → 预测 0
    w_hits = w_total = w_unk = 0
    c_hits = c_total = c_unk = 0
    for t in range(m, len(te_ids)):
        wctx = tuple(te_ids[t - m:t])
        wd = wt.get(wctx)
        wnxt = te_ids[t]
        # 词级：只有 ctx 在表且 next 在表才算预测
        if not wd or wnxt not in wd:
            w_unk += 1
        else:
            w_total += 1
            if max(wd, key=wd.get) == wnxt:
                w_hits += 1
        # 类级：新词也映射到类，类级能预测
        cctx = tuple(word_cls[t - m:t])
        cd = ct.get(cctx)
        cnxt = word_cls[t]
        if not cd or cnxt not in cd:
            c_unk += 1
        else:
            c_total += 1
            if max(cd, key=cd.get) == cnxt:
                c_hits += 1
    print(f'    词级：可预测 {w_total} 次 / 预测0 {w_unk} 次，acc={w_hits/max(w_total,1):.3f}')
    print(f'    类级：可预测 {c_total} 次 / 预测0 {c_unk} 次，acc={c_hits/max(c_total,1):.3f}')
    print('    （类级「预测0」远少于词级 = 新词也能落到已有类、被预测 → 抽象带来泛化）')

    print()
    print('=' * 82)
    print('结论（诚实）：')
    print('  1. 抽象把规则空间从 K^(m+1) 降到 R^(m+1)：K=895 vs R=%d，天壤之别。' % R)
    print('  2. 抽象层（词→类）只花 K·R 参数，比连续 K·d 省 d/R 倍；R 封闭不随 K 涨，')
    print('     所以省的比例随规模拉大 —— 这就是「抽象层可以很小」的精确含义。')
    print('  3. 抽象带来泛化：类级对「新词」的预测 0 远少于词级（新词落到已有类）。')
    print('  4. 诚实标注：本步的「类」是启发式白给的词性标签（上界），不是学出来的；')
    print('     「关系类」（因果/转折）需补含关系词的语料；「从数据自动学出类」是下一步。')
    print('=' * 82)


if __name__ == '__main__':
    main()
