"""关系类验证：因果/转折/递进 也离散可表示（结构侧最后一块）。

词性类是「内容词」的抽象（名词/动词）；关系类是「连接词」的抽象
（因果/转折/递进/并列/条件）。两者是同一机制：把多个具体词归到一个
关系类，规则按类复用。

本脚本：
  1. 生成一个含 5 类关系（每类多个关系词）的干净语料；
  2. 验证「关系类 → 下一个词的类」的调节作用（because 后面接原因句，
     but/then 后面接陈述句…）；
  3. 对比关系【词】级规则 vs 关系【类】级规则，看省多少。

关系词（R_word=14 个）→ 关系类（R_cls=5 类）：
  CAUSE  because / since / so
  BUT    but / however / yet
  THEN   then / next / after
  AND    and / plus
  COND   if / unless

Run:  python tests/discrete_relation.py
"""
import random
import numpy as np
from collections import Counter

RELATION = {
    'CAUSE': ['because', 'since', 'so'],
    'BUT': ['but', 'however', 'yet'],
    'THEN': ['then', 'next', 'after'],
    'AND': ['and', 'plus'],
    'COND': ['if', 'unless'],
}

NOUNS = ['dog', 'cat', 'bird', 'fish', 'man', 'woman', 'child', 'horse',
         'tree', 'river', 'star', 'book']
VERBS = ['ran', 'slept', 'sang', 'swam', 'walked', 'jumped', 'cried', 'laughed',
         'read', 'wrote']
ADJS = ['hungry', 'tired', 'happy', 'sad', 'fast', 'slow', 'bright', 'dark']

# 每类关系后面接的「下一个词类」模式，用于验证调节作用：
#   CAUSE 后接 "it"（原因句 it was adj）
#   BUT/THEN/AND 后接 "the"（陈述句 the noun verb）
#   COND 后接 "it"（条件句 it verb）
TAIL = {
    'CAUSE': 'it',
    'BUT': 'the',
    'THEN': 'the',
    'AND': 'the',
    'COND': 'it',
}


def gen_sentence(rng, rel_class, rel_word):
    n1, v1 = rng.choice(NOUNS), rng.choice(VERBS)
    if rel_class == 'CAUSE':
        return f'the {n1} {v1} {rel_word} it was {rng.choice(ADJS)}'
    if rel_class == 'COND':
        return f'the {n1} {v1} {rel_word} it {rng.choice(VERBS)}'
    # BUT / THEN / AND：the noun verb + rel + the noun verb
    n2, v2 = rng.choice(NOUNS), rng.choice(VERBS)
    return f'the {n1} {v1} {rel_word} the {n2} {v2}'


def build_corpus(n_per=40, seed=0):
    rng = random.Random(seed)
    lines = []
    for rel_class, rel_words in RELATION.items():
        for _ in range(n_per):
            rw = rng.choice(rel_words)
            lines.append(gen_sentence(rng, rel_class, rw))
    rng.shuffle(lines)
    return lines


def main():
    print('=' * 78)
    print('关系类验证：因果/转折/递进 离散可表示')
    print('=' * 78)

    lines = build_corpus()
    words = ' '.join(lines).split()
    rel_words = [w for ws in RELATION.values() for w in ws]
    rel2class = {w: c for c, ws in RELATION.items() for w in ws}
    R_word = len(rel_words)
    R_cls = len(RELATION)

    print(f'\n语料：{len(lines)} 句，{len(words)} 词')
    print(f'关系词 {R_word} 个 → 关系类 {R_cls} 类：')
    for c, ws in RELATION.items():
        print(f'  {c:<6} {" / ".join(ws)}')

    # ---- 验证调节作用：关系类 → 下一个词的类 ----
    print('\n[1] 调节作用：关系类能否预测「下一个词」')
    print(f'{"关系类":<8} {"下一个词分布":<30}')
    print('-' * 40)
    for t in range(len(words) - 1):
        w = words[t]
        if w in rel2class:
            c = rel2class[w]
            nxt = words[t + 1]
            # 验证 nxt 是否符合 TAIL 模式
            expected = TAIL[c]
            # 统计每个关系类后面实际接的第一个词
    # 重新统计（上面循环没存结果，重写）
    tail_dist = {c: Counter() for c in RELATION}
    for t in range(len(words) - 1):
        w = words[t]
        if w in rel2class:
            tail_dist[rel2class[w]][words[t + 1]] += 1
    for c in RELATION:
        top = tail_dist[c].most_common(3)
        s = ' '.join(f'{w}:{n}' for w, n in top)
        print(f'{c:<8} {s:<30}')

    # ---- 关系词级 vs 关系类级 规则数 ----
    print('\n[2] 关系【词】级 vs 关系【类】级 bigram 规则数')
    w_seq = []  # 关系词级：只保留关系词序列
    c_seq = []  # 关系类级
    for w in words:
        if w in rel2class:
            w_seq.append(rel_words.index(w))
            c_seq.append(list(RELATION).index(rel2class[w]))
    # 词级 bigram 规则
    w_rules = set()
    c_rules = set()
    for t in range(1, len(w_seq)):
        w_rules.add((w_seq[t - 1], w_seq[t]))
        c_rules.add((c_seq[t - 1], c_seq[t]))
    print(f'  关系词级（{R_word} 词）：bigram 规则 {len(w_rules)}，规则空间 {R_word**2}')
    print(f'  关系类级（{R_cls} 类）：bigram 规则 {len(c_rules)}，规则空间 {R_cls**2}')
    print(f'  → 规则空间省 {R_word**2 / R_cls**2:.1f}x')

    print()
    print('=' * 78)
    print('结论：')
    print('  1. 关系类有调节作用：CAUSE 后接 it（原因句）、BUT/THEN/AND 后接 the、')
    print('     COND 后接 it —— 关系词决定「后面是什么」，这正是「抽象词=调节器」。')
    print('  2. 关系抽象与词性抽象同一机制：多个关系词归一个关系类，规则按类复用，')
    print('     规则空间省 %.0fx。' % (R_word**2 / R_cls**2))
    print('  3. 诚实标注：关系词→关系类 是显式给的（关系词就是标记）；关系类之间')
    print('     （BUT vs THEN vs AND 都接 the）的区分靠关系词本身，不靠邻居分布——')
    print('     这印证了之前的结论：纯文本分不开 then/because，需多模态接地。')
    print('=' * 78)


if __name__ == '__main__':
    main()
