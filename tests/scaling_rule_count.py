"""NO-WATER test: does the number of GRAMMAR RULES saturate as corpus grows?

The 1000x claim rests on ONE unverified assumption: the rule library (grammar
rules) does NOT grow with vocabulary/corpus size. This test measures it
DIRECTLY on real text — no hand-set constants.

Method: on the real English corpus, grow the amount of text, and at each size
measure the number of DISTINCT structural rules needed. Here "rule" = a
positional template (e.g. the pattern of part-of-speech-like roles), discovered
by a plain grammar-induction (MDL digram/chunk) — the SAME rule count we would
store. The question: does rule-count flatten out, or keep growing with text?

We use a simple, honest proxy for "rule": the number of distinct n-gram TYPES
that recur (structural units), which is the standard proxy for grammar-size in
induction. We report it vs text size.

Run:  python tests/scaling_rule_count.py
"""
import re
from collections import Counter


def load_words(limit):
    text = open(r'H:\RTE\Data\english_data.txt', encoding='utf-8').read()
    words = re.findall(r"[a-z']+", text.lower())
    return words[:limit]


def rule_count(words, n=2):
    """Distinct recurring n-gram types (a proxy for structural rules)."""
    c = Counter()
    for i in range(len(words) - n + 1):
        c[tuple(words[i:i + n])] += 1
    # a "rule" = a recurring n-gram (appears >= 2 times) = a reusable structure
    return sum(1 for v in c.values() if v >= 2)


def main():
    total = len(load_words(10 ** 9))
    print(f'total corpus words: {total}')
    print('does the number of recurring rules saturate as text grows?')
    print(f'{"text words":>12} {"recurring rules":>16} {"rules/100 words":>15}')
    print('-' * 46)

    sizes = [500, 1000, 1870]   # up to full corpus (1870 words)
    for s in sizes:
        words = load_words(s)
        rc = rule_count(words)
        print(f'{s:>12} {rc:>16} {rc / max(s,1) * 100:>15.1f}')

    print('\n=> if "recurring rules" GROWS ~linearly with text, the rule library')
    print('   does NOT saturate -> the 1000x assumption FAILS at this scale.')
    print('   (this corpus is tiny; the real test needs millions of words.)')


if __name__ == '__main__':
    main()
