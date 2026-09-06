"""A COMPLETE training + inference discrete-symbol system, measured as the
corpus GROWS. We record the ACTUAL parameter count (rules + selector + symbol
IDs) at each corpus size, plus real prediction accuracy, to see how parameters
grow with learning (the user's point: params DO grow, but cheaply).

Design (full coverage, no top-K truncation):
  - symbols = integer IDs
  - rule library: one rule per DISTINCT (from,to) transition observed
  - selector: for each symbol, the rule id of its (most frequent) next symbol
  - training = streaming over the corpus; at checkpoints we measure:
      * actual params = 2*(#rules) + (#selector entries) + (#symbols)
      * accuracy = predict next symbol on the seen data

Run:  python tests/discrete_full.py
"""
import re
from collections import defaultdict, Counter


def load_words(limit):
    text = open(r'H:\RTE\Data\english_data.txt', encoding='utf-8').read()
    return re.findall(r"[a-z']+", text.lower())[:limit]


class DiscreteFull:
    def __init__(self):
        self.w2i = {}
        self.i2w = []
        self.rules = []          # list of (from_id, to_id), one per distinct transition
        self.rule_idx = {}       # (from,to) -> rule id
        self.selector = {}       # from_id -> rule id (most frequent next)
        self.next_count = defaultdict(Counter)   # from_id -> Counter(next_id)

    def _id(self, w):
        if w not in self.w2i:
            self.w2i[w] = len(self.i2w)
            self.i2w.append(w)
        return self.w2i[w]

    def train(self, words):
        for i in range(len(words) - 1):
            a, b = self._id(words[i]), self._id(words[i + 1])
            self.next_count[a][b] += 1
            if (a, b) not in self.rule_idx:
                self.rule_idx[(a, b)] = len(self.rules)
                self.rules.append((a, b))
        # rebuild selector: for each from_id, the rule of its most frequent next
        self.selector = {}
        for a, cnt in self.next_count.items():
            best_next = cnt.most_common(1)[0][0]
            self.selector[a] = self.rule_idx[(a, best_next)]

    def predict(self, a):
        if a in self.selector:
            return self.rules[self.selector[a]][1]
        return None

    def n_params(self):
        # actual storage: each rule = 2 ints; selector = 1 int/entry; symbols = 1 int each
        return 2 * len(self.rules) + len(self.selector) + len(self.i2w)

    def accuracy(self, words):
        correct = total = 0
        for i in range(len(words) - 1):
            a = self.w2i[words[i]]
            p = self.predict(a)
            if p is not None:
                total += 1
                if p == self.w2i[words[i + 1]]:
                    correct += 1
        return correct / total if total else 0.0


def main():
    print('COMPLETE discrete system: training + inference, params vs corpus growth')
    print('-' * 62)
    print(f'{"corpus":>7} {"vocab":>6} {"rules":>6} {"selector":>9} '
          f'{"params":>7} {"accuracy":>9} {"cont.(d=128)":>12} {"ratio":>7}')

    for limit in (200, 400, 800, 1200, 1870):
        words = load_words(limit)
        ds = DiscreteFull()
        ds.train(words)
        acc = ds.accuracy(words)
        p = ds.n_params()
        cont = len(ds.i2w) * 128
        print(f'{limit:>7} {len(ds.i2w):>6} {len(ds.rules):>6} '
              f'{len(ds.selector):>9} {p:>7} {acc:>9.3f} {cont:>12} {cont/p:>6.0f}x')

    print('\n=> params GROW with corpus (rules/selector/symbols all grow), but')
    print('   the discrete cost per symbol is a few ints vs 128 floats continuous.')


if __name__ == '__main__':
    main()
