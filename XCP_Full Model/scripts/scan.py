"""Scan XCP configurations to find the best accuracy on real corpus.

Sweeps: n_classes x context length m (frequency-bucketing clustering, fast).
Real corpus: Pride and Prejudice.

Run: python scan.py
"""
import re
import os
import sys
import time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # XCP_Full Model
sys.path.insert(0, ROOT)
from model import XCPModel
from layers import tokenize

CORPUS = os.path.join(os.path.dirname(os.path.dirname(ROOT)), 'Data',
                      'real', 'pride_and_prejudice.txt')


def main():
    with open(CORPUS, encoding='utf-8', errors='ignore') as f:
        words = tokenize(f.read())
    n = len(words)
    split = int(n * 0.9)
    print(f'corpus {n} words, vocab {len(set(words))}\n')

    print(f'{"n_classes":>10} {"m":>3} | {"test acc":>8} {"build s":>8}')
    print('-' * 40)
    for n_classes in (400, 800, 1600, 3200):
        for m in (3, 4):
            t0 = time.time()
            model = XCPModel(n_classes=n_classes, m=m, use_attention=False)
            model.build(words)
            te_acc, _ = model.evaluate(split, n)
            print(f'{n_classes:>10} {m:>3} | {te_acc:>8.3f} {time.time()-t0:>8.1f}')


if __name__ == '__main__':
    main()
