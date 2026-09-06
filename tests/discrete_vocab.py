"""The FIRST, DETERMINISTIC layer of discrete advantage: how the VOCABULARY is
stored. Discrete = one integer ID per word (log2 K bits); continuous = one
d-dim vector per word (d floats). This advantage needs NO abstraction — it is
just storage. As K grows, the gap widens because log2 K grows much slower than d.

This isolates ONLY the vocabulary-storage layer, cleanly:
  continuous vocab cost = K * d floats
  discrete   vocab cost = K * log2(K) bits  (an integer ID per word)
Report the ratio as K grows, with d growing like real transformers (d ~ O(sqrt K)
or a realistic schedule).

Run:  python tests/discrete_vocab.py
"""
import numpy as np


def continuous_cost(K, d):
    return K * d                    # floats


def discrete_cost(K):
    bits = np.log2(K)
    return K * bits                 # bits (an integer ID per word)


def main():
    print('vocabulary-storage layer: discrete (ID) vs continuous (d-vector)')
    print('(this advantage needs NO abstraction — pure storage)')
    print(f'{"K":>8} {"d":>5} {"continuous(floats)":>19} {"discrete(bits)":>16} '
          f'{"ratio":>8}')
    print('-' * 62)

    # realistic d schedule: d grows with K, e.g. d = 64 * (K/1000)^0.25
    for K in (1e3, 1e4, 1e5, 1e6, 1e7, 1e8):
        K = int(K)
        d = int(64 * (K / 1000) ** 0.25)
        c = continuous_cost(K, d)
        disc = discrete_cost(K)
        # ratio: continuous cost in floats vs discrete in floats (bits/32)
        ratio = c / (disc / 32.0)
        print(f'{K:>8} {d:>5} {c:>19.0f} {disc:>16.0f} {ratio:>7.0f}x')

    print('\n=> discrete vocab cost = K*log2 K bits grows MUCH slower than')
    print('   continuous K*d floats, so the ratio widens with K. This is the')
    print('   deterministic (no-abstraction-needed) part of 1000x.')


if __name__ == '__main__':
    main()
