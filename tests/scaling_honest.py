"""HONEST scaling: no assumption that rules stay constant. We USE the MEASURED
rule-growth rate, and compare per-token cost: continuous K*d floats vs discrete
rules * log2(K) bits.

The insight (user's): rules CAN grow with text; what matters is they grow
SLOWLY and each is CHEAP (discrete), while the continuous scheme pays d floats
per token. The ratio then widens as scale grows.

Measured rule-growth from the corpus (scaling_rule_count.py):
  words 500 -> 32 rules, 1000 -> 67, 1870 -> 147
  => rules ~ 0.079 * words  (roughly linear, slope from data)

So we do NOT freeze rules; we let rules = 0.079 * K (measured slope), each rule
costs 2*log2(K) bits (a symbol pair), plus symbol table K*log2(K) bits.
Continuous = K*d floats.

Run:  python tests/scaling_honest.py
"""
import numpy as np


def continuous_params(K, d):
    return K * d                    # embedding: each token a d-dim vector


def discrete_params(K, rule_slope=0.079):
    bits = np.log2(K)
    symbol_table = K * bits         # K symbols, each log2(K) bits
    rules = rule_slope * K          # rules GROW with K (measured slope)
    rule_cost = rules * 2 * bits    # each rule = a symbol pair, 2*log2(K) bits
    selector = K * bits             # selector: K entries, each log2(rule) bits
    total_bits = symbol_table + rule_cost + selector
    return total_bits / 32.0        # convert bits -> float-equivalents


def main():
    print('HONEST scaling: rules GROW (measured slope), but discrete is cheap')
    print(f'{"K":>8} {"d":>5} {"continuous":>12} {"discrete":>10} {"ratio":>8}')
    print('-' * 46)
    for K, d in [(1e3, 128), (1e4, 256), (1e5, 512), (1e6, 1024),
                 (1e7, 2048)]:
        K = int(K)
        c = continuous_params(K, d)
        disc = discrete_params(K)
        print(f'{K:>8} {d:>5} {c:>12.0f} {disc:>10.0f} {c/disc:>7.0f}x')

    print('\n=> rules GROW with K, but each rule is cheap (log2 K bits) while')
    print('   continuous pays d floats/token. Does the ratio still widen?')
    print('   (This is the honest test: no frozen-rule assumption.)')


if __name__ == '__main__':
    main()
