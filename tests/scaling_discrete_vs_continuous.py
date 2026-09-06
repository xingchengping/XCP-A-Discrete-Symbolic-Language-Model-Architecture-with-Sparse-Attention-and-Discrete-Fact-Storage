"""Scaling comparison: DISCRETE (symbols + rule/selector) vs CONTINUOUS
(Transformer-style dense vectors). Does the parameter gap WIDEN as scale grows?

The correct target for 1000x is NOT a constant factor at small scale, but the
SLOPE: as vocabulary K and hidden dim d grow, the continuous scheme pays K*d
(every token a d-dim vector) while the discrete scheme pays ~K*log2(K) bits
(every token a symbol ID) plus a constant rule library + a small selector.

We plot both parameter curves vs K and report the RATIO as K grows.

Run:  python tests/scaling_discrete_vs_continuous.py
"""
import numpy as np


def continuous_params(K, d):
    """Transformer-style: embedding K*d + attention/FFN ~ d^2 (per layer, say 1)."""
    emb = K * d
    attn = 12 * d * d        # QKV+O+FFN, one layer, the O(d^2) part
    return emb + attn


def discrete_params(K, n_rules, selector_hidden=64):
    """Discrete symbols: K * log2(K) bits for the symbol table (bits),
    + rule library (n_rules rules, each a symbol pair ~ 2*log2(K) bits),
    + a small selector (micro network, ~constant size)."""
    bits_per_symbol = np.log2(K)
    symbol_table = K * bits_per_symbol
    rule_library = n_rules * 2 * bits_per_symbol
    selector = selector_hidden * selector_hidden   # micro net, ~constant
    # express "params" in the same unit: number of floats (assume 32 bits/float)
    total_bits = symbol_table + rule_library + selector * 32
    return total_bits / 32.0     # convert to float-equivalents for fair ratio


def main():
    print('scaling: discrete (symbol+rule+selector) vs continuous (dense)')
    print(f'{"K":>8} {"d":>5} {"continuous":>12} {"discrete":>10} {"ratio":>8}')
    print('-' * 46)

    n_rules = 200               # fixed grammar rules (does NOT grow with K)
    for K, d in [(1e3, 128), (1e4, 256), (1e5, 512), (1e6, 1024)]:
        K = int(K)
        c = continuous_params(K, d)
        disc = discrete_params(K, n_rules)
        ratio = c / disc
        print(f'{K:>8} {d:>5} {c:>12.0f} {disc:>10.0f} {ratio:>7.0f}x')

    print('\n=> as K grows, does the ratio WIDEN? That is the 1000x slope.')


if __name__ == '__main__':
    main()
