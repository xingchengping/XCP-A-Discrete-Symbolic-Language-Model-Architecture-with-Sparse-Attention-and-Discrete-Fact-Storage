"""大模型验证：稀疏 85% (attn) + 95% (FFN) 的无损，是否随 d 增大而失效。

核心风险：d=64 小模型上"稀疏无损"成立，但大模型（d 大、多头）下可能失效。
本脚本扫描 d=64/128/256，每个 d 对比「满 vs 压缩」的 loss，看 loss 差异
是否随 d 增大而由负转正（=稀疏在大模型失效）。

Run:  python tests/verify_large.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HIST = os.path.join(ROOT, 'history')
sys.path.insert(0, ROOT)
sys.path.insert(0, HIST)

from verify_direction_ssm import load_corpus, make_batch, softmax, Transformer
from end2end import apply_compression, train


def n_params(tf, masks=None, frozen=None):
    if masks is None:
        return sum(int(np.size(v)) for v in tf.p.values())
    n = 0
    for k, v in tf.p.items():
        if frozen and k in frozen:
            continue
        if k in masks:
            n += int(masks[k].sum())
        else:
            n += int(np.size(v))
    return n


def main():
    print('=' * 74)
    print('大模型验证：稀疏无损是否随 d 增大而失效')
    print('=' * 74)

    data, stoi = load_corpus()
    V = len(stoi)
    T, B, steps, L = 32, 16, 500, 2

    print(f'\n语料 V={V}，T={T}，L={L}，steps={steps}\n')
    print(f'{"d":>4} | {"满 loss":>8} {"压缩 loss":>9} {"Δloss":>8} {"省参":>7}')
    print('-' * 52)

    for d in (64, 128):
        tf_full = Transformer(V, d, L, T, np.random.RandomState(2))
        loss_full = train(tf_full, data, B, T, steps, V)
        p_full = n_params(tf_full)

        tf_c = Transformer(V, d, L, T, np.random.RandomState(2))
        masks, frozen = apply_compression(tf_c, attn_keep=0.15, ffn_keep=0.10, seed=0)
        loss_c = train(tf_c, data, B, T, steps, V, masks=masks, frozen=frozen)
        p_c = n_params(tf_c, masks, frozen)

        print(f'{d:>4} | {loss_full:>8.3f} {loss_c:>9.3f} {loss_c-loss_full:>+8.3f} '
              f'{(1-p_c/p_full)*100:>6.1f}%')

    print()
    print('=' * 74)
    print('读法：Δloss（压缩-满）若随 d 增大保持 ≤0，则稀疏无损在大模型下也成立；')
    print('  若随 d 增大转正，则稀疏在大模型失效（10x 会塌）。')
    print('=' * 74)


if __name__ == '__main__':
    main()
