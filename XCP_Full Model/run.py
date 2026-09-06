"""XCP full model — run entrypoint (two stages: discrete counting + sparse gradients).

Usage (from the XCP_Full Model directory):
    python run.py            # pure discrete chain
    python run.py --attn     # discrete chain + sparse attention

Flow:
    stage 1 build(): discrete counting (vocab + word->class + class rules + fact table + fallback)
    stage 2 fit_attention(): sparse attention gradients (with --attn)
    evaluation: next-token prediction accuracy on train/test + parameter count
"""
import os
import sys
from layers import tokenize

CORPUS = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'english_data.txt')


def main():
    from model import XCPModel

    use_attn = '--attn' in sys.argv
    depth = 2
    if '--depth' in sys.argv:
        depth = int(sys.argv[sys.argv.index('--depth') + 1])

    with open(CORPUS, encoding='utf-8') as f:
        words = tokenize(f.read())

    n = len(words)
    split = int(n * 0.8)

    print('=' * 68)
    print('XCP full model — next-token prediction'
          + (f' (discrete chain + {depth} sparse attention layers)' if use_attn else ' (pure discrete chain)'))
    print('=' * 68)
    print(f'corpus {n} words (train {split} / test {n - split})\n')

    model = XCPModel(n_classes=8, m=2, use_attention=use_attn, d=64,
                     attention_depth=depth)

    # Stage 1: discrete counting.
    model.build(words)

    # Stage 2: sparse gradients (optional).
    if use_attn:
        print(f'stage 2: training {depth} sparse attention layers (next-class prediction on the class-vector sequence)...')
        model.fit_attention(epochs=300, lr=0.05)
        print('  done\n')

    tr_acc, tr_miss = model.evaluate(1, split)
    te_acc, te_miss = model.evaluate(split, n)
    p = model.n_params()

    print(f'train acc = {tr_acc:.3f} (predicted None {tr_miss} times)')
    print(f'test acc = {te_acc:.3f} (predicted None {te_miss} times)')
    print(f'\nparameters:')
    print(f'  discrete (integer entries) = {p["int"]:,} (~{p["int"]*2/1024:.1f} KB)')
    print(f'  sparse (nonzero floats) = {p["float"]:,} (~{p["float"]*4/1024:.1f} KB)')

    print()
    print('=' * 68)
    print('Note: stage 1 (discrete chain) is fully working; stage 2 (sparse attention) is a long-range mixing enhancement.')
    print('Maps to README stages 1-2.')
    print('=' * 68)


if __name__ == '__main__':
    main()
