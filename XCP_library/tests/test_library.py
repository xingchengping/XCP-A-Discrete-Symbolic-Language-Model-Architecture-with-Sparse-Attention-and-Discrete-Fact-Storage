"""XCP library smoke test: build -> evaluate -> fit_attention -> save/load round-trip."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xcp
import xcp.nn as nn

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'Data', 'pride_and_prejudice.txt')


def load_tokens(path):
    with open(path, encoding='utf-8') as f:
        words = re.findall(r"[a-z']+", f.read().lower())
    return words


def main():
    print('=== XCP library smoke test ===')
    tokens = load_tokens(CORPUS)
    print(f'corpus: {len(tokens)} tokens')

    # 1) pure discrete chain
    model = nn.XCPModel(n_classes=8, m=2, use_attention=False)
    ids = model.build(tokens)
    print(f'vocab: {model.vocab_size()}, ids: {len(ids)}')
    split = int(len(ids) * 0.8)
    acc, miss = model.evaluate(ids, split, len(ids))
    print(f'discrete chain: test acc={acc:.3f}, miss={miss}')
    p = model.n_params()
    print(f'params: int={p["int"]}, float={p["float"]}')

    # 2) add sparse attention (CPU training)
    model2 = nn.XCPModel(n_classes=8, m=2, use_attention=True, d=32,
                         attention_depth=2)
    model2.build(tokens)
    model2.fit_attention(epochs=50, lr=0.05)
    acc2, _ = model2.evaluate(ids, split, len(ids))
    print(f'with attention: test acc={acc2:.3f}')

    # 3) save/load round-trip
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp_model.xcp')
    xcp.save(model2, path)
    loaded = xcp.load(path)
    acc3, _ = loaded.evaluate(ids, split, len(ids))
    print(f'after load: test acc={acc3:.3f} (should == {acc2:.3f})')
    assert abs(acc3 - acc2) < 1e-6, 'save/load round-trip mismatch'
    os.remove(path)
    print('save/load round-trip OK')

    print('=== all tests passed ===')


if __name__ == '__main__':
    main()
