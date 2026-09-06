"""XCP trainer: interactive parameters -> train -> save .xcp -> enter chat mode.

Usage (inside the XCP_library directory):
    python train.py

Interactively enter: corpus path, output path, class count, context length,
whether to use attention (and its d / depth / epochs). After training it saves
a .xcp model, then enters chat mode (type text, XCP continues it).
"""
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xcp
import xcp.nn as nn

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'Data', 'english_data.txt')


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def ask(prompt, default=None, cast=str):
    """Interactive prompt; press Enter to use the default."""
    suffix = f' [{default}]' if default is not None else ''
    val = input(f'{prompt}{suffix}: ').strip().lstrip('\ufeff')
    if val == '' and default is not None:
        return default
    return cast(val)


def main():
    print('=' * 60)
    print('XCP Trainer - train a .xcp model and chat with it')
    print('=' * 60)

    # ---- interactive parameters ----
    corpus = ask('Corpus path', DEFAULT_CORPUS)
    if not os.path.exists(corpus):
        print(f'[error] file not found: {corpus}')
        return
    out_path = ask('Output .xcp path', os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'model.xcp'))
    n_classes = ask('Class count n_classes', 50, int)
    m = ask('Context length m', 2, int)
    use_attn = ask('Use sparse attention? (y/n)', 'n').lower().startswith('y')

    d = depth = epochs = None
    if use_attn:
        d = ask('Attention dimension d', 64, int)
        depth = ask('Attention depth', 2, int)
        epochs = ask('Attention training epochs', 100, int)

    # ---- load data + build model ----
    print('\n[training...]')
    with open(corpus, encoding='utf-8', errors='ignore') as f:
        tokens = tokenize(f.read())
    print(f'corpus: {len(tokens)} tokens')

    model = nn.XCPModel(
        n_classes=n_classes, m=m,
        use_attention=use_attn, d=d or 64,
        attention_depth=depth or 2)

    # stage 1: discrete counting
    ids = model.build(tokens)
    print(f'vocabulary: {model.vocab_size()}')

    # stage 2: sparse attention (optional)
    if use_attn:
        model.fit_attention(epochs=epochs or 100, lr=0.05)

    # ---- report ----
    split = int(len(ids) * 0.9)
    acc, miss = model.evaluate(ids, split, len(ids))
    p = model.n_params()
    print(f'\ndiscrete params (integer entries): {p["int"]:,}')
    print(f'sparse params (float nonzeros): {p["float"]:,}')
    print(f'test top-1 accuracy: {acc:.3f} (miss {miss})')

    # ---- save ----
    xcp.save(model, out_path)
    print(f'\nsaved model -> {out_path}')

    # ---- chat mode ----
    print('\n' + '=' * 60)
    print('Chat mode: type English text, XCP continues it (type quit to exit)')
    print('Tip: /temp 0.8 sets temperature (lower = more certain, higher = more random), /temp 1 restores default')
    print('=' * 60)
    emb = model.embedding
    rng = np.random.default_rng()
    temperature = 1.0
    while True:
        text = input('\nYou: ').strip()
        if text.lower() in ('quit', 'exit', 'q'):
            break
        if text.lower().startswith('/temp'):
            try:
                temperature = float(text.split()[1])
                print(f'[temperature set to {temperature}]')
            except (IndexError, ValueError):
                print('[usage: /temp 0.8]')
            continue
        words = tokenize(text)
        cur = [emb.w2i[w] for w in words if w in emb.w2i]
        if not cur:
            print('XCP: (no known words, try another input)')
            continue
        generated = []
        cur_arr = np.array(cur, dtype=np.int64)
        for _ in range(8):
            pred = model.sample_next(cur_arr, len(cur_arr),
                                     temperature=temperature, rng=rng)
            if pred is None:
                break
            generated.append(emb.i2w[pred])
            cur_arr = np.append(cur_arr, pred)
        print('XCP:', ' '.join(generated) if generated else '(cannot predict)')


if __name__ == '__main__':
    main()
