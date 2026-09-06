"""A REAL, running vocabulary-storage comparison.

This is not an arithmetic script: it
  1. builds a real vocabulary from Data/english_data.txt,
  2. really encodes sentences to integer IDs and decodes them back (verified),
  3. measures ACTUAL memory with sys.getsizeof / numpy .nbytes on real objects,
  4. compares discrete storage vs continuous embedding at several K.

Two ways a vocabulary can be stored (the deterministic layer of the advantage):

  continuous : every word -> one d-dim float32 vector  (a real K x d array)
  discrete   : every word -> one integer ID in [0, K)

The discrete side has an information-theoretic floor of K * ceil(log2 K) bits.
A real lookup table stores the ID in a fixed-width cell (int32 = 4 bytes).
A real Python dict has even more overhead (string keys + int objects + slots).
We report ALL of them, and clearly mark which number was actually allocated
and measured versus which is a formula (for K too large to allocate).

Run:  python tests/discrete_vocab_real.py
"""
import sys
import os
import numpy as np

CORPUS = os.path.join(os.path.dirname(__file__), '..', 'Data', 'english_data.txt')


# ----------------------------------------------------------------------------
# real vocabulary from the corpus
# ----------------------------------------------------------------------------
def build_real_vocab(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    words = [w for w in text.lower().split()]
    vocab = sorted(set(words))
    return words, vocab


def demo_real_lookup(words, vocab):
    """Encode the first 2 sentences to IDs and decode back; verify round-trip."""
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}
    line = words[:13]  # "the dog chased the cat yesterday a bird flew over the house quietly"
    ids = [word2id[w] for w in line]
    back = [id2word[i] for i in ids]
    assert back == line, 'round-trip failed'
    print('real lookup round-trip (first 13 tokens):')
    print('  words :', ' '.join(line))
    print('  ids   :', ids)
    print('  decode:', ' '.join(back), '(verified == words)')
    return word2id, id2word


# ----------------------------------------------------------------------------
# real memory measurement helpers
# ----------------------------------------------------------------------------
def dict_bytes(d):
    """Real recursive size of a dict: table + keys + values (deduped by id)."""
    total = sys.getsizeof(d)
    seen = set()
    for k, v in d.items():
        for obj in (k, v):
            oid = id(obj)
            if oid not in seen:
                total += sys.getsizeof(obj)
                seen.add(oid)
    return total


def packed_bits(K):
    """Information-theoretic floor: K IDs need ceil(log2 K) bits each."""
    return K * int(np.ceil(np.log2(max(K, 2))))


def main():
    print('=' * 72)
    print('REAL vocabulary-storage comparison (discrete IDs vs continuous embedding)')
    print('=' * 72)

    # ---- part 1: real corpus vocabulary, real lookup ----------------------
    words, vocab = build_real_vocab(CORPUS)
    K_corpus = len(vocab)
    print(f'\n[1] real corpus: {len(words)} tokens, {K_corpus} distinct words')
    word2id, id2word = demo_real_lookup(words, vocab)
    print(f'    real word->id dict: {dict_bytes(word2id):,} bytes '
          f'(dict + string keys + int values)')

    # ---- part 2: scale K, measure REAL memory both ways --------------------
    print('\n[2] scale K, measure REAL memory (allocated & measured, not formula)')
    print('    d schedule: d = 64 * (K/1000)^0.25   (grows with K)')
    print()
    print(f'{"K":>8} {"d":>5} | {"continuous(K*d*4)":>17} | {"packed floor":>14} '
          f'{"int32 table":>13} {"python dict":>15} | {"ratio=int32":>12} '
          f'{"ratio=packed":>13}')
    print('-' * 104)

    results = []
    for K in (1_000, 10_000, 100_000, 1_000_000):
        d = int(64 * (K / 1000) ** 0.25)

        # continuous: REAL K x d float32 array
        try:
            cont = np.zeros((K, d), dtype=np.float32)
            cont_bytes = int(cont.nbytes)
            cont_real = True
        except MemoryError:
            cont_bytes = K * d * 4
            cont_real = False

        # discrete int32 array: REAL K int32
        try:
            ids = np.zeros(K, dtype=np.int32)
            id_bytes = int(ids.nbytes)
            id_real = True
        except MemoryError:
            id_bytes = K * 4
            id_real = False

        # discrete real Python dict (only up to 1e5; 1e6 dict is slow but ok)
        if K <= 100_000:
            w2i = {f'w{i}': i for i in range(K)}
            dict_b = dict_bytes(w2i)
        else:
            w2i = None
            dict_b = None

        pk = packed_bits(K) // 8  # packed bits -> bytes

        cont_str = f'{cont_bytes:,}' + ('' if cont_real else '  [formula]')
        id_str = f'{id_bytes:,}' + ('' if id_real else '  [formula]')
        dict_str = f'{dict_b:,}' if dict_b is not None else '(skipped)'
        ratio_int32 = cont_bytes / id_bytes
        ratio_packed = cont_bytes / pk

        print(f'{K:>8,} {d:>5} | {cont_str:>17} | {pk:>14,} {id_str:>13} '
              f'{dict_str:>15} | {ratio_int32:>11.0f}x {ratio_packed:>12.0f}x')
        results.append((K, d, cont_bytes, cont_real, pk, id_bytes, id_real,
                        ratio_int32, ratio_packed))

    # ---- part 3: formula rows for K too large to allocate ------------------
    print()
    print('[3] K beyond safe allocation: FORMULA only (honestly labeled)')
    print(f'{"K":>8} {"d":>5} | {"continuous(K*d*4)":>17} | {"discrete packed(bits)":>21} '
          f'{"int32=K*4":>11} | {"ratio(int32)":>12} {"ratio(packed)":>13}')
    print('-' * 88)
    for K in (10_000_000, 100_000_000):
        K = int(K)
        d = int(64 * (K / 1000) ** 0.25)
        cont = K * d * 4
        pk = packed_bits(K)  # bits
        id32 = K * 4
        ratio_int32 = cont / id32
        ratio_packed = cont / (pk / 8)
        print(f'{K:>8,} {d:>5} | {cont:>17,} | {pk:>21,} {id32:>11,} | '
              f'{ratio_int32:>11.0f}x {ratio_packed:>12.0f}x')

    # ---- summary -----------------------------------------------------------
    print()
    print('=' * 72)
    print('SUMMARY (honest):')
    print('  - discrete info-theoretic floor : K * ceil(log2 K) bits')
    print('  - discrete real int32 table     : K * 4 bytes   (ratio vs continuous = d)')
    print('  - discrete real Python dict     : measured above (heaviest, still far below')
    print('                                   continuous once d is large)')
    print('  - continuous real embedding     : K * d * 4 bytes (measured .nbytes)')
    print('  The ratio WIDENS with K because d grows with K while log2 K barely moves.')
    print('=' * 72)


if __name__ == '__main__':
    main()
