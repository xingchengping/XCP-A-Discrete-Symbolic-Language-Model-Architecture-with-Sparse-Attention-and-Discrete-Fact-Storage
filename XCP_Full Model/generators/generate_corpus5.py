"""Generate the "smooth generalization" corpus v5: every word appears in training, but half the collocations are held out for test.

Tests true "within-class collocation generalization":
  - Each class has 4 nouns + 4 verbs, and every word appears in the train region.
  - Train region: each noun pairs only with the "first 2 verbs".
  - Test region: each noun pairs with the "last 2 verbs" (a collocation unseen in training).

Generates ~500k words, saved to Data/gen_corpus_sem2.txt.
"""
import random
import os

GROUPS = [
    ('canine', ['dog', 'wolf', 'fox', 'coyote'], ['run', 'bark', 'howl', 'chase']),
    ('feline', ['cat', 'tiger', 'lion', 'leopard'], ['sleep', 'pounce', 'climb', 'stalk']),
    ('bird', ['bird', 'crow', 'owl', 'sparrow'], ['fly', 'sing', 'chirp', 'soar']),
    ('fish', ['fish', 'shark', 'whale', 'salmon'], ['swim', 'dive', 'glide', 'float']),
    ('human', ['man', 'woman', 'child', 'elder'], ['walk', 'talk', 'read', 'write']),
    ('vehicle', ['car', 'truck', 'bus', 'train'], ['drive', 'move', 'stop', 'roll']),
    ('weather', ['rain', 'snow', 'wind', 'storm'], ['fall', 'blow', 'howl', 'rage']),
    ('plant', ['tree', 'flower', 'grass', 'bush'], ['grow', 'bloom', 'sway', 'wilt']),
]

DETS = ['the', 'a', 'my', 'his']
ADJS = ['big', 'small', 'fast', 'slow', 'quiet', 'old', 'bright', 'dark']


def main():
    target = 500_000
    rng = random.Random(31)
    train_sents, test_sents = [], []
    t_tr = t_te = 0

    while t_tr < target * 0.9:
        name, nouns, verbs = rng.choice(GROUPS)
        s = f'{rng.choice(DETS)} {rng.choice(ADJS)} {rng.choice(nouns)} {rng.choice(verbs[:2])}'
        train_sents.append(s)
        t_tr += len(s.split())

    while t_te < target * 0.1:
        name, nouns, verbs = rng.choice(GROUPS)
        s = f'{rng.choice(DETS)} {rng.choice(ADJS)} {rng.choice(nouns)} {rng.choice(verbs[2:])}'
        test_sents.append(s)
        t_te += len(s.split())

    text = '. '.join(train_sents + test_sents) + '.'
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data',
                        'gen_corpus_sem2.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'generated {t_tr + t_te} words, vocab {len(set(text.split()))} -> {path}')


if __name__ == '__main__':
    main()
