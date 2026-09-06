"""Generate a "semantically related" corpus: a real test of smooth generalization vs discrete lookup.

Core design:
  - Words are grouped by semantic class (canine/feline/bird/...); words in the
    same class share a set of "collocation verbs".
  - Each class keeps 1 "test word" that appears only in the test region (unseen in training).

Generates ~500k words: the first 90% is the train region (without test words),
the last 10% is the test region (with test words). Saved to Data/gen_corpus_sem.txt.
"""
import random
import os

GROUPS = [
    ('canine', ['dog', 'wolf', 'fox', 'coyote', 'hound'], ['run', 'bark', 'howl', 'chase']),
    ('feline', ['cat', 'tiger', 'lion', 'leopard', 'puma'], ['sleep', 'pounce', 'climb', 'stalk']),
    ('bird', ['bird', 'crow', 'owl', 'sparrow', 'eagle'], ['fly', 'sing', 'chirp', 'soar']),
    ('fish', ['fish', 'shark', 'whale', 'salmon', 'trout'], ['swim', 'dive', 'glide', 'float']),
    ('human', ['man', 'woman', 'child', 'elder', 'youth'], ['walk', 'talk', 'read', 'write']),
    ('vehicle', ['car', 'truck', 'bus', 'train', 'boat'], ['drive', 'move', 'stop', 'roll']),
    ('weather', ['rain', 'snow', 'wind', 'storm', 'hail'], ['fall', 'blow', 'howl', 'rage']),
    ('plant', ['tree', 'flower', 'grass', 'bush', 'vine'], ['grow', 'bloom', 'sway', 'wilt']),
]

DETS = ['the', 'a', 'my', 'his']
ADJS = ['big', 'small', 'fast', 'slow', 'quiet', 'old', 'new', 'bright', 'dark']


def main():
    target = 500_000
    rng = random.Random(23)
    train_sents, test_sents = [], []
    t_tr = t_te = 0

    def gen(group_words, verb):
        return f'{rng.choice(DETS)} {rng.choice(ADJS)} {rng.choice(group_words)} {verb}'

    while t_te < target * 0.1:
        name, nouns, verbs = rng.choice(GROUPS)
        s = gen([nouns[-1]], rng.choice(verbs))
        test_sents.append(s)
        t_te += len(s.split())

    while t_tr < target * 0.9:
        name, nouns, verbs = rng.choice(GROUPS)
        s = gen(nouns[:-1], rng.choice(verbs))
        train_sents.append(s)
        t_tr += len(s.split())

    text = '. '.join(train_sents + test_sents) + '.'
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data',
                        'gen_corpus_sem.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'generated {t_tr + t_te} words, vocab {len(set(text.split()))} -> {path}')
    print(f'test words: {[g[1][-1] for g in GROUPS]}')


if __name__ == '__main__':
    main()
