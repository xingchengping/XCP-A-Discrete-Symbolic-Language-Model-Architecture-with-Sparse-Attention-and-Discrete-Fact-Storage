"""Generate a comparison corpus: structured text with long-range coreference.

Design: a paragraph = topic sentence (introduces an entity) + coreference
sentences (pronoun referring back) + relation sentence (because/but).
The long-range dependency is: the pronoun it/he refers to an earlier entity,
so predicting the verb after it requires knowing what it refers to (exactly
where attention earns its keep).

Generates about `target_words` words, saved to Data/gen_corpus.txt.
"""
import random
import os

NOUNS = ['dog', 'cat', 'bird', 'fish', 'horse', 'rabbit', 'mouse', 'wolf',
         'bear', 'fox', 'deer', 'sheep', 'lion', 'tiger', 'monkey', 'frog',
         'duck', 'hen', 'crow', 'owl']
VERBS = ['ran', 'jumped', 'slept', 'sang', 'swam', 'flew', 'crept', 'climbed',
         'barked', 'chased', 'ate', 'drank', 'hid', 'walked', 'danced', 'fought']
ADJS = ['fast', 'slow', 'loud', 'quiet', 'big', 'small', 'quick', 'gentle',
        'wild', 'tame', 'bright', 'dark', 'happy', 'sad', 'brave', 'shy']
DETS = ['the', 'a']
PRONOUNS = {'he': ['dog', 'horse', 'wolf', 'bear', 'fox', 'lion', 'tiger'],
            'she': ['cat', 'bird', 'rabbit', 'sheep', 'deer', 'mouse', 'frog',
                    'duck', 'hen', 'crow', 'owl']}
RELATIONS = ['because', 'but', 'then', 'so', 'and', 'after']


def make_paragraph(rng):
    n1 = rng.choice(NOUNS)
    v1 = rng.choice(VERBS)
    det1 = rng.choice(DETS)
    lines = [f'{det1} {n1} {v1}']
    pron = 'he' if n1 in PRONOUNS['he'] else 'she'
    for _ in range(rng.randint(1, 3)):
        v2 = rng.choice(VERBS)
        obj = rng.choice(NOUNS)
        det2 = rng.choice(DETS)
        lines.append(f'{pron} {v2} {det2} {obj}')
    rel = rng.choice(RELATIONS)
    adj = rng.choice(ADJS)
    lines.append(f'{rel} it was {adj}')
    return ' . '.join(lines) + ' .'


def main():
    target = 500_000
    rng = random.Random(42)
    out = []
    total = 0
    while total < target:
        para = make_paragraph(rng)
        out.append(para)
        total += len(para.split())
    text = ' '.join(out)
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data',
                        'gen_corpus.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'generated {total} words -> {path}')


if __name__ == '__main__':
    main()
