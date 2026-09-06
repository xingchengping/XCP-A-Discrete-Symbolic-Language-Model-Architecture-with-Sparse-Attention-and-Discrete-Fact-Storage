"""Generate a medium-difficulty corpus: vocabulary ~140, 4 structures.

Sits between the simple corpus (64 words, too rigid) and the hard one (447
words, too noisy), pushing the next-token ceiling to ~0.2-0.3 so both sides
learn something meaningful.

Generates ~500k words, saved to Data/gen_corpus_mid.txt.
"""
import random
import os
import re

NOUNS = (['dog', 'cat', 'bird', 'fish', 'horse', 'rabbit', 'mouse', 'wolf',
          'bear', 'fox', 'lion', 'tiger', 'monkey', 'frog', 'duck', 'crow',
          'owl', 'snake', 'cow', 'pig'] +
         ['man', 'woman', 'boy', 'girl', 'child', 'teacher', 'doctor', 'farmer',
          'cook', 'driver', 'writer', 'artist', 'singer', 'friend', 'neighbor'] +
         ['book', 'table', 'chair', 'door', 'house', 'car', 'road', 'tree',
          'flower', 'river', 'mountain', 'sun', 'moon', 'star', 'rain', 'wind',
          'fire', 'water', 'ball', 'box'])
VERBS = ['run', 'jump', 'sleep', 'sing', 'swim', 'fly', 'walk', 'talk', 'read',
         'write', 'eat', 'drink', 'see', 'hear', 'feel', 'think', 'know', 'want',
         'give', 'take', 'make', 'build', 'open', 'close', 'buy', 'teach',
         'learn', 'help', 'love', 'find', 'lose', 'come', 'go', 'stay', 'live',
         'work', 'play', 'look', 'watch', 'listen', 'smile', 'cry', 'laugh']
ADJS = ['big', 'small', 'fast', 'slow', 'loud', 'quiet', 'young', 'old', 'new',
        'good', 'bad', 'happy', 'sad', 'bright', 'dark', 'hot', 'cold', 'warm',
        'soft', 'hard', 'strong', 'weak', 'kind', 'rich', 'clean']
ADVS = ['quickly', 'slowly', 'loudly', 'quietly', 'happily', 'sadly', 'always',
        'never']
PREPS = ['in', 'on', 'at', 'with', 'from', 'to', 'of', 'over']
CONJS = ['because', 'but', 'then', 'and', 'while']
DETS = ['the', 'a', 'my', 'his']
PRONS = ['he', 'she', 'it', 'they']

TEMPLATES = [
    '{det} {noun} {verb}',
    '{det} {adj} {noun} {verb} {adv}',
    '{det} {noun} {verb} {prep} {det} {noun}',
    '{pron} {verb} {det} {noun}',
]


def make_sentence(rng):
    tpl = rng.choice(TEMPLATES)
    pools = {'det': DETS, 'noun': NOUNS, 'verb': VERBS, 'adj': ADJS,
             'adv': ADVS, 'prep': PREPS, 'conj': CONJS, 'pron': PRONS}
    return re.sub(r'\{(\w+)\}', lambda m: rng.choice(pools[m.group(1)]), tpl)


def main():
    target = 500_000
    rng = random.Random(11)
    sents = []
    total = 0
    while total < target:
        s = make_sentence(rng)
        sents.append(s)
        total += len(s.split())
    text = '. '.join(sents) + '.'
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data',
                        'gen_corpus_mid.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'generated {total} words, vocab {len(set(text.split()))} -> {path}')


if __name__ == '__main__':
    main()
