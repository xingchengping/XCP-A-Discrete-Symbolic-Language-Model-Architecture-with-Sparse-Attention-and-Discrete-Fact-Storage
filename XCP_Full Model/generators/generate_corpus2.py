"""Generate a harder corpus: larger vocabulary (400+ words), more varied structure, a large combination space.

The key to "hard": a big vocabulary, varied sentence patterns, and a large
word-combination space -> most train/test sentence combinations do not
overlap -> forcing "smooth generalization" (having seen "the dog ran", the
model can infer "the cat ran").

Generates ~500k words, saved to Data/gen_corpus_hard.txt.
"""
import random
import os
import re

ANIMALS = ['dog', 'cat', 'bird', 'fish', 'horse', 'rabbit', 'mouse', 'wolf',
           'bear', 'fox', 'deer', 'sheep', 'lion', 'tiger', 'monkey', 'frog',
           'duck', 'hen', 'crow', 'owl', 'snake', 'eagle', 'whale', 'shark',
           'spider', 'ant', 'bee', 'cow', 'pig', 'goat']
PEOPLE = ['man', 'woman', 'boy', 'girl', 'child', 'teacher', 'student', 'doctor',
          'nurse', 'farmer', 'baker', 'cook', 'driver', 'writer', 'artist',
          'singer', 'pilot', 'soldier', 'king', 'queen', 'friend', 'neighbor',
          'mother', 'father', 'brother', 'sister', 'baby', 'uncle']
OBJECTS = ['book', 'table', 'chair', 'door', 'window', 'house', 'car', 'road',
           'tree', 'flower', 'river', 'mountain', 'sky', 'sun', 'moon', 'star',
           'cloud', 'rain', 'snow', 'wind', 'fire', 'water', 'stone', 'wood',
           'paper', 'cup', 'plate', 'knife', 'ball', 'box', 'key', 'clock',
           'lamp', 'bed', 'shoe', 'hat', 'coat', 'bag']
CONCEPTS = ['love', 'hope', 'fear', 'joy', 'anger', 'peace', 'truth', 'dream',
            'idea', 'time', 'life', 'light', 'music', 'art', 'beauty', 'wisdom']
NOUNS = ANIMALS + PEOPLE + OBJECTS + CONCEPTS

VERBS = ['run', 'jump', 'sleep', 'sing', 'swim', 'fly', 'walk', 'talk', 'read',
         'write', 'eat', 'drink', 'see', 'hear', 'feel', 'think', 'know', 'want',
         'give', 'take', 'make', 'build', 'break', 'open', 'close', 'buy', 'sell',
         'teach', 'learn', 'help', 'love', 'find', 'lose', 'keep', 'leave',
         'come', 'go', 'stay', 'live', 'work', 'play', 'rest', 'look', 'watch',
         'listen', 'speak', 'smile', 'cry', 'laugh', 'dance', 'drive', 'ride',
         'climb', 'fall', 'grow', 'change', 'move', 'stop', 'start', 'wait',
         'meet']
ADJS = ['big', 'small', 'fast', 'slow', 'loud', 'quiet', 'young', 'old', 'new',
        'good', 'bad', 'happy', 'sad', 'bright', 'dark', 'hot', 'cold', 'warm',
        'cool', 'soft', 'hard', 'strong', 'weak', 'brave', 'kind', 'rich', 'poor',
        'clean', 'full', 'empty', 'wet', 'dry', 'deep', 'high', 'low', 'near',
        'far', 'early', 'late', 'easy', 'true', 'long', 'short', 'wide', 'narrow',
        'sweet', 'smooth', 'rough']
ADVS = ['quickly', 'slowly', 'loudly', 'quietly', 'happily', 'sadly', 'carefully',
        'suddenly', 'gradually', 'always', 'never', 'often', 'early', 'late',
        'well', 'gently', 'fiercely', 'calmly', 'proudly', 'shyly']
PREPS = ['in', 'on', 'at', 'with', 'from', 'to', 'of', 'for', 'by', 'about',
         'over', 'under', 'near', 'after', 'before']
CONJS = ['because', 'but', 'then', 'and', 'so', 'while', 'although', 'if',
         'since', 'unless']
DETS = ['the', 'a', 'my', 'his', 'her', 'our', 'their']
PRONS = ['he', 'she', 'it', 'they']

TEMPLATES = [
    '{det} {noun} {verb}',
    '{det} {adj} {noun} {verb} {adv}',
    '{det} {noun} {verb} {prep} {det} {noun}',
    '{pron} {verb} {det} {adj} {noun}',
    '{det} {noun} {verb} {conj} {det} {noun} {verb}',
    '{det} {noun} {verb} {conj} {pron} {verb} {adv}',
    '{det} {noun} {verb} {prep} {det} {adj} {noun}',
    '{pron} {verb} {prep} {det} {noun} {adv}',
]


def make_sentence(rng):
    tpl = rng.choice(TEMPLATES)
    pools = {'det': DETS, 'noun': NOUNS, 'verb': VERBS, 'adj': ADJS,
             'adv': ADVS, 'prep': PREPS, 'conj': CONJS, 'pron': PRONS}
    return re.sub(r'\{(\w+)\}', lambda m: rng.choice(pools[m.group(1)]), tpl)


def main():
    target = 500_000
    rng = random.Random(7)
    sents = []
    total = 0
    while total < target:
        s = make_sentence(rng)
        sents.append(s)
        total += len(s.split())
    text = '. '.join(sents) + '.'
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data',
                        'gen_corpus_hard.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    vocab = set(text.split())
    print(f'generated {total} words, vocab {len(vocab)} -> {path}')


if __name__ == '__main__':
    main()
