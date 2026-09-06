"""RelationGate: discourse connectives (cause/adversative/sequential) -> regulator signal."""
from collections import Counter, defaultdict
from .module import Module


class RelationGate(Module):
    """Relation word -> relation class -> next-word regulator (because precedes a reason, but precedes a statement)."""

    def __init__(self, relations=None):
        super().__init__()
        self.relations = relations or {
            'CAUSE': ['because', 'since', 'so'],
            'BUT': ['but', 'however', 'yet'],
            'THEN': ['then', 'next', 'after'],
        }
        self.word2rel = {w: c for c, ws in self.relations.items() for w in ws}
        self.rel_next = {}
        self._discrete = {}

    def build(self, ids, i2w):
        cnt = defaultdict(Counter)
        for t in range(len(ids) - 1):
            if i2w[ids[t]] in self.word2rel:
                cnt[ids[t]][ids[t + 1]] += 1
        self.rel_next = {w: c.most_common(1)[0][0] for w, c in cnt.items()}
        self._discrete = {'rel_next': self.rel_next}
        return self

    def relation_of(self, word):
        return self.word2rel.get(word)

    def next_word(self, left_word_id):
        return self.rel_next.get(left_word_id)
