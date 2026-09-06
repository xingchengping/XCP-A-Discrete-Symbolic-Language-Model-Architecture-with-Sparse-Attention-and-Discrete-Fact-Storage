"""DiscreteFFN: structure rules (class->class) + fact lookup ((left-word, class)->word) + class fallback.

Replaces the Transformer's FFN, but uses integer tables only (no matmul). Counting
construction is a closed-form optimum: one pass over the corpus suffices (no iterative gradients).
"""
from collections import Counter, defaultdict
import numpy as np
from .module import Module


def _sample_counts(counts, temperature, rng):
    """Sample from a count distribution; temperature controls randomness (lower -> closer to argmax)."""
    words = list(counts.keys())
    w = np.array([counts[x] for x in words], dtype=np.float64)
    if temperature != 1.0:
        w = w ** (1.0 / max(temperature, 1e-3))
    p = w / w.sum()
    return int(rng.choice(words, p=p))


class DiscreteFFN(Module):
    """Discrete feed-forward: class->class m-gram (structure) + (left-word,class)->word lookup (facts) + class fallback."""

    def __init__(self, m=2):
        super().__init__()
        self.m = m
        self.class_table = {}       # class context -> {next class: count}
        self.fact_table = {}        # (left word, next class) -> {right word: count}
        self.class_default = {}     # class -> most frequent word
        self.class_word_dist = {}   # class -> {word: count} (full distribution for sampled fallback)
        self._discrete = {}

    def build(self, ids, word_cls):
        cls_seq = word_cls[ids]
        m = self.m
        # store all 1-gram .. m-gram class transitions (supports backoff to any shorter context)
        for k in range(1, m + 1):
            for t in range(k, len(cls_seq)):
                ctx = tuple(cls_seq[t - k:t])
                nxt = cls_seq[t]
                self.class_table.setdefault(ctx, {}).setdefault(nxt, 0)
                self.class_table[ctx][nxt] += 1
        for t in range(1, len(ids)):
            key = (ids[t - 1], word_cls[ids[t]])
            self.fact_table.setdefault(key, {}).setdefault(ids[t], 0)
            self.fact_table[key][ids[t]] += 1
        cls_word = defaultdict(Counter)
        for t in range(len(ids)):
            cls_word[word_cls[ids[t]]][ids[t]] += 1
        for c, cnt in cls_word.items():
            self.class_default[c] = cnt.most_common(1)[0][0]
            self.class_word_dist[c] = dict(cnt)

        self._discrete = {
            'class_table': self.class_table,
            'fact_table': self.fact_table,
            'class_default': self.class_default,
            'class_word_dist': self.class_word_dist,
        }
        return self

    def predict_class(self, cls_seq, t, fallback=None):
        """Predict the class at position t, with backoff: m-gram -> (m-1)-gram -> ... -> unigram."""
        for k in range(self.m, 0, -1):
            ctx = tuple(cls_seq[t - k:t])
            dist = self.class_table.get(ctx)
            if dist:
                return max(dist, key=dist.get), True
        if fallback is not None and (t - 1) < len(fallback):
            return fallback[t - 1], False
        return None, False

    def predict_word(self, left_id, pred_cls):
        """(left word, predicted class) -> right word (argmax); class fallback on miss."""
        f = self.fact_table.get((left_id, pred_cls))
        if f:
            return max(f, key=f.get)
        return self.class_default.get(pred_cls)

    def predict_word_sample(self, left_id, pred_cls, temperature=1.0, rng=None):
        """(left word, predicted class) -> right word (frequency sampling with temperature); class-distribution sampling on miss."""
        rng = rng or np.random.default_rng()
        f = self.fact_table.get((left_id, pred_cls))
        if f:
            return _sample_counts(f, temperature, rng)
        dist = self.class_word_dist.get(pred_cls)
        if dist:
            return _sample_counts(dist, temperature, rng)
        return None

    def forward(self, ids, word_cls, fallback=None):
        """Return the predicted word at each position (for evaluation)."""
        cls_seq = word_cls[ids]
        preds = []
        for t in range(len(ids)):
            if t < 1:
                preds.append(None)
                continue
            pc, _ = self.predict_class(cls_seq, t, fallback)
            if pc is None:
                preds.append(None)
            else:
                preds.append(self.predict_word(ids[t - 1], pc))
        return preds
