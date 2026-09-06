"""DiscreteEmbedding: token -> integer ID -> semantic class (discrete, no float vectors)."""
import numpy as np
from collections import Counter
from .module import Module


def _kmeans_cosine(X, R, iters=60, restarts=10, seed=0):
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    N = len(Xn)
    best = None
    for _ in range(restarts):
        idx = rng.choice(N, R, replace=False)
        C = Xn[idx].copy()
        labels = np.zeros(N, dtype=int)
        for _ in range(iters):
            sim = Xn @ C.T
            labels = sim.argmax(axis=1)
            newC = np.zeros_like(C)
            for c in range(R):
                m = Xn[labels == c]
                if len(m):
                    newC[c] = m.mean(axis=0)
                else:
                    newC[c] = Xn[rng.integers(N)]
            newC /= (np.linalg.norm(newC, axis=1, keepdims=True) + 1e-12)
            if np.allclose(C, newC):
                C = newC
                break
            C = newC
        score = (Xn @ C.T).max(axis=1).sum()
        if best is None or score > best[0]:
            best = (score, labels, C)
    return best[1]


class DiscreteEmbedding(Module):
    """Token -> integer ID -> semantic class.

    Replaces the Transformer's tokenizer + embedding, but stores integer IDs
    instead of continuous vectors: K*log2(K) bits vs K*d floats (~1000x).
    """

    def __init__(self, n_classes=8, cluster='context'):
        super().__init__()
        self.n_classes = n_classes
        self.cluster = cluster          # 'frequency' (fast) | 'context' (kmeans neighbors)
        self.w2i = {}
        self.i2w = []
        self.word_cls = None
        self._discrete = {}

    def build(self, tokens):
        """tokens: list of word strings. Returns the integer ID sequence."""
        vocab = sorted(set(tokens))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = vocab
        K = len(vocab)
        ids = [self.w2i[w] for w in tokens]
        R = min(self.n_classes, K)

        if self.cluster == 'context' and K <= 2000:
            F = np.zeros((K, 2 * K), dtype=np.float32)
            for t in range(len(ids) - 1):
                F[ids[t], ids[t + 1]] += 1
                F[ids[t + 1], K + ids[t]] += 1
            self.word_cls = _kmeans_cosine(F, R)
        else:
            # frequency bucketing: sort by frequency and split evenly into R classes (O(K), fast)
            freq = Counter(ids)
            order = sorted(range(K), key=lambda i: -freq[i])
            self.word_cls = np.zeros(K, dtype=int)
            for rank, w in enumerate(order):
                self.word_cls[w] = min(rank * R // K, R - 1)

        self._discrete = {
            'word_cls': self.word_cls,
            'i2w': self.i2w,
        }
        return ids

    def forward(self, ids):
        """ids: integer ID sequence -> class ID sequence."""
        return self.word_cls[ids]

    def class_of(self, ids):
        return self.word_cls[ids]
