"""XCP full architecture — layers (discrete layers + sparse layers).

This is the layer implementation for the full "discrete-symbolic + sparse"
technical stack, mapped to Transformer components:

  Transformer           ->   XCP
  ------------------------- -------------------------------
  Tokenizer / Embedding ->   DiscreteEmbedding (word -> integer ID -> class)
  FFN                   ->   DiscreteFFN (structure rules + fact lookup + class fallback)
  Attention             ->   SparseAttention (sparse Q/K/V/O 85%)
  (no equivalent, XCP-only) -> RelationGate (cause/contrast regulator)

Every layer exposes a uniform interface: build() learns / forward() runs.
Discrete layers count (no gradient); sparse layers use gradients (trained
from scratch). See README.md.
"""
import re
import numpy as np
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def kmeans_cosine(X, R, iters=60, restarts=10, seed=0):
    """Cosine-distance K-means. X: N x F, cluster into R classes, return labels."""
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


# ---------------------------------------------------------------------------
# Layer 1: discrete Embedding (word -> integer ID -> class)
# ---------------------------------------------------------------------------
class DiscreteEmbedding:
    """word -> integer ID -> semantic class (left+right neighbor distribution clustering, unsupervised).

    Maps to Transformer's Tokenizer + Embedding, but replaces continuous
    vectors with discrete IDs, saving ~1000x (storage layer).
    """

    def __init__(self, n_classes=8):
        self.n_classes = n_classes
        self.w2i = {}
        self.i2w = []
        self.word_cls = None   # word ID -> class ID

    def build(self, words, fast=True):
        """Learn the vocabulary + word->class mapping from a word sequence.

        fast=True: frequency bucketing (O(K), fast, for large real corpora).
        fast=False: left+right neighbor distribution clustering (kmeans, slow, small vocab).
        """
        vocab = sorted(set(words))
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.i2w = vocab
        K = len(vocab)
        ids = [self.w2i[w] for w in words]
        R = min(self.n_classes, K)

        if fast or K > 2000:
            # Frequency bucketing: sort by frequency and split evenly into R classes.
            freq = Counter(ids)
            order = sorted(range(K), key=lambda i: -freq[i])
            self.word_cls = np.zeros(K, dtype=int)
            for rank, w in enumerate(order):
                self.word_cls[w] = min(rank * R // K, R - 1)
        else:
            F = np.zeros((K, 2 * K), dtype=np.float32)
            for t in range(len(ids) - 1):
                F[ids[t], ids[t + 1]] += 1
                F[ids[t + 1], K + ids[t]] += 1
            self.word_cls = kmeans_cosine(F, R)
        return ids

    def class_of(self, ids):
        return self.word_cls[ids]


# ---------------------------------------------------------------------------
# Layer 2: discrete FFN (structure rules + fact lookup + class fallback)
# ---------------------------------------------------------------------------
class DiscreteFFN:
    """Discrete FFN: class->class structure rules (m-gram) + (left-word, class)->word fact lookup + class fallback.

    Maps to Transformer's FFN, but everything is integer tables, no matmul:
      - structure side: class->class m-gram count table (generalization / grammar)
      - fact side: (left-word ID, next class) -> right-word ID lookup (memorize high-frequency collocations)
      - fallback: miss -> class-level transition (class, class) -> word (smoothing)

    Measured (fact_lookup_hybrid.py): test-set accuracy ties the sparse FFN,
    using 112x fewer parameters.
    """

    def __init__(self, m=2):
        self.m = m
        self.class_table = {}   # class context -> {next class: count}
        self.fact_table = {}    # (left word, next class) -> {right word: count}
        self.class_default = {} # class -> most frequent word

    def build(self, ids, word_cls):
        """Scan the corpus and count into tables."""
        cls_seq = word_cls[ids] if word_cls is not None else ids
        m = self.m
        # Structure: class->class m-gram.
        for t in range(m, len(cls_seq)):
            ctx = tuple(cls_seq[t - m:t])
            nxt = cls_seq[t]
            self.class_table.setdefault(ctx, {}).setdefault(nxt, 0)
            self.class_table[ctx][nxt] += 1
        # Facts: (left word, next class) -> right word.
        for t in range(1, len(ids)):
            key = (ids[t - 1], word_cls[ids[t]] if word_cls is not None else ids[t])
            self.fact_table.setdefault(key, {}).setdefault(ids[t], 0)
            self.fact_table[key][ids[t]] += 1
        # Class fallback: most frequent word per class.
        cls_word = defaultdict(Counter)
        for t in range(len(ids)):
            c = word_cls[ids[t]] if word_cls is not None else ids[t]
            cls_word[c][ids[t]] += 1
        for c, cnt in cls_word.items():
            self.class_default[c] = cnt.most_common(1)[0][0]

    def predict_next(self, ids, word_cls, t):
        """Predict ids[t] given the history. Return a word ID or None."""
        cls_seq = word_cls[ids] if word_cls is not None else ids
        m = self.m
        if t < 1:
            return None
        # 1) Structure: class context -> predict next class.
        ctx = tuple(cls_seq[t - m:t])
        dist = self.class_table.get(ctx)
        if not dist:
            return None
        pred_cls = max(dist, key=dist.get)
        # 2) Facts: (left word, predicted class) -> right word; miss -> class fallback.
        key = (ids[t - 1], pred_cls)
        f = self.fact_table.get(key)
        if f:
            return max(f, key=f.get)
        return self.class_default.get(pred_cls)


# ---------------------------------------------------------------------------
# Layer 3: sparse Attention (optional, long-range mixing)
# ---------------------------------------------------------------------------
class SparseAttention:
    """Sparse Q/K/V/O attention (85% sparse, trained from scratch).

    Maps to Transformer's Attention, but each of the four projection matrices
    keeps only 15% of its connections.
    Measured (../tests/attention_sparse.py): retrieval capability is preserved,
    and it does not degrade with d.
    """

    def __init__(self, d, keep_frac=0.15, seed=0):
        self.d = d
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d)
        self.WQ = rng.standard_normal((d, d)) * scale
        self.WK = rng.standard_normal((d, d)) * scale
        self.WV = rng.standard_normal((d, d)) * scale
        self.WO = rng.standard_normal((d, d)) * scale
        self.masks = {}
        for name in ('WQ', 'WK', 'WV', 'WO'):
            m = np.zeros((d, d), dtype=bool)
            k = max(1, int(d * keep_frac))
            for j in range(d):
                m[j, rng.choice(d, size=k, replace=False)] = True
            self.masks[name] = m
            setattr(self, name, getattr(self, name) * m)

    def forward(self, X):
        """X: (T, d) -> contextual representation (T, d)."""
        Q = X @ self.WQ.T
        K = X @ self.WK.T
        V = X @ self.WV.T
        scores = Q @ K.T / np.sqrt(self.d)
        probs = softmax_rows(scores)
        ctx = probs @ V
        return ctx @ self.WO.T


# ---------------------------------------------------------------------------
# Layer 4: relation gate (cause/contrast regulator, XCP-only)
# ---------------------------------------------------------------------------
class RelationGate:
    """Relation word -> relation class -> regulation signal (cause/contrast/sequence).

    Maps to "abstract words = regulators": 'because' introduces a cause clause,
    'but' introduces a statement.
    """

    def __init__(self, relations=None):
        self.relations = relations or {
            'CAUSE': ['because', 'since', 'so'],
            'BUT': ['but', 'however', 'yet'],
            'THEN': ['then', 'next', 'after'],
        }
        self.word2rel = {w: c for c, ws in self.relations.items() for w in ws}
        self.rel_next = {}   # relation word ID -> most frequent next-word ID

    def relation_of(self, word):
        return self.word2rel.get(word)

    def build(self, ids, i2w):
        """Learn from the corpus: what a relation word is most often followed by."""
        cnt = defaultdict(Counter)
        for t in range(len(ids) - 1):
            if i2w[ids[t]] in self.word2rel:
                cnt[ids[t]][ids[t + 1]] += 1
        self.rel_next = {w: c.most_common(1)[0][0] for w, c in cnt.items()}

    def next_word(self, left_word_id):
        """Left word is a relation word -> return its next word (else None)."""
        return self.rel_next.get(left_word_id)


def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
