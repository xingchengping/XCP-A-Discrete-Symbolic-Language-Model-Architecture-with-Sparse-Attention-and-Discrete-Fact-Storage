"""XCP full model — composes the layers into an end-to-end next-token predictor.

Where Transformer is "Embedding -> [Attention + FFN] x L -> output head", XCP is:

    word -> DiscreteEmbedding (word -> ID -> class)
         -> [L layers of SparseAttention refine the class-vector sequence (optional)]
         -> DiscreteFFN (structure rules class->class + fact lookup (left-word, class)->word + class fallback)
         -> RelationGate (relation-word regulation)
         -> output word

Two-stage training (matching "discrete counting + sparse gradients"):
    stage 1 build(): scan the corpus and count (vocab, class rules, fact table, fallback table) — no gradient
    stage 2 fit_attention(): train the L sparse attention layers one by one — gradient

Usage: see run.py.
"""
import numpy as np
from layers import (DiscreteEmbedding, DiscreteFFN, SparseAttention,
                    RelationGate, softmax_rows)


class XCPModel:
    def __init__(self, n_classes=8, m=2, use_attention=False, d=64,
                 attention_depth=2):
        self.n_classes = n_classes
        self.m = m
        self.embedding = DiscreteEmbedding(n_classes)
        self.ffn = DiscreteFFN(m)
        self.relation = RelationGate()
        self.attention_depth = attention_depth
        self.attentions = ([SparseAttention(d) for _ in range(attention_depth)]
                           if use_attention else [])
        self.class_vectors = None
        self.attn_next_class = None
        self.word_cls = None
        self.ids = None

    # ------------------------------------------------------------------
    # Stage 1: discrete counting (no gradient)
    # ------------------------------------------------------------------
    def build(self, words):
        self.ids = self.embedding.build(words)
        self.word_cls = self.embedding.word_cls
        self.cls_seq = self.word_cls[self.ids]   # cache the class sequence
        self.ffn.build(self.ids, self.word_cls)
        self.relation.build(self.ids, self.embedding.i2w)
        return self.ids

    # ------------------------------------------------------------------
    # Stage 2: sparse gradients (train L attention layers one by one)
    # ------------------------------------------------------------------
    def _forward_attention(self, X, C):
        """Forward pass of the L attention layers. Returns (logits, caches, Xs)."""
        Xs = [X]
        caches = []
        for attn in self.attentions:
            Xl = Xs[-1]
            Q = Xl @ attn.WQ.T
            K = Xl @ attn.WK.T
            V = Xl @ attn.WV.T
            scores = Q @ K.T / np.sqrt(attn.d)
            probs = softmax_rows(scores)
            ctx = probs @ V
            Xn = ctx @ attn.WO.T
            caches.append((Q, K, V, probs, ctx, Xl))
            Xs.append(Xn)
        logits = Xs[-1] @ C.T
        return logits, caches, Xs

    def fit_attention(self, epochs=300, lr=0.05, seed=0, window=64):
        """Train the L sparse attention layers on the class-vector sequence for next-class prediction.

        Trains with "windows" (each step takes a T=window slice) to avoid the
        O(N^2) score matrix of full-sequence attention (infeasible at N ~ 390k).
        """
        if not self.attentions:
            raise ValueError('fit_attention requires use_attention=True')
        d = self.attentions[0].d
        R = self.n_classes
        rng = np.random.default_rng(seed)
        C = rng.standard_normal((R, d)) / np.sqrt(d)
        cls_seq = self.word_cls[self.ids]
        # Slice windows: wins[i] is the input classes, tgts[i] the matching next-class.
        wins, tgts = [], []
        for i in range(0, len(cls_seq) - window, window):
            wins.append(cls_seq[i:i + window])
            tgts.append(cls_seq[i + 1:i + window + 1])
        if not wins:
            raise ValueError('corpus too short to fill one window')

        params = {'C': C}
        for l, attn in enumerate(self.attentions):
            params[f'WQ_{l}'] = attn.WQ
            params[f'WK_{l}'] = attn.WK
            params[f'WV_{l}'] = attn.WV
            params[f'WO_{l}'] = attn.WO
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        t = 0
        self.train_losses = []
        for ep in range(epochs):
            bi = rng.integers(0, len(wins))
            win = wins[bi]
            tgt = tgts[bi]
            X = C[win]                       # (window, d)
            N = len(win)
            logits, caches, Xs = self._forward_attention(X, C)
            z = logits - logits.max(1, keepdims=True)
            p = np.exp(z); p = p / p.sum(1, keepdims=True)
            g = p
            g[np.arange(N), tgt] -= 1
            g /= N
            if ep % 10 == 0:
                # Numerically stable log-softmax loss (avoid softmax overflow).
                z2 = logits - logits.max(1, keepdims=True)
                logp = z2 - np.log(np.exp(z2).sum(1, keepdims=True))
                ce = -logp[np.arange(N), tgt].mean()
                self.train_losses.append([ep, float(ce)])
            # ---- Backward: propagate layer by layer ----
            dX = g @ C
            dC = g.T @ Xs[-1]
            grads = {'C': dC}
            for l in reversed(range(len(self.attentions))):
                attn = self.attentions[l]
                Q, K, V, probs, ctx, Xl = caches[l]
                dctx = dX @ attn.WO
                dWO = dX.T @ ctx
                dprobs = dctx @ V.T
                dscores = probs * (dprobs - (dprobs * probs).sum(1, keepdims=True))
                dQ = dscores @ K / np.sqrt(attn.d)
                dK = dscores.T @ Q / np.sqrt(attn.d)
                dV = probs.T @ dctx
                dWQ = dQ.T @ Xl
                dWK = dK.T @ Xl
                dWV = dV.T @ Xl
                dX = dQ @ attn.WQ + dK @ attn.WK + dV @ attn.WV
                for name, gr in (('WQ', dWQ), ('WK', dWK), ('WV', dWV),
                                 ('WO', dWO)):
                    grads[f'{name}_{l}'] = gr * attn.masks[name]
            dC_in = np.zeros_like(C)
            np.add.at(dC_in, win, dX)
            grads['C'] = grads['C'] + dC_in
            t += 1
            lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / epochs))
            for k in params:
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                mh = m[k] / (1 - 0.9 ** t)
                vh = v[k] / (1 - 0.999 ** t)
                params[k] -= lr_t * mh / (np.sqrt(vh) + 1e-8)
        self.class_vectors = C
        # After training: forward all windows and assemble attn_next_class ([p] = predicted cls_seq[p+1]).
        preds = np.zeros(len(cls_seq) - 1, dtype=int)
        for i in range(0, len(cls_seq) - window, window):
            X = C[cls_seq[i:i + window]]
            logits, _, _ = self._forward_attention(X, C)
            preds[i:i + window] = logits.argmax(1)
        self.attn_next_class = preds
        return C

    # ------------------------------------------------------------------
    # Forward: predict the next word (relation gate -> short-range class rules -> long-range attention -> fact lookup)
    # ------------------------------------------------------------------
    def predict_next(self, ids, t):
        cls_seq = self.cls_seq
        m = self.m
        if t < 1:
            return None
        rw = self.relation.next_word(ids[t - 1])
        if rw is not None:
            return rw
        ctx = tuple(cls_seq[t - m:t])
        dist = self.ffn.class_table.get(ctx)
        if dist:
            pred_cls = max(dist, key=dist.get)
        elif self.attn_next_class is not None and (t - 1) < len(self.attn_next_class):
            pred_cls = self.attn_next_class[t - 1]
        else:
            return None
        key = (ids[t - 1], pred_cls)
        f = self.ffn.fact_table.get(key)
        if f:
            return max(f, key=f.get)
        return self.ffn.class_default.get(pred_cls)

    # ------------------------------------------------------------------
    def evaluate(self, start, end):
        hits = total = miss = 0
        for t in range(start, end):
            pred = self.predict_next(self.ids, t)
            if pred is None:
                miss += 1
                continue
            total += 1
            if pred == self.ids[t]:
                hits += 1
        return hits / total if total else 0.0, miss

    def vocab_size(self):
        return len(self.embedding.i2w)

    def n_params(self):
        n_int = 0
        n_int += sum(len(v) for v in self.ffn.class_table.values())
        n_int += sum(len(v) for v in self.ffn.fact_table.values())
        n_int += len(self.ffn.class_default)
        n_float = 0
        for attn in self.attentions:
            for mk in attn.masks.values():
                n_float += int(mk.sum())
        return {'int': n_int, 'float': n_float}
