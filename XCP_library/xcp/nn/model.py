"""XCPModel: composes the layers for end-to-end next-token prediction.

Data flow: token -> DiscreteEmbedding(token->ID->class) -> [L x SparseAttention]
       -> DiscreteFFN(class rules + fact lookup + class fallback) -> RelationGate -> token

Two stages: build() discrete counting (one pass); fit_attention() sparse gradient (GPU optional).
"""
import numpy as np
from .module import Module
from .embedding import DiscreteEmbedding
from .ffn import DiscreteFFN
from .attention import SparseAttention
from .relation import RelationGate


class XCPModel(Module):
    def __init__(self, n_classes=8, m=2, use_attention=False, d=64,
                 attention_depth=2, attn_keep=0.15):
        super().__init__()
        self.n_classes = n_classes
        self.m = m
        self.d = d
        self.embedding = DiscreteEmbedding(n_classes)
        self.ffn = DiscreteFFN(m)
        self.relation = RelationGate()
        self.attentions = []
        if use_attention:
            for l in range(attention_depth):
                attn = SparseAttention(d, attn_keep, seed=l)
                setattr(self, f'attention_{l}', attn)
                self.attentions.append(attn)
        self.class_vectors = None
        self.attn_next_class = None
        self.ids = None
        self.word_cls = None
        self.cls_seq = None

    # ------------------------------------------------------------------
    # stage 1: discrete counting (one pass, no gradients)
    # ------------------------------------------------------------------
    def build(self, tokens):
        self.ids = np.asarray(self.embedding.build(tokens), dtype=np.int64)
        self.word_cls = self.embedding.word_cls
        self.cls_seq = self.word_cls[self.ids]
        self.ffn.build(self.ids, self.word_cls)
        self.relation.build(self.ids, self.embedding.i2w)
        return self.ids

    # ------------------------------------------------------------------
    # stage 2: sparse attention gradient training (GPU optional)
    # ------------------------------------------------------------------
    def fit_attention(self, epochs=300, lr=0.05, window=64, seed=0):
        if not self.attentions:
            return self
        import torch
        d = self.attentions[0].d
        R = self.n_classes
        device = 'cuda' if (self._device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        for attn in self.attentions:
            attn.to(device)
        torch.manual_seed(seed)
        C = (torch.randn(R, d) / np.sqrt(d)).detach().requires_grad_(True)
        C = C.to(device)

        cls_seq = self.cls_seq
        wins, tgts = [], []
        for i in range(0, len(cls_seq) - window, window):
            wins.append(cls_seq[i:i + window])
            tgts.append(cls_seq[i + 1:i + window + 1])
        if not wins:
            return self

        # sparse parameters: (values, mask) pairs; C uses an all-ones mask (not sparse)
        # reference leaf parameters directly (attn.to(device) already moved WQ etc.)
        params = [(C, torch.ones_like(C, dtype=torch.bool))]
        for attn in self.attentions:
            for name in ('WQ', 'WK', 'WV', 'WO'):
                params.append((getattr(attn, name), attn.mask))

        from ..optim.sparse_adam import SparseAdam, CosineAnnealingLR
        opt = SparseAdam(params, lr=lr)
        sched = CosineAnnealingLR(opt, epochs)

        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            bi = int(rng.integers(0, len(wins)))
            win = torch.tensor(wins[bi], dtype=torch.long, device=device)
            tgt = torch.tensor(tgts[bi], dtype=torch.long, device=device)
            X = C[win]
            for attn in self.attentions:
                X = attn.forward(X)
            logits = X @ C.T
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()

        self.class_vectors = C
        self.attn_next_class = self._compute_attn_next_class(C, window, device)
        return self

    def _compute_attn_next_class(self, C, window, device):
        """After training: forward all windows and assemble the long-range fallback [p] = predicted cls_seq[p+1]."""
        import torch
        cls_seq = self.cls_seq
        preds = np.zeros(len(cls_seq) - 1, dtype=int)
        with torch.no_grad():
            for i in range(0, len(cls_seq) - window, window):
                win = torch.tensor(cls_seq[i:i + window], dtype=torch.long,
                                   device=device)
                X = C[win]
                for attn in self.attentions:
                    X = attn.forward(X)
                logits = X @ C.T
                preds[i:i + window] = logits.argmax(1).cpu().numpy()
        return preds

    # ------------------------------------------------------------------
    # forward: predict the next token (ids passed externally, so a loaded model can also infer)
    # ------------------------------------------------------------------
    def predict_next(self, ids, t):
        cls_seq = self.word_cls[ids]
        if t < 1:
            return None
        # 0) relation gate (highest priority)
        rw = self.relation.next_word(int(ids[t - 1]))
        if rw is not None:
            return rw
        # 1) short-range class rule (with backoff) + long-range attention fallback
        pred_cls, _ = self.ffn.predict_class(
            cls_seq, t, fallback=self.attn_next_class)
        if pred_cls is None:
            return None
        # 2) fact lookup + class fallback
        return self.ffn.predict_word(int(ids[t - 1]), pred_cls)

    def sample_next(self, ids, t, temperature=1.0, rng=None):
        """Sample the token at position t (with temperature; lower -> closer to argmax)."""
        cls_seq = self.word_cls[ids]
        if t < 1:
            return None
        rw = self.relation.next_word(int(ids[t - 1]))
        if rw is not None:
            return rw
        pred_cls, _ = self.ffn.predict_class(
            cls_seq, t, fallback=self.attn_next_class)
        if pred_cls is None:
            return None
        return self.ffn.predict_word_sample(int(ids[t - 1]), pred_cls,
                                            temperature, rng)

    def forward(self, ids):
        return [self.predict_next(ids, t) for t in range(len(ids))]

    def evaluate(self, ids, start=None, end=None):
        start = 1 if start is None else start
        end = len(ids) if end is None else end
        hits = total = miss = 0
        for t in range(start, end):
            pred = self.predict_next(ids, t)
            if pred is None:
                miss += 1
                continue
            total += 1
            if pred == ids[t]:
                hits += 1
        return hits / total if total else 0.0, miss

    def vocab_size(self):
        return len(self.embedding.i2w)

    def n_params(self):
        n_int = 0
        n_int += sum(len(v) for v in self.ffn.class_table.values())
        n_int += sum(len(v) for v in self.ffn.fact_table.values())
        n_int += len(self.ffn.class_default)
        n_float = sum(attn.nonzeros() for attn in self.attentions)
        return {'int': n_int, 'float': n_float}
