"""SparseAdam: sparse Adam optimizer (updates only masked non-zero positions).

Mirrors torch.optim.Adam's calling convention, but updates only the non-zero
elements of sparse parameters (values + fixed mask), and re-applies the mask
after each step to preserve sparsity.
"""
import torch


class SparseAdam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)      # [(values, mask), ...] values are torch tensors
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.state = [{'step': 0, 'm': None, 'v': None} for _ in self.params]

    def zero_grad(self):
        for values, _ in self.params:
            if values.grad is not None:
                values.grad.zero_()

    def step(self):
        for i, (values, mask) in enumerate(self.params):
            if values.grad is None:
                continue
            g = values.grad * mask
            st = self.state[i]
            if st['m'] is None:
                st['m'] = torch.zeros_like(g)
                st['v'] = torch.zeros_like(g)
            st['step'] += 1
            t = st['step']
            st['m'] = self.betas[0] * st['m'] + (1 - self.betas[0]) * g
            st['v'] = self.betas[1] * st['v'] + (1 - self.betas[1]) * g * g
            m_hat = st['m'] / (1 - self.betas[0] ** t)
            v_hat = st['v'] / (1 - self.betas[1] ** t)
            values.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
            values.data *= mask       # keep sparsity (zero out non-mask positions)


class CosineAnnealingLR:
    """Cosine-annealing learning rate (mirrors torch.optim.lr_scheduler)."""

    def __init__(self, optimizer, T_max, lr_min=0.0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.lr_min = lr_min
        self.base_lr = optimizer.lr
        self.t = 0

    def step(self):
        self.t += 1
        import math
        self.optimizer.lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * \
            (1 + math.cos(math.pi * self.t / self.T_max))
