""".xcp model format: a zip archive = meta.json + discrete.pkl + sparse/*.npy.

  model.xcp
    ├── meta.json        # architecture config
    ├── discrete.pkl     # discrete tables (integer lookups, pickle)
    ├── class_vectors.npy (optional)
    ├── attn_next_class.npy (optional)
    └── sparse/
        ├── attention_0.WQ.npy      # values
        ├── attention_0.WQ.mask.npy # mask
        └── ...
"""
import os
import json
import pickle
import zipfile
import numpy as np

from .nn.model import XCPModel


def _to_numpy(v):
    if hasattr(v, 'detach'):          # torch tensor
        return v.detach().cpu().numpy()
    return np.asarray(v)


def save(model, path):
    """Save the model as a .xcp file."""
    meta = {
        'n_classes': model.n_classes,
        'm': model.m,
        'd': model.d,
        'use_attention': bool(model.attentions),
        'attention_depth': len(model.attentions),
        'attn_keep': model.attentions[0].keep_frac if model.attentions else 0.15,
    }
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('meta.json', json.dumps(meta))
        # discrete tables
        discrete = {}
        for name, m in model.named_modules():
            if getattr(m, '_discrete', None):
                discrete[name] = m._discrete
        z.writestr('discrete.pkl', pickle.dumps(discrete))
        # sparse parameters
        for name, m in model.named_modules():
            for k, (v, mask) in getattr(m, '_sparse_params', {}).items():
                z.writestr(f'sparse/{name}.{k}.npy',
                           _to_numpy(v).tobytes())
                z.writestr(f'sparse/{name}.{k}.mask.npy',
                           _to_numpy(mask).tobytes())
        # attention class vectors and long-range fallback
        if model.class_vectors is not None:
            z.writestr('class_vectors.npy', _to_numpy(model.class_vectors).tobytes())
        if model.attn_next_class is not None:
            z.writestr('attn_next_class.npy',
                       np.asarray(model.attn_next_class, dtype=np.int64).tobytes())


def load(path):
    """Load a model from a .xcp file."""
    import torch
    with zipfile.ZipFile(path, 'r') as z:
        meta = json.loads(z.read('meta.json').decode('utf-8'))
        discrete = pickle.loads(z.read('discrete.pkl'))

        model = XCPModel(
            n_classes=meta['n_classes'], m=meta['m'], d=meta['d'],
            use_attention=meta['use_attention'],
            attention_depth=meta['attention_depth'],
            attn_keep=meta.get('attn_keep', 0.15))

        # restore discrete tables (name has a trailing dot, e.g. 'embedding.')
        for name, m in model.named_modules():
            if name in discrete:
                for attr, val in discrete[name].items():
                    setattr(m, attr, val)
                if name == 'embedding.' and getattr(m, 'i2w', None):
                    m.w2i = {w: i for i, w in enumerate(m.i2w)}
                    model.word_cls = m.word_cls

        # restore sparse parameters
        for name, m in model.named_modules():
            for k in list(getattr(m, '_sparse_params', {})):
                key = f'sparse/{name}.{k}.npy'
                mkey = f'sparse/{name}.{k}.mask.npy'
                if key in z.namelist():
                    v = np.frombuffer(z.read(key), dtype=np.float32)
                    mk = np.frombuffer(z.read(mkey), dtype=np.bool_)
                    d = meta['d']
                    v = v.reshape(d, d)
                    mk = mk.reshape(d, d)
                    tv = torch.tensor(v)
                    tm = torch.tensor(mk)
                    setattr(m, k, tv * tm)
                    m.mask = tm
                    m._sparse_params[k] = (tv, tm)

        # restore class vectors / long-range fallback
        if 'class_vectors.npy' in z.namelist():
            model.class_vectors = torch.tensor(
                np.frombuffer(z.read('class_vectors.npy'), dtype=np.float32)
                .reshape(meta['n_classes'], meta['d']))
        if 'attn_next_class.npy' in z.namelist():
            model.attn_next_class = np.frombuffer(
                z.read('attn_next_class.npy'), dtype=np.int64)
    return model
