"""Auto-imported at interpreter startup (site-packages/sitecustomize.py).

The DocLayout-YOLO fork calls torch.load() without weights_only during its
post-training strip_optimizer/final_eval step. On PyTorch >=2.6 the new
weights_only=True default rejects the fork's own model class and crashes the
run *after* training finished. We trust these checkpoints, so force
weights_only=False for this (isolated) venv.
"""
import functools

import torch

if not getattr(torch.load, "_wo_patched", False):
    _orig = torch.load

    @functools.wraps(_orig)
    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._wo_patched = True
    torch.load = _load
