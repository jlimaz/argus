"""AdaBN (PLAN.md 5.8): re-estimate BatchNorm statistics on unlabeled windows. Imports torch.

Every BatchNorm layer's running mean and variance are reset and recomputed as
a plain average over the given batches (momentum None), with all weights
frozen and every other layer in eval mode. Labels are never used. Report the
cross-dataset score with and without it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def adapt_batchnorm(model: nn.Module, batches: Iterable, device="cpu") -> nn.Module:
    """Batches: float32 (B, 2, 4096) arrays or tensors. Returns the model in eval mode."""
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        raise ValueError("model has no BatchNorm layers")
    model.to(device).eval()
    saved = []
    for norm in norms:
        saved.append(norm.momentum)
        norm.reset_running_stats()
        norm.momentum = None  # cumulative average over all batches seen
        norm.train()
    seen = 0
    try:
        for batch in batches:
            x = torch.as_tensor(np.asarray(batch, dtype=np.float32)) if not torch.is_tensor(batch) else batch
            if x.shape[0] < 2:
                continue  # BatchNorm cannot estimate a variance from one window
            model(x.to(device, dtype=torch.float32))
            seen += x.shape[0]
    finally:
        for norm, momentum in zip(norms, saved):
            norm.momentum = momentum
        model.eval()
    if seen == 0:
        raise ValueError("adapt_batchnorm saw no batch of 2 or more windows")
    return model
