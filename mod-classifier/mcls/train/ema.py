"""Exponential moving average of weights (PLAN.md 5.3, decay 0.999). Imports torch.

After every optimizer step: ema = decay * ema + (1 - decay) * weight, for
every floating-point parameter and buffer; integer buffers (BatchNorm's
num_batches_tracked) are copied. Validation, the best checkpoint and the
deployed model all use the EMA weights.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("ema decay must be in [0, 1)")
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            target = ema_state[key]
            if target.dtype.is_floating_point:
                target.mul_(self.decay).add_(value.detach().to(target.dtype), alpha=1.0 - self.decay)
            else:
                target.copy_(value)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)
