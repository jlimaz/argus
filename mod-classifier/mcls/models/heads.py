"""Classification heads: the main softmax head and the optional family head (PLAN.md 5.6). Imports torch."""

from __future__ import annotations

import torch
from torch import nn


class Head(nn.Module):
    """Dropout -> Linear(features -> K); plus Linear(features -> F) when num_families > 0.

    Returns logits (B, K), or (logits, family_logits) with the auxiliary head.
    Softmax is applied by the loss and by the Classifier, not here.
    """

    def __init__(self, features: int, num_classes: int, dropout: float = 0.2, num_families: int = 0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classes = nn.Linear(features, num_classes)
        self.families = nn.Linear(features, num_families) if num_families else None

    def forward(self, features: torch.Tensor):
        features = self.dropout(features)
        logits = self.classes(features)
        if self.families is None:
            return logits
        return logits, self.families(features)


def avg_max_pool(x: torch.Tensor) -> torch.Tensor:
    """Global average and global max over every axis after channels, concatenated: (B, C, ...) -> (B, 2C)."""
    flat = x.flatten(2)
    return torch.cat([flat.mean(dim=2), flat.amax(dim=2)], dim=1)


def main_logits(output) -> torch.Tensor:
    """The class logits of a model output, with or without the auxiliary head."""
    return output[0] if isinstance(output, tuple) else output
