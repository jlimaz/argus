"""Both views fused (PLAN.md 5.2 ablation). Imports torch.

The 1D I/Q backbone and the 2D spectrogram backbone run side by side on the
same window; their pooled features are concatenated (16w + 8w) and one head
classifies them.
"""

from __future__ import annotations

import torch
from torch import nn

from mcls.models.cnn1d import Backbone1d
from mcls.models.cnn2d import Backbone2d
from mcls.models.heads import Head


class FusedCNN(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.2, num_families: int = 0):
        super().__init__()
        self.iq = Backbone1d(width)
        self.spec = Backbone2d(width)
        self.head = Head(self.iq.feature_dim + self.spec.feature_dim, num_classes, dropout, num_families)

    def forward(self, x: torch.Tensor):
        return self.head(torch.cat([self.iq(x), self.spec(x)], dim=1))
