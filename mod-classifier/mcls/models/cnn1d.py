"""1D residual CNN on raw I/Q (PLAN.md 5.2). Imports torch.

    input  2 x 4096 (I, Q), unit RMS
    stem   Conv1d(2 -> 32, k=7, stride=2) -> BatchNorm -> GELU
    stage1 2 x ResBlock(32,  k=5)  -> downsample x2
    stage2 2 x ResBlock(64,  k=5)  -> downsample x2
    stage3 2 x ResBlock(128, k=5)  -> downsample x2
    stage4 2 x ResBlock(256, k=3)  -> downsample x2
    pool   global average + global max, concatenated (512)
    head   Dropout(0.2) -> Linear(512 -> K)

Choices the plan leaves open, made here: post-activation residual blocks
(conv-BN-GELU-conv-BN, add, GELU); the first block of a stage that widens
carries a 1x1 conv + BN on its skip; downsampling is MaxPool1d(2), which has no
parameters. `width` scales every stage (32 is the plan).
"""

from __future__ import annotations

import torch
from torch import nn

from mcls.models import stage_plan_1d
from mcls.models.heads import Head, avg_max_pool


class ResBlock1d(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(cin, cout, kernel, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.GELU()
        self.skip = (
            nn.Identity()
            if cin == cout
            else nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class Backbone1d(nn.Module):
    """Stem and four stages; `features(x)` gives the pooled (B, 16 * width) vector."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, width, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(width), nn.GELU()
        )
        stages = []
        cin = width
        for cout, kernel in stage_plan_1d(width):
            stages.append(nn.Sequential(ResBlock1d(cin, cout, kernel), ResBlock1d(cout, cout, kernel), nn.MaxPool1d(2)))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.feature_dim = 2 * cin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_max_pool(self.stages(self.stem(x)))


class CNN1D(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.2, num_families: int = 0):
        super().__init__()
        self.backbone = Backbone1d(width)
        self.head = Head(self.backbone.feature_dim, num_classes, dropout, num_families)

    def forward(self, x: torch.Tensor):
        return self.head(self.backbone(x))
