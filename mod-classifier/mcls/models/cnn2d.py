"""2D CNN on the 64 x 64 spectrogram view (PLAN.md 5.2 ablation). Imports torch.

The plan names this ablation without an architecture; the one here mirrors the
1D network at a similar size:

    input  2 x 4096 (I, Q) -> Spectrogram -> 1 x 64 x 64 (time x frequency)
    stem   Conv2d(1 -> w, 3) -> BatchNorm -> GELU
    stage1 2 x ResBlock2d(w)   -> MaxPool 2      64 -> 32
    stage2 2 x ResBlock2d(2w)  -> MaxPool 2      32 -> 16
    stage3 2 x ResBlock2d(4w)  -> MaxPool 2      16 -> 8
    stage4 2 x ResBlock2d(4w)  -> MaxPool 2       8 -> 4
    pool   global average + global max (8w)
    head   Dropout -> Linear(8w -> K)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mcls.models import SPEC_EPS, SPEC_FRAMES, SPEC_NFFT
from mcls.models.heads import Head, avg_max_pool


class Spectrogram(nn.Module):
    """(B, 2, 4096) I/Q -> (B, 1, 64, 64) standardised log10 power; matches mcls.models.spectrogram_np.

    The DFT is two real matrix products (cos and sin tables, rows already in
    fftshift order) rather than torch.fft, so it runs the same on CPU, CUDA and
    MPS. It always computes in float32, outside autocast.
    """

    def __init__(self, nfft: int = SPEC_NFFT, frames: int = SPEC_FRAMES):
        super().__init__()
        self.nfft, self.frames = nfft, frames
        n = torch.arange(nfft, dtype=torch.float64)
        k = (torch.arange(nfft, dtype=torch.float64) + nfft // 2) % nfft  # fftshift order
        angle = 2 * math.pi * k[:, None] * n[None, :] / nfft
        window = torch.hann_window(nfft, periodic=True, dtype=torch.float64)
        # Fold the window into the tables: X = (a*w) C^T + ... = a (C*w)^T + ...
        self.register_buffer("cos_w", (torch.cos(angle) * window).to(torch.float32), persistent=False)
        self.register_buffer("sin_w", (torch.sin(angle) * window).to(torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            with torch.autocast(device_type="cuda", enabled=False):
                return self._compute(x)
        return self._compute(x)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        batch = x.shape[0]
        a = x[:, 0].reshape(batch, self.frames, self.nfft)
        b = x[:, 1].reshape(batch, self.frames, self.nfft)
        cos_t, sin_t = self.cos_w.t(), self.sin_w.t()
        real = a @ cos_t + b @ sin_t
        imag = b @ cos_t - a @ sin_t
        power = torch.log10(real * real + imag * imag + SPEC_EPS)
        mean = power.mean(dim=(1, 2), keepdim=True)
        std = power.std(dim=(1, 2), keepdim=True, correction=0)
        return ((power - mean) / (std + 1e-6)).unsqueeze(1)


class ResBlock2d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.GELU()
        self.skip = (
            nn.Identity()
            if cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class Backbone2d(nn.Module):
    """Spectrogram, stem and four stages; forward gives the pooled (B, 8 * width) vector."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.spectrogram = Spectrogram()
        self.stem = nn.Sequential(nn.Conv2d(1, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.GELU())
        stages = []
        cin = width
        for cout in (width, 2 * width, 4 * width, 4 * width):
            stages.append(nn.Sequential(ResBlock2d(cin, cout), ResBlock2d(cout, cout), nn.MaxPool2d(2)))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.feature_dim = 2 * cin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return avg_max_pool(self.stages(self.stem(self.spectrogram(x))))


class CNN2D(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.2, num_families: int = 0):
        super().__init__()
        self.backbone = Backbone2d(width)
        self.head = Head(self.backbone.feature_dim, num_classes, dropout, num_families)

    def forward(self, x: torch.Tensor):
        return self.head(self.backbone(x))
