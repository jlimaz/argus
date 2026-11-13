"""Neural models (PLAN.md 5.2, 5.6).

Importing this package does not import torch: the laptop has none. `build`
imports the torch modules when it is called. The numpy helpers here (input
packing, the spectrogram reference, the parameter-count formula) are shared by
the torch code and tested without it.

Every model takes one input: float32 (B, 2, 4096), the I and Q rows of a
unit-RMS window. The spectrogram models compute their 1 x 64 x 64 view inside
the network, so data loading only ever supplies I/Q.
"""

from __future__ import annotations

import numpy as np

from mcls.contracts import WINDOW_SAMPLES, check_windows

NEURAL_MODELS = ("cnn1d", "cnn2d", "fused")

#: Modulation families for the optional auxiliary head (PLAN.md 5.6).
FAMILIES = ("ook", "fsk", "psk_qam", "ofdm", "noise_like")

#: Spectrogram view: non-overlapping frames of SPEC_NFFT samples, so
#: WINDOW_SAMPLES / SPEC_NFFT frames by SPEC_NFFT frequency bins.
SPEC_NFFT = 64
SPEC_FRAMES = WINDOW_SAMPLES // SPEC_NFFT
SPEC_EPS = 1e-10


def iq_array(windows: np.ndarray) -> np.ndarray:
    """Complex64 windows (n, 4096) -> float32 (n, 2, 4096): row 0 is I, row 1 is Q."""
    check_windows(windows)
    return np.stack([windows.real, windows.imag], axis=1).astype(np.float32, copy=False)


def spectrogram_np(iq: np.ndarray) -> np.ndarray:
    """Reference for the in-model spectrogram: float32 (n, 2, 4096) -> (n, 1, 64, 64).

    Frames of 64 samples without overlap, Hann window (periodic), two-sided DFT
    with zero frequency moved to the centre, log10 power, then each window is
    standardised to zero mean and unit variance. Axis 2 is time, axis 3 is
    frequency.
    """
    z = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    frames = z.reshape(len(z), SPEC_FRAMES, SPEC_NFFT) * hann(SPEC_NFFT)
    spectrum = np.fft.fftshift(np.fft.fft(frames, axis=-1), axes=-1)
    power = np.log10(np.abs(spectrum) ** 2 + SPEC_EPS)
    mean = power.mean(axis=(1, 2), keepdims=True)
    std = power.std(axis=(1, 2), keepdims=True)
    return ((power - mean) / (std + 1e-6)).astype(np.float32)[:, None]


def hann(n: int) -> np.ndarray:
    """Periodic Hann window (the STFT convention, as torch.hann_window's default)."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def cnn1d_param_count(width: int = 32, num_classes: int = 4, num_families: int = 0) -> int:
    """Trainable parameters of `CNN1D` (PLAN.md 5.2), counted by formula.

    Convolutions carry no bias (BatchNorm follows each one); BatchNorm has a
    weight and a bias per channel; the first block of a stage that changes
    width has a 1x1 projection plus BatchNorm on its skip path.
    """
    def conv(cin: int, cout: int, k: int) -> int:
        return cin * cout * k

    def bn(c: int) -> int:
        return 2 * c

    def block(cin: int, cout: int, k: int) -> int:
        total = conv(cin, cout, k) + bn(cout) + conv(cout, cout, k) + bn(cout)
        if cin != cout:
            total += conv(cin, cout, 1) + bn(cout)
        return total

    total = conv(2, width, 7) + bn(width)
    cin = width
    for cout, k in stage_plan_1d(width):
        total += block(cin, cout, k) + block(cout, cout, k)
        cin = cout
    features = 2 * cin
    total += features * num_classes + num_classes
    if num_families:
        total += features * num_families + num_families
    return total


def stage_plan_1d(width: int = 32) -> list[tuple[int, int]]:
    """(channels, kernel) of the four stages of CNN1D: PLAN.md 5.2 at width 32."""
    return [(width, 5), (2 * width, 5), (4 * width, 5), (8 * width, 3)]


def build(name: str, num_classes: int, *, width: int = 32, dropout: float = 0.2, num_families: int = 0):
    """Build a model by name. Imports torch."""
    if name not in NEURAL_MODELS:
        raise ValueError(f"unknown neural model {name!r}; expected one of {NEURAL_MODELS}")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    kwargs = {"num_classes": num_classes, "width": width, "dropout": dropout, "num_families": num_families}
    if name == "cnn1d":
        from mcls.models.cnn1d import CNN1D

        return CNN1D(**kwargs)
    if name == "cnn2d":
        from mcls.models.cnn2d import CNN2D

        return CNN2D(**kwargs)
    from mcls.models.fused import FusedCNN

    return FusedCNN(**kwargs)
