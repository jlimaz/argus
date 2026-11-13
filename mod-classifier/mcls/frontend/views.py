"""The two model inputs made from a window: raw I/Q and a log-magnitude STFT.

Both are cheap (one 4096-point pass per window), so training computes them on
the fly from the complex64 shard; nothing is cached on disk.
"""

from __future__ import annotations

import numpy as np

from mcls.contracts import WINDOW_SAMPLES, check_windows

STFT_BINS = 64
STFT_HOP = WINDOW_SAMPLES // STFT_BINS  # 64: 64 frames, no overlap
#: Added to power before the log so silent bins stay finite: -100 dB.
POWER_EPS = 1e-10


def iq_view(windows: np.ndarray) -> np.ndarray:
    """(n, 4096) complex64 -> (n, 2, 4096) float32: channel 0 is I, channel 1 is Q."""
    check_windows(windows)
    return np.stack([windows.real, windows.imag], axis=1).astype(np.float32)


def stft_view(windows: np.ndarray) -> np.ndarray:
    """(n, 4096) complex64 -> (n, 1, 64, 64) float32 log power, dB.

    64 Hann-windowed frames of 64 samples (hop 64), each a 64-point FFT,
    fftshifted so row 32 is DC. Axis 2 is frequency (low to high), axis 3 is time.
    Values are 10 log10(power + 1e-10), power normalized by the window energy.
    """
    check_windows(windows)
    n = windows.shape[0]
    frames = windows.reshape(n, STFT_BINS, STFT_BINS)  # (n, time, sample)
    hann = np.hanning(STFT_BINS + 2)[1:-1].astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(frames * hann, axis=2), axes=2)
    power = (np.abs(spectrum) ** 2) / np.sum(hann**2)
    db = 10 * np.log10(power + POWER_EPS)
    return np.transpose(db, (0, 2, 1))[:, None, :, :].astype(np.float32)
