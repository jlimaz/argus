"""Augmentation of real training windows (PLAN 5.4), numpy only.

Every function is pure: f(window, rng, ...) -> new window. Inputs are never
modified and all randomness comes from `rng`, so the same generator state gives
the same output. Windows are complex64 [WINDOW_SAMPLES] at SAMPLE_RATE_HZ.

Augmentation applies to the train split only; `Augmenter` enforces that when it
is given the row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, SAMPLE_RATE_HZ, ManifestRow


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(x) ** 2)))


def _as_window(x: np.ndarray, like: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=SAMPLE_DTYPE).reshape(like.shape)


def _keep_rms(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale x to the RMS of `reference` (the front end hands on unit-RMS windows)."""
    current = _rms(x)
    return x if current == 0 else x * (_rms(reference) / current)


def add_noise_to_snr(
    window: np.ndarray,
    rng: np.random.Generator,
    snr_db_range: tuple[float, float] = (0.0, 30.0),
    current_snr_db: float | None = None,
) -> np.ndarray:
    """Add complex white Gaussian noise so the window's SNR drops to a random target. Never raises SNR.

    The target is drawn uniformly from `snr_db_range`, capped at the window's
    current SNR when that is known (`row.snr_db_est`). If the target is not
    below the current SNR, the window is returned unchanged. With the current
    SNR unknown, the window is treated as noise-free when sizing the noise, so
    the result's SNR is at most the target. Adding independent noise can only
    lower the SNR; the output keeps the input's RMS.
    """
    low, high = snr_db_range
    if current_snr_db is not None:
        high = min(high, current_snr_db)
        if high <= low:
            return window.copy()
    target = rng.uniform(low, high)
    power = float(np.mean(np.abs(window) ** 2))
    if power == 0:
        return window.copy()
    if current_snr_db is None:
        signal, noise_now = power, 0.0
    else:
        noise_now = power / (1.0 + 10 ** (current_snr_db / 10))
        signal = power - noise_now
    noise_add = signal / 10 ** (target / 10) - noise_now
    if noise_add <= 0:
        return window.copy()
    noise = rng.standard_normal(window.shape) + 1j * rng.standard_normal(window.shape)
    noisy = window + noise * np.sqrt(noise_add / 2)
    return _as_window(_keep_rms(noisy, window), window)


def frequency_offset(
    window: np.ndarray,
    rng: np.random.Generator,
    bandwidth_hz: float | None = None,
    fraction: float = 0.05,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Shift by a random frequency up to `fraction` of the occupied bandwidth (default 5%).

    With the bandwidth unknown (`row.bandwidth_hz` is None) the whole sample
    rate is taken as occupied. Magnitudes are unchanged.
    """
    occupied = bandwidth_hz if bandwidth_hz else sample_rate_hz
    offset = rng.uniform(-fraction, fraction) * occupied
    return _as_window(window * shift_tone(offset, window.shape[-1], sample_rate_hz), window)


def shift_tone(offset_hz: float, n: int, sample_rate_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """exp(j 2 pi f t) for n samples."""
    return np.exp(2j * np.pi * offset_hz * np.arange(n) / sample_rate_hz)


def time_shift(window: np.ndarray, rng: np.random.Generator, max_shift: int | None = None) -> np.ndarray:
    """Circularly shift by a random number of samples (up to `max_shift`, default the whole window)."""
    n = window.shape[-1]
    limit = n - 1 if max_shift is None else min(max_shift, n - 1)
    shift = int(rng.integers(-limit, limit + 1)) if limit > 0 else 0
    return np.roll(window, shift).astype(SAMPLE_DTYPE, copy=False)


def time_crop(window: np.ndarray, rng: np.random.Generator, min_fraction: float = 0.75) -> np.ndarray:
    """Keep a random contiguous stretch (at least `min_fraction` long) at a random place; zeros elsewhere.

    The kept samples are the original samples scaled by one constant so the
    output keeps the input's RMS.
    """
    n = window.shape[-1]
    length = int(rng.integers(max(1, int(np.ceil(min_fraction * n))), n + 1))
    source = int(rng.integers(0, n - length + 1))
    target = int(rng.integers(0, n - length + 1))
    out = np.zeros_like(window)
    out[target:target + length] = window[source:source + length]
    return _as_window(_keep_rms(out, window), window)


def random_phase(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rotate by one random global phase in [0, 2 pi). Magnitudes are unchanged."""
    return _as_window(window * np.exp(1j * rng.uniform(0, 2 * np.pi)), window)


def random_gain(window: np.ndarray, rng: np.random.Generator, gain_db: float = 3.0) -> np.ndarray:
    """Scale by a random gain within plus or minus `gain_db` dB. Phases are unchanged."""
    return _as_window(window * 10 ** (rng.uniform(-gain_db, gain_db) / 20), window)


def resample_jitter(window: np.ndarray, rng: np.random.Generator, jitter: float = 0.01) -> np.ndarray:
    """Resample by a random factor 1 + u, |u| <= jitter (radio clock error), keeping the length.

    The window is stretched about its centre by linear interpolation of I and
    Q; edges hold their last value. A tone at f comes out at about f * (1 + u).
    """
    n = window.shape[-1]
    factor = 1.0 + rng.uniform(-jitter, jitter)
    centre = (n - 1) / 2
    positions = centre + (np.arange(n) - centre) * factor
    grid = np.arange(n)
    real = np.interp(positions, grid, window.real)
    imag = np.interp(positions, grid, window.imag)
    return _as_window(real + 1j * imag, window)


def multipath(
    window: np.ndarray,
    rng: np.random.Generator,
    taps: tuple[int, int] = (2, 3),
    max_delay: int = 8,
    max_relative_db: float = -6.0,
) -> np.ndarray:
    """Mild 2 to 3 tap multipath: the direct path plus 1 or 2 weaker, delayed, randomly phased echoes.

    Echo delays are distinct, 1..`max_delay` samples (400 ns at 20 MS/s for 8);
    echo power is at most `max_relative_db` below the direct path. The output
    keeps the input's length and RMS.
    """
    count = int(rng.integers(taps[0], taps[1] + 1))
    delays = np.sort(rng.choice(np.arange(1, max_delay + 1), size=count - 1, replace=False))
    response = np.zeros(max_delay + 1, dtype=np.complex128)
    response[0] = 1.0
    for delay in delays:
        amplitude = 10 ** (rng.uniform(max_relative_db - 14, max_relative_db) / 20)
        response[delay] = amplitude * np.exp(1j * rng.uniform(0, 2 * np.pi))
    out = np.convolve(window, response)[: window.shape[-1]]
    return _as_window(_keep_rms(out, window), window)


def interference_mix(
    window: np.ndarray,
    other: np.ndarray,
    rng: np.random.Generator,
    sir_db_range: tuple[float, float] = (0.0, 20.0),
) -> tuple[np.ndarray, float]:
    """Add `other` (a window of a different class) at a random signal-to-interference ratio.

    Version 2 and off by default (PLAN 5.4). Returns (mixed window, SIR in dB);
    the caller keeps the label only when SIR is above its threshold, and
    otherwise drops the window.
    """
    sir = rng.uniform(*sir_db_range)
    p_signal = float(np.mean(np.abs(window) ** 2))
    p_other = float(np.mean(np.abs(other) ** 2))
    if p_other == 0 or p_signal == 0:
        return window.copy(), float("inf")
    scale = np.sqrt(p_signal / (p_other * 10 ** (sir / 10)))
    return _as_window(window + other * scale, window), float(sir)


@dataclass
class Augmenter:
    """The PLAN 5.4 chain, configured from a run config's [augment] section.

    Order: resampling jitter, multipath (on `multipath_fraction` of windows),
    frequency offset, time shift or crop, noise to a lower SNR, global phase,
    gain. Interference mixing is never applied here (it needs a second window).
    """

    enabled: bool = True
    snr_db_range: tuple[float, float] = (0.0, 30.0)
    noise_probability: float = 0.5
    freq_offset_fraction: float = 0.05
    time_shift: bool = True
    crop_probability: float = 0.25
    random_phase: bool = True
    gain_db: float = 3.0
    resample_jitter: float = 0.01
    multipath_fraction: float = 0.2

    @classmethod
    def from_config(cls, section: dict[str, Any] | None, **overrides: Any) -> "Augmenter":
        """Build from the keys `mcls.config` allows in [augment]; extra knobs via `overrides`."""
        section = dict(section or {})
        kwargs: dict[str, Any] = {}
        if "enabled" in section:
            kwargs["enabled"] = bool(section["enabled"])
        if "snr_db_range" in section:
            low, high = section["snr_db_range"]
            kwargs["snr_db_range"] = (float(low), float(high))
        if "freq_offset_fraction" in section:
            kwargs["freq_offset_fraction"] = float(section["freq_offset_fraction"])
        if "time_shift" in section:
            kwargs["time_shift"] = bool(section["time_shift"])
        if "random_phase" in section:
            kwargs["random_phase"] = bool(section["random_phase"])
        if "resample_jitter" in section:
            kwargs["resample_jitter"] = float(section["resample_jitter"])
        kwargs.update(overrides)
        return cls(**kwargs)

    def __call__(self, window: np.ndarray, rng: np.random.Generator, row: ManifestRow | None = None) -> np.ndarray:
        """Augment one window; returns a copy unchanged when disabled or the row is not in train."""
        if not self.enabled or (row is not None and row.split != "train"):
            return window.copy()
        out = window
        if self.resample_jitter > 0:
            out = resample_jitter(out, rng, self.resample_jitter)
        if self.multipath_fraction > 0 and rng.random() < self.multipath_fraction:
            out = multipath(out, rng)
        if self.freq_offset_fraction > 0:
            bandwidth = row.bandwidth_hz if row is not None else None
            out = frequency_offset(out, rng, bandwidth, self.freq_offset_fraction)
        if self.time_shift:
            out = time_crop(out, rng) if rng.random() < self.crop_probability else time_shift(out, rng)
        if self.noise_probability > 0 and rng.random() < self.noise_probability:
            current = row.snr_db_est if row is not None else None
            out = add_noise_to_snr(out, rng, self.snr_db_range, current)
        if self.random_phase:
            out = random_phase(out, rng)
        if self.gain_db > 0:
            out = random_gain(out, rng, self.gain_db)
        return np.asarray(out, dtype=SAMPLE_DTYPE).copy() if out is window else out
