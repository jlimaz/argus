"""Hand-made features of one window, for the baseline models (PLAN.md section 5.1).

Two groups, all numpy:

- Burst features, the ones `pipeline.features.detect_bursts` writes: bandwidth,
  SNR, duty cycle, burst duration, pulse interval, hopping. They come from a
  burst row handed in per window (the front end's burst table) or, when none is
  given, from running `detect_bursts` on the window itself.
- Spectral and envelope shape: spectral flatness, peak count, occupied-bandwidth
  ratios, spectral spread, envelope kurtosis, peak-to-average power ratio and
  envelope variation.

Every feature is a ratio or a level relative to the window itself (windows are
unit RMS), so no absolute power reaches a model. A value that cannot be measured
(no burst, a truncated burst, a single pulse) is set to 0 and flagged in its own
`*_missing` column, so the models can tell "zero" from "unknown".
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mcls.contracts import SAMPLE_RATE_HZ, WINDOW_SAMPLES

#: The repository root (argus/), home of `pipeline.features`.
REPO_ROOT = Path(__file__).resolve().parents[3]

FEATURE_NAMES: tuple[str, ...] = (
    # burst features
    "burst_missing",
    "n_bursts",
    "bandwidth_frac",
    "snr_db",
    "duty_cycle",
    "burst_duration_us",
    "burst_duration_missing",
    "pulse_interval_us",
    "pulse_interval_missing",
    "hopping",
    # spectral shape
    "spectral_flatness",
    "peak_count",
    "obw50_frac",
    "obw90_frac",
    "obw99_frac",
    "spectral_spread_frac",
    # envelope shape
    "envelope_kurtosis",
    "papr_db",
    "envelope_cv",
)

#: Welch frame for the spectral features: 256 bins of about 78 kHz at 20 MS/s.
PSD_FRAME = 256
#: A spectral peak is a run of smoothed bins at least this many times the median bin.
PEAK_FACTOR = 10.0
PEAK_SMOOTH_BINS = 5
SNR_CLIP_DB = (-50.0, 100.0)


def detect_bursts():
    """`pipeline.features.detect_bursts`, adding the repository root to sys.path if needed."""
    try:
        from pipeline.features import detect_bursts as found
    except ImportError:
        if str(REPO_ROOT) not in sys.path:
            sys.path.append(str(REPO_ROOT))
        from pipeline.features import detect_bursts as found
    return found


def extract(
    windows: np.ndarray,
    bursts: Sequence[Any] | None = None,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> np.ndarray:
    """Features of each window, shaped (n, len(FEATURE_NAMES)), float32.

    `bursts[i]` describes window i: None (measure it here with detect_bursts),
    one dict in detect_bursts' format, or a list of such dicts (empty = no burst).
    """
    windows = np.asarray(windows)
    if windows.ndim != 2:
        raise ValueError(f"windows must be 2-D (n, samples), got {windows.shape}")
    if bursts is not None and len(bursts) != len(windows):
        raise ValueError(f"{len(bursts)} burst entries for {len(windows)} windows")
    out = np.zeros((len(windows), len(FEATURE_NAMES)), dtype=np.float64)
    find = None
    for i, window in enumerate(windows):
        window = np.ascontiguousarray(window, dtype=np.complex64)
        entry = None if bursts is None else bursts[i]
        if entry is None:
            find = find or detect_bursts()
            entry = find(window, sample_rate_hz, 0.0)
        out[i, :10] = burst_features(entry, sample_rate_hz)
        out[i, 10:16] = spectral_features(window)
        out[i, 16:] = envelope_features(window)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def burst_features(entry: Any, sample_rate_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """The ten burst columns from a detect_bursts dict or list of dicts.

    With several bursts in one window, the longest one describes the window and
    `n_bursts` counts them.
    """
    rows = [entry] if isinstance(entry, dict) else list(entry or [])
    values = np.zeros(10)
    if not rows:
        values[0] = 1.0  # burst_missing
        values[6] = values[8] = 1.0
        return values
    main = max(rows, key=_span)
    snr = main.get("snr_db", main.get("snr_db_est"))
    duration = main.get("burst_duration_s")
    interval = main.get("pulse_interval_s")
    values[1] = float(main.get("n_bursts", len(rows)))
    values[2] = _number(main.get("bandwidth_hz")) / sample_rate_hz
    values[3] = float(np.clip(_number(snr), *SNR_CLIP_DB))
    values[4] = _number(main.get("duty_cycle"))
    values[5] = _number(duration) * 1e6
    values[6] = 0.0 if _finite(duration) else 1.0
    values[7] = _number(interval) * 1e6
    values[8] = 0.0 if _finite(interval) else 1.0
    values[9] = 1.0 if main.get("hopping") else 0.0
    return values


def spectral_features(window: np.ndarray) -> np.ndarray:
    """flatness, peak count, 50/90/99% occupied-bandwidth fractions, spectral spread."""
    psd = _welch(window)
    total = psd.sum()
    if total <= 0:
        return np.zeros(6)
    p = psd / total
    flatness = float(np.exp(np.mean(np.log(p + 1e-20))) / np.mean(p))
    smooth = np.convolve(p, np.ones(PEAK_SMOOTH_BINS) / PEAK_SMOOTH_BINS, mode="same")
    above = smooth > PEAK_FACTOR * np.median(smooth)
    peaks = int(np.count_nonzero(np.diff(np.concatenate([[False], above]).astype(np.int8)) == 1))
    held = np.cumsum(np.sort(p)[::-1])
    obw = [float((np.searchsorted(held, share) + 1) / len(p)) for share in (0.5, 0.9, 0.99)]
    position = np.arange(len(p)) / len(p) - 0.5  # bin centre as a fraction of the sample rate
    centroid = float(np.sum(p * position))
    spread = float(np.sqrt(np.sum(p * (position - centroid) ** 2)))
    return np.array([flatness, peaks, *obw, spread])


def envelope_features(window: np.ndarray) -> np.ndarray:
    """Kurtosis of the envelope |x|, peak-to-average power (dB), coefficient of variation of |x|."""
    envelope = np.abs(window.astype(np.complex128))
    mean = envelope.mean()
    centred = envelope - mean
    variance = np.mean(centred**2)
    # A constant envelope (a pure tone) has no spread; rounding error must not read as kurtosis.
    flat = variance <= 1e-12 * mean**2
    kurtosis = 0.0 if flat else float(np.mean(centred**4) / variance**2)
    power = envelope**2
    papr = float(10 * np.log10(power.max() / power.mean())) if power.mean() > 0 else 0.0
    cv = 0.0 if flat or mean <= 0 else float(np.sqrt(variance) / mean)
    return np.array([kurtosis, papr, cv])


def _welch(window: np.ndarray) -> np.ndarray:
    frames = len(window) // PSD_FRAME
    if frames == 0:
        frame = np.zeros(PSD_FRAME, dtype=np.complex128)
        frame[: len(window)] = window
        segments = frame[None, :]
    else:
        segments = window[: frames * PSD_FRAME].astype(np.complex128).reshape(frames, PSD_FRAME)
    taper = np.hanning(PSD_FRAME + 2)[1:-1]
    spectrum = np.fft.fft(segments * taper, axis=1)
    return np.fft.fftshift(np.mean(np.abs(spectrum) ** 2, axis=0))


def _span(row: dict) -> float:
    start, end = row.get("start_s"), row.get("end_s")
    if _finite(start) and _finite(end):
        return float(end) - float(start)
    return _number(row.get("burst_duration_s"))


def _finite(value: Any) -> bool:
    return value is not None and np.isfinite(float(value))


def _number(value: Any) -> float:
    return float(value) if _finite(value) else 0.0


assert WINDOW_SAMPLES % PSD_FRAME == 0
