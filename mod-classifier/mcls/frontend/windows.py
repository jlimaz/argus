"""Cut 4096-sample windows from 20 MS/s samples: inside bursts, or (background) between them.

Bursts come from the repo's `pipeline.features.detect_bursts`, the same code that
measures Argus's capture rows. A burst there is a group of pulses less than 20 ms
apart, so it can contain long quiet gaps; windows that are mostly below the
noise floor are therefore dropped (`active_share`).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, SAMPLE_RATE_HZ, WINDOW_SAMPLES

#: 50% overlap.
HOP = WINDOW_SAMPLES // 2
#: Default cap of windows per burst. detect_bursts groups pulses under 20 ms apart,
#: so on dense drone links one burst spans a whole 0.21 s detection block.
MAX_PER_BURST = 64
#: Activity test: 64-sample frames whose mean power is ACTIVE_FACTOR x the block's
#: noise floor count as on; a burst window needs at least MIN_ACTIVE of them on.
FRAME = 64
ACTIVE_FACTOR = 4.0  # 6 dB
MIN_ACTIVE = 0.25
#: The floor is this percentile of the 64-sample frame means in the block.
FLOOR_PERCENTILE = 10
#: Background windows keep this far (samples) from any burst edge.
IDLE_MARGIN = 1024


def import_features():
    """`pipeline.features`, via the loaders' helper when present, else by adding the repo root."""
    try:
        from mcls.sources._common import import_pipeline_features
    except ImportError:
        try:
            from pipeline import features
        except ImportError:
            root = str(Path(__file__).resolve().parents[3])
            if root not in sys.path:
                sys.path.insert(0, root)
            from pipeline import features
        return features
    return import_pipeline_features()


@dataclass
class Window:
    start: int  # sample index in the block it was cut from
    burst: dict | None  # the detect_bursts dict it lies in; None for background


def frame_power(iq: np.ndarray) -> np.ndarray:
    usable = (len(iq) // FRAME) * FRAME
    x = iq[:usable].reshape(-1, FRAME)
    return (x.real.astype(np.float64) ** 2 + x.imag.astype(np.float64) ** 2).mean(axis=1)


def noise_floor(iq: np.ndarray) -> float:
    p = frame_power(iq)
    return float(np.percentile(p, FLOOR_PERCENTILE)) if p.size else 0.0


def active_share(window: np.ndarray, floor: float) -> float:
    """Share of 64-sample frames in `window` whose power exceeds ACTIVE_FACTOR x floor."""
    return float((frame_power(window) > ACTIVE_FACTOR * floor).mean())


def detect(iq20: np.ndarray, center_frequency_hz: float) -> list[dict]:
    """detect_bursts at 20 MS/s, with sample spans added: "start", "end" (indices into iq20)."""
    bursts = import_features().detect_bursts(iq20, SAMPLE_RATE_HZ, center_frequency_hz)
    for b in bursts:
        b["start"] = int(round(b["start_s"] * SAMPLE_RATE_HZ))
        b["end"] = int(round(b["end_s"] * SAMPLE_RATE_HZ))
    return bursts


def burst_starts(start: int, end: int, total: int, max_per_burst: int) -> list[int]:
    """Window starts inside [start, end): hop HOP, at most max_per_burst.

    A burst shorter than a window gets one window centred on it, clamped to [0, total).
    """
    if total < WINDOW_SAMPLES:
        return []
    if end - start < WINDOW_SAMPLES:
        centre = (start + end) // 2
        return [int(min(max(centre - WINDOW_SAMPLES // 2, 0), total - WINDOW_SAMPLES))]
    return list(range(start, end - WINDOW_SAMPLES + 1, HOP))[:max_per_burst]


def burst_windows(iq20: np.ndarray, bursts: list[dict], max_per_burst: int = MAX_PER_BURST,
                  min_active: float = MIN_ACTIVE, floor: float | None = None) -> tuple[list[Window], int]:
    """(windows inside bursts that pass the activity test, number rejected as inactive).

    Every 50%-overlap position in a burst is tested; when more than
    `max_per_burst` pass, that many are kept, spread evenly over the burst.
    """
    floor = noise_floor(iq20) if floor is None else floor
    out, rejected = [], 0
    for b in bursts:
        starts = burst_starts(b["start"], b["end"], len(iq20), len(iq20))
        active = [s for s in starts if active_share(iq20[s : s + WINDOW_SAMPLES], floor) >= min_active]
        rejected += len(starts) - len(active)
        if len(active) > max_per_burst:
            pick = np.unique(np.linspace(0, len(active) - 1, max_per_burst).round().astype(int))
            active = [active[i] for i in pick]
        out.extend(Window(s, b) for s in active)
    return out, rejected


def idle_windows(iq20: np.ndarray, bursts: list[dict], max_per_stretch: int = MAX_PER_BURST,
                 margin: int = IDLE_MARGIN) -> list[Window]:
    """Windows from stretches no burst touches (kept `margin` samples clear of every burst).

    At most `max_per_stretch` per idle stretch, spread evenly over it.
    """
    busy = np.zeros(len(iq20) + 1, dtype=bool)
    for b in bursts:
        busy[max(0, b["start"] - margin) : b["end"] + margin] = True
    busy[-1] = True
    out = []
    edges = np.flatnonzero(np.diff(np.concatenate([[True], busy]).astype(np.int8)))
    # edges alternate: idle start (busy -> idle), idle end (idle -> busy)
    for a, e in zip(edges[0::2], edges[1::2]):
        starts = list(range(int(a), int(e) - WINDOW_SAMPLES + 1, HOP))
        if len(starts) > max_per_stretch:
            pick = np.unique(np.linspace(0, len(starts) - 1, max_per_stretch).round().astype(int))
            starts = [starts[i] for i in pick]
        out.extend(Window(s, None) for s in starts)
    return out


def normalize(windows: np.ndarray) -> np.ndarray:
    """Each row scaled to unit RMS (rows of zeros stay zero). complex64 (n, WINDOW_SAMPLES)."""
    w = np.asarray(windows, dtype=SAMPLE_DTYPE)
    if w.ndim == 1:
        w = w[None, :]
    rms = np.sqrt((w.real.astype(np.float64) ** 2 + w.imag.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    return (w / np.where(rms > 0, rms, 1.0)).astype(SAMPLE_DTYPE)


def cut(iq20: np.ndarray, windows: list[Window]) -> np.ndarray:
    """Stack and normalize the windows' samples: (n, WINDOW_SAMPLES) complex64."""
    if not windows:
        return np.zeros((0, WINDOW_SAMPLES), dtype=SAMPLE_DTYPE)
    return normalize(np.stack([iq20[w.start : w.start + WINDOW_SAMPLES] for w in windows]))


def windows_for_inference(iq20: np.ndarray, center_frequency_hz: float,
                          max_per_burst: int = MAX_PER_BURST) -> tuple[np.ndarray, list[dict], list[int]]:
    """For Engineer 10: (windows (n, 4096) unit RMS, bursts, burst index of each window)."""
    bursts = detect(np.asarray(iq20, dtype=SAMPLE_DTYPE), center_frequency_hz)
    wins, _ = burst_windows(iq20, bursts, max_per_burst)
    index = {id(b): i for i, b in enumerate(bursts)}
    return cut(iq20, wins), bursts, [index[id(w.burst)] for w in wins]
