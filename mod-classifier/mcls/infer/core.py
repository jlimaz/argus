"""Bursts in, one class and a calibrated confidence per burst out.

Order of operations, per burst (PLAN 5.5 and 5.7):

1. Take the burst's native samples plus context each side (one window, or a
   quarter of the burst's length if more, so the activity test's noise floor
   has noise to measure; with no noise around the burst it finds none) and
   hand them to Engineer 4's `mcls.frontend.process.windows_from_iq`, the same
   code path preprocess uses for training: 20 MHz slice centred on the burst
   (ignored at 20 MS/s or less), resampling to 20 MS/s, burst detection,
   the activity test, at most `max_windows` windows per burst, unit RMS.
2. Keep the windows of the detected burst(s) that overlap the requested burst
   (native sample times, resampler delay corrected by e4). None found: the
   burst is `unknown` with no windows.
3. Raw logits per window (`classifier.logits`, never divided by a
   temperature), divided once by calibration.json's temperature. A classifier
   without `logits` gives probabilities; their logs are used instead.
4. Average the windows' logits (as Engineer 8's burst-level evaluation does),
   softmax, multiply by the band prior and renormalise.
5. Top probability under the threshold -> `unknown`.

The temperature is applied exactly once, here. Engineer 7's `classify` also
applies calibration.json's temperature when its loader is given one; that is
why only `logits` is used when it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from mcls.contracts import WINDOW_SAMPLES, Classifier, check_probabilities, check_windows
from mcls.infer.calibration import Calibration
from mcls.infer.priors import BandPriors

UNKNOWN = "unknown"
#: Windows per burst at inference, for latency (PLAN 6.2: 50 ms per burst). Training keeps
#: up to 64 (e4's default); burst-level evaluation should use the same cap as here.
MAX_WINDOWS = 8
#: Native context added each side of a burst, as a share of its length (see burst_windows).
CONTEXT_SHARE = 0.25


@dataclass
class BurstClass:
    signal_class: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)
    windows: int = 0

    def row_fields(self, digits: int = 4) -> dict:
        """The three fields `pipeline.ingest --classify` adds to a row."""
        return {
            "signal_class": self.signal_class,
            "class_confidence": round(self.confidence, digits),
            "class_probs": {name: round(p, digits) for name, p in self.probs.items()},
        }


def window_logits(classifier: Classifier, windows: np.ndarray) -> np.ndarray:
    """(n, K) logits before any temperature."""
    check_windows(windows)
    logits = getattr(classifier, "logits", None)
    if callable(logits):
        out = np.asarray(logits(windows), dtype=np.float64)
        if out.shape != (len(windows), len(classifier.classes)) or not np.isfinite(out).all():
            raise ValueError(f"classifier.logits returned {out.shape}, expected {(len(windows), len(classifier.classes))} finite")
        return out
    probs = np.asarray(classifier.classify(windows), dtype=np.float64)
    check_probabilities(probs, len(windows), tuple(classifier.classes))
    return np.log(np.clip(probs, 1e-12, None))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def aggregate(logits: np.ndarray, classes: Sequence[str], *, temperature: float = 1.0,
              prior: np.ndarray | None = None, unknown_threshold: float | None = None) -> BurstClass:
    """One burst's class from its windows' logits (n, K): temperature, mean logits, prior, threshold."""
    classes = tuple(classes)
    if len(logits) == 0:
        uniform = 1.0 / len(classes)
        return BurstClass(UNKNOWN, uniform, {c: uniform for c in classes}, 0)
    score = (np.asarray(logits, dtype=np.float64) / temperature).mean(axis=0)
    if prior is not None:
        score = score + np.log(np.asarray(prior, dtype=np.float64))
    probs = np.exp(_log_softmax(score))
    top = int(np.argmax(probs))
    confidence = float(probs[top])
    name = classes[top] if unknown_threshold is None or confidence >= unknown_threshold else UNKNOWN
    return BurstClass(name, confidence, {c: float(p) for c, p in zip(classes, probs)}, len(logits))


def _span(burst: dict, sample_rate_hz: float, total: int) -> tuple[int, int]:
    """Native-rate sample range [start, end) of a requested burst; a burst with no end runs MAX_WINDOWS windows."""
    start = min(total, max(0, int(np.floor(burst.get("start_s", 0.0) * sample_rate_hz))))
    end_s = burst.get("end_s")
    if end_s is None and burst.get("burst_duration_s") is not None:
        end_s = burst.get("start_s", 0.0) + burst["burst_duration_s"]
    if end_s is None:
        end = start + int(np.ceil((WINDOW_SAMPLES + (MAX_WINDOWS - 1) * WINDOW_SAMPLES // 2) * sample_rate_hz / 20e6))
    else:
        end = int(np.ceil(end_s * sample_rate_hz))
    return start, min(total, max(end, start + 1))


def burst_windows(iq: np.ndarray, sample_rate_hz: float, center_frequency_hz: float, burst: dict,
                  max_windows: int = MAX_WINDOWS, min_active: float | None = None) -> np.ndarray:
    """(n, 4096) unit-RMS windows of one requested burst, cut by Engineer 4's `windows_from_iq`.

    `min_active` overrides the front end's activity test (FrontendParams.min_active).
    0.0 keeps every window inside the burst: the retry for an always-on signal (a
    jammer), whose windows the test rejects because the whole block looks like the
    noise floor. Training cut such signals evenly for the same reason (CONTINUOUS
    in mcls.frontend.process), but at inference the class is not known in advance.
    """
    from mcls.frontend.process import FrontendParams, windows_from_iq  # Engineer 4

    total = len(iq)
    lo, hi = _span(burst, sample_rate_hz, total)
    # Context each side: one window, and at least CONTEXT_SHARE of the burst's length, so the front
    # end's activity test (floor = 10th percentile of the block) sees some noise, as in a training block.
    pad = max(int(np.ceil(WINDOW_SAMPLES * sample_rate_hz / 20e6)) + 256, int(CONTEXT_SHARE * (hi - lo)))
    ctx_lo, ctx_hi = max(0, lo - pad), min(total, hi + pad)
    burst_hz = float(burst.get("center_frequency_hz", center_frequency_hz))
    result = windows_from_iq(iq[ctx_lo:ctx_hi], sample_rate_hz, center_frequency_hz,
                             offset_hz=burst_hz - center_frequency_hz,
                             params=(FrontendParams(max_windows_per_burst=max_windows) if min_active is None
                                     else FrontendParams(max_windows_per_burst=max_windows, min_active=min_active)))
    a, b = lo - ctx_lo, hi - ctx_lo
    hits = {d["burst_index"] for d in result.bursts if min(d["native_end"], b) > max(d["native_start"], a)}
    keep = [i for i, k in enumerate(result.window_burst) if k in hits]
    if len(keep) > max_windows:
        keep = [keep[i] for i in np.unique(np.linspace(0, len(keep) - 1, max_windows).round().astype(int))]
    return result.windows[keep] if keep else np.zeros((0, WINDOW_SAMPLES), dtype=np.complex64)


def classify_bursts(iq: np.ndarray, sample_rate_hz: float, center_frequency_hz: float, bursts: Sequence[dict],
                    classifier: Classifier, *, calibration: Calibration | None = None,
                    priors: BandPriors | None = None, max_windows: int = MAX_WINDOWS) -> list[BurstClass]:
    """One BurstClass per burst, in order.

    `bursts` are `detect_bursts(iq, sample_rate_hz, center_frequency_hz)` dicts:
    `start_s`, `end_s` (or `burst_duration_s`; neither means "runs on") and
    `center_frequency_hz` (the burst's RF centre, used for the band prior and to
    centre the 20 MHz slice).
    """
    calibration = calibration or Calibration()
    classes = tuple(classifier.classes)
    iq = np.asarray(iq, dtype=np.complex64)
    results = []
    for burst in bursts:
        windows = burst_windows(iq, sample_rate_hz, center_frequency_hz, burst, max_windows)
        logits = window_logits(classifier, windows) if len(windows) else np.zeros((0, len(classes)))
        burst_hz = float(burst.get("center_frequency_hz", center_frequency_hz))
        prior = priors.for_frequency(burst_hz, classes) if priors is not None else None
        results.append(aggregate(logits, classes, temperature=calibration.temperature, prior=prior,
                                 unknown_threshold=calibration.unknown_threshold))
    return results
