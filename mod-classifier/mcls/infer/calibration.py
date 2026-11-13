"""Read the temperature and unknown threshold Engineer 8's calibrator writes.

`python -m mcls.evaluate.calibrate --run runs/<name>` writes
`runs/<name>/calibration.json`; the two fields used here are

    "temperature": float
    "unknown_threshold": {"score": "max_softmax", "value": float, "applied_after_temperature": true, ...}

The threshold was chosen on validation *windows* (95% kept, PLAN 5.5). Here it
is applied to the *burst* probability after averaging and the band prior, so
the share of bursts kept need not be 95%. That has not been measured; the
evaluator's burst-level coverage curve is where to check it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    #: Below this top probability a burst is `unknown`. None: never unknown (only for tests and ablations).
    unknown_threshold: float | None = None
    source: str = ""

    def __post_init__(self):
        if not (math.isfinite(self.temperature) and self.temperature > 0):
            raise ValueError(f"temperature must be a positive number, got {self.temperature}")
        if self.unknown_threshold is not None and not 0.0 <= self.unknown_threshold <= 1.0:
            raise ValueError(f"unknown_threshold must be in 0..1, got {self.unknown_threshold}")


def load_calibration(path: str | Path, classes: tuple[str, ...] | list[str] | None = None, *,
                     manifest_sha256: str | None = None, run: str | None = None) -> Calibration:
    """Calibration from a calibration.json (or a run folder holding one).

    With `classes` (the loaded model's class list, in order), the file is checked
    with Engineer 8's `check_calibration`: its classes must match exactly, and its
    manifest_sha256 and run folder name too when given. A calibration.json for
    another model is refused with ValueError.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "calibration.json"
    data = json.loads(path.read_text())
    if classes is not None:
        from mcls.evaluate.calibrate import check_calibration  # Engineer 8

        check_calibration(data, list(classes), manifest_sha256, run)
    threshold = data.get("unknown_threshold")
    if isinstance(threshold, dict):
        if threshold.get("score", "max_softmax") != "max_softmax":
            raise ValueError(f"{path}: only a max_softmax unknown threshold is supported, got {threshold.get('score')!r}")
        if threshold.get("applied_after_temperature") is False:
            raise ValueError(f"{path}: the threshold must be fitted after temperature scaling")
        threshold = threshold.get("value")
    if threshold is None:
        raise ValueError(f"{path} has no unknown_threshold; refusing to classify without one (SRS NFR-4)")
    return Calibration(temperature=float(data["temperature"]), unknown_threshold=float(threshold), source=str(path))
