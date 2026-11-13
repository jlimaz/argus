"""The contracts every stage of the pipeline shares.

Each dataset arrives in its own format (little- or big-endian int16, int8, real
floats inside RAR archives, HDF5, SigMF, MATLAB). Those differences are allowed
in exactly one place: the dataset's loader in `mcls/sources/`. Everything a
loader hands on follows this module, so the front end, training and evaluation
never see a dataset's quirks.

Three contracts:

1. `Recording`: what a loader yields. A handle on one continuous capture that
   reads complex64 samples on demand, plus its metadata and label.
2. `ManifestRow`: one window of the training set, after the front end has
   resampled to `SAMPLE_RATE_HZ` and cut `WINDOW_SAMPLES` samples.
3. `Classifier`: what a trained model offers the rest of Argus.

Change a contract only through the orchestrator: every engineer builds on it.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Canonical sample format
# ---------------------------------------------------------------------------

#: Every window is resampled to this rate: the HackRF One's widest bandwidth.
SAMPLE_RATE_HZ = 20_000_000.0
#: Samples per window: about 205 microseconds at SAMPLE_RATE_HZ.
WINDOW_SAMPLES = 4096
#: Samples are complex64 (float32 I and Q), scaled to roughly -1..1 by the loader.
SAMPLE_DTYPE = np.complex64
#: Each window is normalized to unit RMS by the front end; absolute power never
#: reaches a model (matches pipeline/features.py, which keeps only ratios).

LABEL_STRENGTHS = ("strong", "weak")
#: `generalization` holds whole held-out emitters (drone models) for the separate
#: "generalize to a new drone" test (PLAN 3.5); it never feeds the headline test.
SPLITS = ("train", "val", "test", "cross_test", "generalization")

# ---------------------------------------------------------------------------
# Bands: derived from the centre frequency, never learned
# ---------------------------------------------------------------------------

#: (name, low Hz, high Hz). Checked in order, so narrow bands come before the
#: wide ones that contain them.
BANDS: tuple[tuple[str, float, float], ...] = (
    ("adsb_1090", 1_085e6, 1_095e6),
    ("gnss_l1", 1_559e6, 1_610e6),
    ("fm_broadcast", 88e6, 108e6),
    ("ism_433", 433.05e6, 434.79e6),
    ("srd_868", 863e6, 870e6),
    ("ism_915", 902e6, 928e6),
    ("ism_2400", 2_400e6, 2_483.5e6),
    ("cbrs_3500", 3_550e6, 3_700e6),
    ("ism_5800", 5_725e6, 5_875e6),
    ("unii_5ghz", 5_150e6, 5_925e6),
    ("hf", 3e6, 30e6),
)


def band_of(center_frequency_hz: float) -> str:
    """Name the project band a centre frequency falls in, or "other"."""
    for name, low, high in BANDS:
        if low <= center_frequency_hz <= high:
            return name
    return "other"


# ---------------------------------------------------------------------------
# 1. Recording: what a loader yields
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    """One continuous capture, read on demand.

    A loader never loads a whole file: recordings can be gigabytes. `read`
    returns `count` complex64 samples starting at `start`, scaled to about
    -1..1, at the recording's native `sample_rate_hz`.

    `recording_id` is the split key: two files that are halves of the same
    capture share one id so they can never land on both sides of a split.
    """

    dataset: str
    recording_id: str
    source_file: str
    sample_rate_hz: float
    center_frequency_hz: float
    num_samples: int
    label: str
    label_strength: str
    is_real: bool
    license: str
    read: Callable[[int, int], np.ndarray] = field(repr=False)
    emitter_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def band(self) -> str:
        return band_of(self.center_frequency_hz)

    @property
    def duration_s(self) -> float:
        return self.num_samples / self.sample_rate_hz

    def validate(self) -> None:
        """Raise ValueError when the metadata breaks the contract."""
        if not self.dataset or not self.recording_id:
            raise ValueError("dataset and recording_id are required")
        if self.sample_rate_hz <= 0 or self.num_samples <= 0:
            raise ValueError(f"{self.recording_id}: sample rate and length must be positive")
        if self.center_frequency_hz < 0:
            raise ValueError(f"{self.recording_id}: centre frequency must be zero or more")
        if self.label_strength not in LABEL_STRENGTHS:
            raise ValueError(f"{self.recording_id}: label_strength must be one of {LABEL_STRENGTHS}")
        if not self.license:
            raise ValueError(f"{self.recording_id}: every recording carries its licence")


def check_samples(samples: np.ndarray) -> None:
    """Raise ValueError unless `samples` is a 1-D complex64 array of finite values."""
    if samples.dtype != SAMPLE_DTYPE or samples.ndim != 1:
        raise ValueError(f"samples must be 1-D {np.dtype(SAMPLE_DTYPE)}, got {samples.ndim}-D {samples.dtype}")
    if not np.isfinite(samples.view(np.float32)).all():
        raise ValueError("samples contain NaN or infinity")


# ---------------------------------------------------------------------------
# 2. ManifestRow: one training window
# ---------------------------------------------------------------------------


@dataclass
class ManifestRow:
    """One window in the training set.

    The samples live in `shard` (a .npy file of complex64 samples at
    SAMPLE_RATE_HZ) from `offset` for `length` samples. Fields left as None
    are allowed; everything else is required.
    """

    window_id: str
    shard: str
    offset: int
    length: int
    dataset: str
    recording_id: str
    label: str
    label_strength: str
    native_rate_hz: float
    center_frequency_hz: float
    band: str
    is_real: bool
    license: str
    split: str
    emitter_id: str | None = None
    snr_db_est: float | None = None
    bandwidth_hz: float | None = None
    duty_cycle: float | None = None
    #: Windows cut from the same detected burst share this id (front end sets it;
    #: None for background windows). Burst-level metrics group by it.
    burst_id: str | None = None

    def validate(self, classes: tuple[str, ...] | None = None) -> None:
        """Raise ValueError when the row breaks the contract."""
        if self.length != WINDOW_SAMPLES:
            raise ValueError(f"{self.window_id}: length must be {WINDOW_SAMPLES}, got {self.length}")
        if self.offset < 0:
            raise ValueError(f"{self.window_id}: offset must be zero or more")
        if self.split not in SPLITS:
            raise ValueError(f"{self.window_id}: split must be one of {SPLITS}")
        if self.label_strength not in LABEL_STRENGTHS:
            raise ValueError(f"{self.window_id}: label_strength must be one of {LABEL_STRENGTHS}")
        if classes is not None and self.label not in classes:
            raise ValueError(f"{self.window_id}: label {self.label!r} is not one of {classes}")
        if not self.license:
            raise ValueError(f"{self.window_id}: licence missing")
        for name in ("snr_db_est", "bandwidth_hz", "duty_cycle"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{self.window_id}: {name} is not finite")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestRow":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown manifest fields: {sorted(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------------
# 3. Classifier: what a trained model offers Argus
# ---------------------------------------------------------------------------


@runtime_checkable
class Classifier(Protocol):
    """A trained model, as the rest of Argus calls it.

    `classify` takes windows shaped (n, WINDOW_SAMPLES), complex64, at
    SAMPLE_RATE_HZ and normalized to unit RMS, and returns probabilities shaped
    (n, len(classes)) whose rows sum to 1 (a softmax output). Unknown is not a
    class: the caller treats a low top probability as unknown.
    """

    classes: tuple[str, ...]

    def classify(self, windows: np.ndarray) -> np.ndarray: ...


def check_windows(windows: np.ndarray) -> None:
    """Raise ValueError unless `windows` is shaped (n, WINDOW_SAMPLES) complex64."""
    if windows.ndim != 2 or windows.shape[1] != WINDOW_SAMPLES or windows.dtype != SAMPLE_DTYPE:
        raise ValueError(
            f"windows must be (n, {WINDOW_SAMPLES}) {np.dtype(SAMPLE_DTYPE)}, got {windows.shape} {windows.dtype}"
        )


def check_probabilities(probabilities: np.ndarray, n: int, classes: tuple[str, ...]) -> None:
    """Raise ValueError unless `probabilities` is a valid classify() output."""
    if probabilities.shape != (n, len(classes)):
        raise ValueError(f"expected shape {(n, len(classes))}, got {probabilities.shape}")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("each row of probabilities must sum to 1")
