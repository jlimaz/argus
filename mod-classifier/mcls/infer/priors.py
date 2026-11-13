"""Band prior (PLAN 5.7): how plausible each class is at a centre frequency.

`configs/band_priors.toml` holds a list of bands:

    [[band]]
    name = "fm_broadcast"
    lo_hz = 88e6
    hi_hz = 108e6
    source = "https://www.ntia.gov/..."   # or "unsourced"
    note = "..."
    [band.weights]
    wifi = 0.05

A class missing from `weights` gets 1.0, and a frequency outside every band
gets 1.0 for every class, so the prior only ever lowers what a band makes
implausible. The first band that contains the frequency wins, so list narrow
bands first. Weights must be positive: a soft prior, never a mask, so a
surprise stays visible.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: mod-classifier/configs/band_priors.toml
DEFAULT_PRIORS = Path(__file__).resolve().parents[2] / "configs" / "band_priors.toml"


@dataclass(frozen=True)
class Band:
    name: str
    lo_hz: float
    hi_hz: float
    weights: dict[str, float]
    source: str
    note: str = ""


@dataclass(frozen=True)
class BandPriors:
    bands: tuple[Band, ...] = field(default_factory=tuple)

    def band_for(self, frequency_hz: float) -> Band | None:
        for band in self.bands:
            if band.lo_hz <= frequency_hz <= band.hi_hz:
                return band
        return None

    def for_frequency(self, frequency_hz: float, classes: tuple[str, ...]) -> np.ndarray:
        """Weights shaped (len(classes),) for a centre frequency; all ones outside every band."""
        band = self.band_for(frequency_hz)
        if band is None:
            return np.ones(len(classes))
        return np.array([band.weights.get(name, 1.0) for name in classes], dtype=np.float64)


def parse_priors(data: dict) -> BandPriors:
    bands = []
    for index, entry in enumerate(data.get("band", [])):
        where = f"band {index + 1} ({entry.get('name', '?')})"
        lo, hi = float(entry["lo_hz"]), float(entry["hi_hz"])
        if not (0 <= lo < hi):
            raise ValueError(f"{where}: need 0 <= lo_hz < hi_hz")
        source = entry.get("source")
        if not source or not (source == "unsourced" or source.startswith("https://")):
            raise ValueError(f"{where}: source must be an https URL that was opened, or \"unsourced\"")
        weights = {str(k): float(v) for k, v in entry.get("weights", {}).items()}
        for name, value in weights.items():
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{where}: weight for {name} must be positive (a prior, not a mask)")
        bands.append(Band(str(entry.get("name", f"band{index + 1}")), lo, hi, weights, source, str(entry.get("note", ""))))
    return BandPriors(tuple(bands))


def load_priors(path: str | Path | None = None) -> BandPriors:
    """Band priors from a TOML file (default: configs/band_priors.toml)."""
    path = Path(path) if path is not None else DEFAULT_PRIORS
    with open(path, "rb") as handle:
        return parse_priors(tomllib.load(handle))
