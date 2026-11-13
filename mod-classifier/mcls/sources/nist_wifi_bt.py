"""NIST "Wi-Fi and Bluetooth I/Q Recordings in the 2.4 GHz and 5 GHz Bands with
Low-Cost Software Defined Radios", https://data.nist.gov/od/id/mds2-2731
(DOI 10.18434/mds2-2731; methods in NIST TN 2237, DOI 10.6028/NIST.TN.2237).

Five HDF5 files of 180 one-second captures each, at 30 MS/s, each with a
metadata CSV of the same stem:

- `2_4ghz_indoor.h5`, `2_4ghz_outdoor.h5`: Wi-Fi, centre 2437 MHz, configuration #1
- `5ghz_indoor.h5`, `5ghz_outdoor.h5`: Wi-Fi, centre 5825 MHz, configuration #1
- `2_4ghz_bluetooth.h5`: co-channel Wi-Fi and Bluetooth, 2437 MHz, configuration #2

Each capture is one Recording, read lazily by slicing the HDF5 dataset.

Verified against official documentation (README.txt on data.nist.gov and
TN 2237 section 2, read Nov 11; no file opened):
- file names, centres, rate, 180 captures per file, 1 s per capture;
- "each I/Q capture is stored as interleaved 16-bit signed integers", I first,
  "with timestamps as their recording keys" (top-level keys of the file), and
  the official scaling is /32768.

Assumed (not checked against a file):
- each capture is a 1-D dataset directly under the root, length 2 x 30e6
  (`KEY_GROUP` and the expected length are configurable if not);
- the CSV columns are unknown. The CSV is read as-is: a row whose value equals a
  capture key becomes that capture's `meta["csv"]`; otherwise a single-row CSV is
  attached to every capture of the file.

Leakage: the indoor and outdoor files of one band were recorded simultaneously
(the same transmissions seen by two receivers), so captures with the same
timestamp in `<band>_indoor` and `<band>_outdoor` share one recording_id.
That grouping is only as good as the keys: it assumes the two receivers wrote
identical timestamp keys for one moment (not verified against a file). If their
clocks differ, simultaneous captures get different ids and could be split apart.
`check()` therefore reports any key found in only one file of an indoor/outdoor
pair; resolve those (e.g. by matching nearest timestamps) before NIST windows
feed a reported split.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from mcls.contracts import Recording
from mcls.sources._common import check_range, pairs_to_complex
from mcls.sources.base import Source, register

SAMPLE_RATE_HZ = 30e6
CAPTURE_SECONDS = 1.0
CAPTURES_PER_FILE = 180
#: file stem prefix -> centre frequency Hz
CENTRES = {"2_4ghz": 2437e6, "5ghz": 5825e6}
NAME = re.compile(r"^(?P<band>2_4ghz|5ghz)_(?P<setup>indoor|outdoor|bluetooth)\.h5$", re.IGNORECASE)


def parse_name(file_name: str) -> dict | None:
    """'2_4ghz_indoor.h5' -> band prefix, setup, centre; None when the name does not fit."""
    match = NAME.match(file_name)
    if not match:
        return None
    band, setup = match["band"].lower(), match["setup"].lower()
    return {
        "band": band,
        "setup": setup,
        "centre_hz": CENTRES[band],
        # indoor and outdoor of one band were recorded at the same time
        "group": f"{band}_wifi" if setup in ("indoor", "outdoor") else f"{band}_{setup}",
    }


def require_h5py():
    try:
        import h5py
    except ImportError as err:  # pragma: no cover - depends on the venv
        raise ImportError("the NIST Wi-Fi/Bluetooth loader needs h5py: pip install h5py") from err
    return h5py


def read_metadata(csv_path: Path, keys: list[str]) -> dict[str, dict]:
    """Per-capture CSV rows keyed by capture key, or {"*": row} for a one-row CSV."""
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    wanted = set(keys)
    per_key = {}
    for row in rows:
        hit = next((v.strip() for v in row.values() if isinstance(v, str) and v.strip() in wanted), None)
        if hit is not None:
            per_key[hit] = row
    if per_key:
        return per_key
    return {"*": rows[0]} if len(rows) == 1 else {}


def grouping_problems(keys_by_file: dict[str, list[str]]) -> list[str]:
    """Keys of an indoor/outdoor pair that do not match: their captures would not share a recording_id."""
    by_group: dict[str, dict[str, set[str]]] = {}
    for name, keys in keys_by_file.items():
        parsed = parse_name(name)
        if parsed and parsed["setup"] in ("indoor", "outdoor"):
            by_group.setdefault(parsed["group"], {})[name] = set(keys)
    problems = []
    for group, files in sorted(by_group.items()):
        if len(files) != 2:
            continue  # only one of the pair present: nothing to pair
        (a, ka), (b, kb) = sorted(files.items())
        for here, there, only in ((a, b, ka - kb), (b, a, kb - ka)):
            if only:
                sample = ", ".join(sorted(only)[:3])
                problems.append(
                    f"{here}: {len(only)} capture keys have no twin in {there} (e.g. {sample}); "
                    "the indoor/outdoor recording_id grouping does not hold for them"
                )
    return problems


@register
class NistWifiBt(Source):
    name = "nist_wifi_bt"
    #: HDF5 group holding the captures ("/" per the README); change if the files differ.
    key_group = "/"
    expected_components = int(2 * SAMPLE_RATE_HZ * CAPTURE_SECONDS)

    def files(self) -> list[Path]:
        return sorted(p for p in self.root.glob("*.h5") if parse_name(p.name))

    def _capture_keys(self, handle) -> list[str]:
        h5py = require_h5py()
        group = handle[self.key_group]
        return sorted(k for k, v in group.items() if isinstance(v, h5py.Dataset))

    def _handle(self, path: Path):
        """One open HDF5 file per Source (the last one read), so handles do not pile up."""
        cached = getattr(self, "_open", None)
        if cached is None or cached[0] != path:
            if cached is not None:
                cached[1].close()
            self._open = (path, require_h5py().File(path, "r"))
        return self._open[1]

    def _make_reader(self, path: Path, key: str, num_samples: int):
        def read(start: int, count: int) -> np.ndarray:
            check_range(f"{path.name}:{key}", start, count, num_samples)
            pairs = self._handle(path)[self.key_group][key][2 * start : 2 * (start + count)]
            return pairs_to_complex(pairs, np.dtype(np.int16))

        return read

    def recordings(self) -> Iterator[Recording]:
        h5py = require_h5py()
        info = self.taxonomy.dataset(self.name)
        for path in self.files():
            parsed = parse_name(path.name)
            label = self.taxonomy.label_for(self.name, path.name)
            if label.excluded:
                continue
            with h5py.File(path, "r") as handle:
                keys = self._capture_keys(handle)
                shapes = {k: (handle[self.key_group][k].shape, handle[self.key_group][k].dtype) for k in keys}
            metadata = read_metadata(path.with_suffix(".csv"), keys)
            for key in keys:
                shape, dtype = shapes[key]
                if len(shape) != 1 or shape[0] % 2 or dtype.kind != "i" or dtype.itemsize != 2:
                    raise ValueError(f"{path.name}:{key}: expected 1-D interleaved int16, got {shape} {dtype}")
                num_samples = shape[0] // 2
                recording = Recording(
                    dataset=self.name,
                    recording_id=f"{parsed['group']}_{key}",
                    source_file=f"{path.name}:{key}",
                    sample_rate_hz=SAMPLE_RATE_HZ,
                    center_frequency_hz=parsed["centre_hz"],
                    num_samples=num_samples,
                    label=label.label,
                    label_strength=label.strength,
                    is_real=info.is_real,
                    license=info.license,
                    read=self._make_reader(path, key, num_samples),
                    emitter_id=label.emitter_id,
                    meta={
                        "setup": parsed["setup"],
                        "capture_key": key,
                        "csv": metadata.get(key, metadata.get("*")),
                        "note": label.note,
                    },
                )
                recording.validate()
                yield recording

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        h5py = require_h5py()
        if not self.files():
            return [f"no NIST .h5 files in {self.root}"]
        keys_by_file: dict[str, list[str]] = {}
        for path in self.files():
            if not path.with_suffix(".csv").exists():
                problems.append(f"{path.name}: metadata {path.with_suffix('.csv').name} missing")
            try:
                with h5py.File(path, "r") as handle:
                    keys = self._capture_keys(handle)
                    keys_by_file[path.name] = keys
                    short = [k for k in keys if handle[self.key_group][k].shape != (self.expected_components,)]
            except OSError as err:
                problems.append(f"{path.name}: cannot open as HDF5 ({err}); partial download?")
                continue
            if len(keys) != CAPTURES_PER_FILE:
                problems.append(f"{path.name}: {len(keys)} captures, expected {CAPTURES_PER_FILE}")
            if short:
                problems.append(f"{path.name}: {len(short)} captures are not {self.expected_components} int16 values long")
        problems += grouping_problems(keys_by_file)
        return problems


if __name__ == "__main__":
    source = NistWifiBt()
    for problem in source.check():
        print("problem:", problem)
    for rec in source.recordings():
        print(f"{rec.source_file:<44} {rec.recording_id:<40} {rec.band:<9} {rec.label} ({rec.label_strength})")
