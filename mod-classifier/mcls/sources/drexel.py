"""Drexel CBRS radar waveforms through a USRP, https://zenodo.org/records/4521678 (CC BY 4.0).

One archive, `Group6_data.tar` (34.1 GB). Its layout, read from the tar headers on
Nov 12 (no file extracted except to read its HDF5 layout):

    Group6_data/Group6_orig/Group6/group6_subset_<1..9>.mat   NIST simulated originals, 2.46 GB each
    Group6_data/Group6_exp/Group6/group6_subset_<1..9>.mat    the same waveforms sent through a USRP
                                                              over cable and recorded, 1.33 GB each
    Group6_data/Group6_<orig|exp>/Group6/group6_subset_CSVInfo/group6_waveformTableSubset_<n>.csv
    Group6_data/README.md

Only the `Group6_exp` files are read: they are the ones that passed through real
receive hardware. The `orig` files are skipped.

Each `.mat` is MATLAB v7.3 (HDF5) with, for subset n (the variable suffix does not
always match the file: group6_subset_2.mat holds `waveformSubset_1` on CRC, Nov 12,
so variables are found by prefix, one of each per file):

- `waveformSubset_n`: shape (200, 800000), compound {real, imag} float64. Row i is
  waveform i: 80 ms at 10 MS/s. Stored gzip-compressed in chunks of (200, 20),
  i.e. 20 samples of all 200 waveforms per chunk, so one waveform cannot be read
  without decompressing the whole dataset. The loader therefore reads a file's
  whole dataset once (in column blocks, as complex64: about 1.3 GB) and serves
  its 200 recordings from that; one file is held at a time. Read in order.
- `radarStatusSubset_n`: shape (200, 1), logical. 1 = radar present, 0 = noise
  only. 100 of each in the subset checked (subset 5's CSV).

Labels come from the taxonomy (`drexel_cbrs_radar`): each waveform is looked up as
`<file>#radar` or `<file>#noise`, so the taxonomy's `noise` override gives the
noise-only waveforms `background` and the rest get the default `pulsed_radar`.
The dataset is `is_real = false` (simulated waveforms), so its rows may appear in
training only, never in a reported test (PLAN 3.4).

Each waveform is its own `recording_id` (`<subset>_<index>`): the 200 waveforms of
a subset are separately generated, with their own radar parameters and noise.

Assumed (not stated in the files read): the RF centre. The CBRS band is
3550-3700 MHz; 3.6 GHz is used as a nominal centre for metadata and band priors.
The CSV's `SamplingFrequency` column (10,000,000) gives the rate; it is checked
when the CSV is present next to the file.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, Recording
from mcls.sources._common import check_range
from mcls.sources.base import Source, register

SAMPLE_RATE_HZ = 10e6
CENTER_FREQUENCY_HZ = 3.6e9  # nominal CBRS centre; see the module docstring
FILE = re.compile(r"^group6_subset_(?P<n>\d+)\.mat$")
COLUMN_BLOCK = 50_000  # samples per read when loading a file (200 x 50,000 x 16 bytes = 160 MB)


def subset_number(path: Path) -> int | None:
    """'.../Group6_exp/Group6/group6_subset_3.mat' -> 3; None for other files or the orig folder."""
    match = FILE.match(path.name)
    if not match or not any(part.endswith("_exp") for part in path.parts):
        return None
    return int(match["n"])


def only_key(hf, prefix: str, path: Path) -> str:
    """The one top-level variable whose name starts with prefix; an error when there is not exactly one."""
    keys = [k for k in hf.keys() if k.startswith(prefix)]
    if len(keys) != 1:
        raise ValueError(f"{path.name}: expected one {prefix}<n> variable, found {sorted(hf.keys())}")
    return keys[0]


def read_csv_rows(mat_path: Path, n: int) -> list[dict] | None:
    """Rows of the subset's waveform table CSV when it sits next to the file, else None."""
    path = mat_path.parent / "group6_subset_CSVInfo" / f"group6_waveformTableSubset_{n}.csv"
    if not path.is_file():
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def to_complex64(block: np.ndarray) -> np.ndarray:
    """Compound {real, imag} array (any float type) -> complex64 of the same shape."""
    out = np.empty(block.shape, dtype=SAMPLE_DTYPE)
    out.real = block["real"]
    out.imag = block["imag"]
    return out


class _FileCache:
    """Holds the decoded waveforms of one file; loading another file replaces them."""

    def __init__(self):
        self.path: Path | None = None
        self.data: np.ndarray | None = None

    def get(self, path: Path, key: str) -> np.ndarray:
        if self.path != path:
            import h5py

            self.data = None  # free the previous file before loading the next
            with h5py.File(path, "r") as hf:
                dataset = hf[key]
                rows, cols = dataset.shape
                data = np.empty((rows, cols), dtype=SAMPLE_DTYPE)
                for start in range(0, cols, COLUMN_BLOCK):
                    stop = min(start + COLUMN_BLOCK, cols)
                    data[:, start:stop] = to_complex64(dataset[:, start:stop])
            self.path, self.data = path, data
        return self.data


@register
class DrexelCbrsRadar(Source):
    name = "drexel_cbrs_radar"

    def __init__(self, root: Path | None = None, taxonomy=None):
        super().__init__(root, taxonomy)
        self._cache = _FileCache()

    def files(self) -> list[tuple[int, Path]]:
        found = ((subset_number(p), p) for p in self.root.rglob("group6_subset_*.mat"))
        return sorted((n, p) for n, p in found if n is not None)

    def recordings(self) -> Iterator[Recording]:
        import h5py

        info = self.taxonomy.dataset(self.name)
        for n, path in self.files():
            with h5py.File(path, "r") as hf:
                key = only_key(hf, "waveformSubset_", path)
                rows, num_samples = hf[key].shape
                status = np.asarray(hf[only_key(hf, "radarStatusSubset_", path)]).reshape(-1).astype(bool)
            if len(status) != rows:
                raise ValueError(f"{path.name}: {rows} waveforms but {len(status)} radar flags")
            table = read_csv_rows(path, n)
            if table is not None:
                if len(table) != rows:
                    raise ValueError(f"{path.name}: {rows} waveforms but the CSV has {len(table)} rows")
                flags = [row.get("radarStatus") for row in table]
                if flags != [str(int(s)) for s in status]:
                    raise ValueError(f"{path.name}: radar flags in the file and the CSV disagree")
                rates = {float(row["SamplingFrequency"]) for row in table if row.get("SamplingFrequency")}
                if rates and rates != {SAMPLE_RATE_HZ}:
                    raise ValueError(f"{path.name}: CSV sampling frequency {sorted(rates)}, expected {SAMPLE_RATE_HZ:g}")
            source_file = path.relative_to(self.root).as_posix()
            for i in range(rows):
                kind = "radar" if status[i] else "noise"
                label = self.taxonomy.label_for(self.name, f"{source_file}#{kind}")
                if label.excluded:
                    continue
                meta = {"subset": n, "index": i, "radar_status": int(status[i]), "note": label.note}
                if table is not None:
                    row = table[i]
                    meta.update({k: row.get(k) for k in ("BinNo", "SNR", "PulseWidth", "PulsesPerSecond", "SUID")})
                recording = Recording(
                    dataset=self.name,
                    recording_id=f"subset{n}_{i:03d}",
                    source_file=f"{source_file}#{i}",
                    sample_rate_hz=SAMPLE_RATE_HZ,
                    center_frequency_hz=CENTER_FREQUENCY_HZ,
                    num_samples=num_samples,
                    label=label.label,
                    label_strength=label.strength,
                    is_real=info.is_real,
                    license=info.license,
                    read=self._reader(path, key, i, num_samples),
                    emitter_id=label.emitter_id,
                    meta=meta,
                )
                recording.validate()
                yield recording

    def _reader(self, path: Path, key: str, index: int, num_samples: int):
        def read(start: int, count: int) -> np.ndarray:
            check_range(f"{path.name}#{index}", start, count, num_samples)
            return self._cache.get(path, key)[index, start : start + count].copy()

        return read

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        if not self.files():
            zipped = list(self.root.rglob("Group6_data.tar"))
            hint = " (Group6_data.tar is there: extract it first)" if zipped else ""
            problems.append(f"no Group6_exp/.../group6_subset_<n>.mat files under {self.root}{hint}")
        return problems
