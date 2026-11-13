"""rtl_433_tests, https://github.com/merbanan/rtl_433_tests (no licence stated; see below).

Sample recordings of consumer devices in the sub-GHz ISM bands (weather stations,
tyre-pressure sensors, doorbells, remotes, meters), made with RTL-SDR receivers
for the rtl_433 decoder's test suite. Layout, from the repository tree read on
Nov 12 (GitHub API, no file opened):

    tests/<device>/[<sub>/]<name>.cu8     3,724 files, 1.37 GB, 335 device folders

`.cu8` is interleaved unsigned 8-bit I/Q (127.5 is zero), the RTL-SDR's native
format. A few `.cs16` / `.cs8` files exist; they are skipped (15 files).
`false_positives_test/` (signals that must NOT decode) is skipped too.

Frequency and rate: 1,518 names state them as `..._<MHz>M_<kS/s>k.cu8`
(e.g. `g001_433.92M_250k.cu8`), some only the rate (`..._2500k.cu8`). The other
2,206 (`gfile001.cu8` ...) state neither; for those the rtl_433 defaults are
used, 433.92 MHz and 250 kS/s (assumed, not checked per file).

Split key: `recording_id` = the device folder, so all files of one device land
in one split and the test set holds devices the model has never seen.

Licence: the repository states none (GitHub API, Nov 12). Odon decided on Nov 12
to train on it anyway; the taxonomy records that, and every window carries the
licence string, so a model trained on it can be identified (PLAN 3.3).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from mcls.contracts import Recording
from mcls.sources._common import check_range, pairs_to_complex
from mcls.sources.base import Source, register

DEFAULT_FREQUENCY_HZ = 433.92e6
DEFAULT_RATE_HZ = 250e3
MIN_SAMPLES = 4096  # shorter files cannot give one window at their own rate
FREQ_RATE = re.compile(r"_(?P<mhz>\d+(?:\.\d+)?)M_(?P<ks>\d+(?:\.\d+)?)k(?:_|\.|$)", re.IGNORECASE)
RATE_ONLY = re.compile(r"_(?P<ks>\d+(?:\.\d+)?)k\.cu8$", re.IGNORECASE)
SKIP_PARTS = {"false_positives_test", ".git"}


def parse_name(name: str) -> tuple[float, float, bool]:
    """(centre Hz, rate Hz, stated) from a file name; stated is False when defaults were used."""
    match = FREQ_RATE.search(name)
    if match:
        return float(match["mhz"]) * 1e6, float(match["ks"]) * 1e3, True
    match = RATE_ONLY.search(name)
    if match:
        return DEFAULT_FREQUENCY_HZ, float(match["ks"]) * 1e3, False
    return DEFAULT_FREQUENCY_HZ, DEFAULT_RATE_HZ, False


def device_of(relative: Path) -> str | None:
    """'tests/6sc2/01/g001.cu8' -> '6sc2'; None outside tests/ or in a skipped folder."""
    parts = relative.parts
    if any(p in SKIP_PARTS for p in parts):
        return None
    if "tests" in parts:
        i = parts.index("tests")
        return parts[i + 1] if len(parts) > i + 2 else None
    return parts[0] if len(parts) > 1 else None


def make_reader(path: Path, num_samples: int):
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(2 * num_samples,))

    def read(start: int, count: int) -> np.ndarray:
        check_range(path.name, start, count, num_samples)
        return pairs_to_complex(raw[2 * start : 2 * (start + count)], np.uint8)

    return read


@register
class Rtl433(Source):
    name = "rtl_433"

    def files(self) -> list[tuple[str, Path]]:
        found = []
        for path in self.root.rglob("*.cu8"):
            device = device_of(path.relative_to(self.root))
            if device is not None:
                found.append((device, path))
        return sorted(found)

    def recordings(self) -> Iterator[Recording]:
        info = self.taxonomy.dataset(self.name)
        for device, path in self.files():
            relative = path.relative_to(self.root).as_posix()
            num_samples = path.stat().st_size // 2
            if num_samples < MIN_SAMPLES:
                continue
            label = self.taxonomy.label_for(self.name, relative)
            if label.excluded:
                continue
            centre, rate, stated = parse_name(path.name)
            recording = Recording(
                dataset=self.name,
                recording_id=device,
                source_file=relative,
                sample_rate_hz=rate,
                center_frequency_hz=centre,
                num_samples=num_samples,
                label=label.label,
                label_strength=label.strength,
                is_real=info.is_real,
                license=info.license,
                read=make_reader(path, num_samples),
                emitter_id=label.emitter_id or device,
                meta={"device": device, "rate_and_centre_from_name": stated, "note": label.note},
            )
            recording.validate()
            yield recording

    def check(self) -> list[str]:
        problems = super().check()
        if not problems and not self.files():
            problems.append(f"no tests/<device>/*.cu8 files under {self.root} (clone rtl_433_tests there)")
        return problems
