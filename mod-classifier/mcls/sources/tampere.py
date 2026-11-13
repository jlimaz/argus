"""Tampere "Radio-Frequency Control and Video Signal Recordings of Drones".

https://zenodo.org/records/4264467 (CC BY 4.0). Ten drones recorded in an
anechoic chamber. Each .bin file is interleaved I/Q as int16 little-endian
(confirmed against the dataset's own load_bin.py):

- `*_2G*.bin`: centred at 2.44 GHz, 120 MS/s, 120e6 samples (1 s, 480 MB)
- `*_5G*.bin`: centred at 5.8 GHz (Yuneec Typhoon H: 5.7 GHz), 200 MS/s,
  100e6 samples (0.5 s, 400 MB)

Files named `_1of2` and `_2of2` are two halves of one capture, so they share a
`recording_id` and always land in the same split.

Labels and emitter ids come from the taxonomy; files it excludes are skipped,
and the emitter parsed from the file name is only the fallback.

This is the reference loader for the `Source` contract: the other loaders
follow its shape.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, Recording
from mcls.sources.base import Source, register

#: band code -> (centre frequency Hz, sample rate Hz)
BANDS = {"2G": (2.44e9, 120e6), "5G": (5.8e9, 200e6)}
#: drones whose 5G recording is centred elsewhere (per the dataset page)
CENTRE_OVERRIDES = {("Yuneec_typhoon_h", "5G"): 5.7e9}

NAME = re.compile(r"^(?P<emitter>.+?)(?:_(?P<link>control|video))?_(?P<band>2G|5G)(?:_(?P<part>\d)of(?P<parts>\d))?\.bin$")

BYTES_PER_SAMPLE = 4  # int16 I + int16 Q
SCALE = 1.0 / 32768.0


def parse_name(file_name: str) -> dict | None:
    """'DJI_inspire_2_5G_1of2.bin' -> emitter, link, band, part; None when the name does not fit."""
    match = NAME.match(file_name)
    if not match:
        return None
    fields = match.groupdict()
    return {
        "emitter": fields["emitter"],
        "link": fields["link"] or "control_and_video",
        "band": fields["band"],
        "part": int(fields["part"]) if fields["part"] else 1,
        "parts": int(fields["parts"]) if fields["parts"] else 1,
    }


def make_reader(path: Path, num_samples: int):
    """A read(start, count) function over a memory-mapped int16 LE file."""
    raw = np.memmap(path, dtype="<i2", mode="r", shape=(2 * num_samples,))

    def read(start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > num_samples:
            raise IndexError(f"{path.name}: samples {start}..{start + count} outside 0..{num_samples}")
        pairs = raw[2 * start : 2 * (start + count)].astype(np.float32) * SCALE
        return pairs.view(SAMPLE_DTYPE)

    return read


@register
class Tampere(Source):
    name = "tampere"

    def files(self) -> list[Path]:
        return sorted(p for p in self.root.glob("*.bin") if parse_name(p.name))

    def recordings(self) -> Iterator[Recording]:
        info = self.taxonomy.dataset(self.name)
        for path in self.files():
            parsed = parse_name(path.name)
            centre, rate = BANDS[parsed["band"]]
            centre = CENTRE_OVERRIDES.get((parsed["emitter"], parsed["band"]), centre)
            size = path.stat().st_size
            if size % BYTES_PER_SAMPLE:
                raise ValueError(f"{path.name}: {size} bytes is not a whole number of int16 I/Q pairs")
            num_samples = size // BYTES_PER_SAMPLE
            label = self.taxonomy.label_for(self.name, path.name)
            if label.excluded:
                continue
            link = parsed["link"]
            recording = Recording(
                dataset=self.name,
                recording_id=f"{parsed['emitter']}_{link}_{parsed['band']}",
                source_file=path.name,
                sample_rate_hz=rate,
                center_frequency_hz=centre,
                num_samples=num_samples,
                label=label.label,
                label_strength=label.strength,
                is_real=info.is_real,
                license=info.license,
                read=make_reader(path, num_samples),
                emitter_id=label.emitter_id or parsed["emitter"],
                meta={"link": link, "part": parsed["part"], "parts": parsed["parts"], "note": label.note},
            )
            recording.validate()
            yield recording

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        problems += [f"{p.name}: name does not match the Tampere pattern" for p in self.root.glob("*.bin") if not parse_name(p.name)]
        for path in self.files():
            band = parse_name(path.name)["band"]
            expected = {"2G": 480_000_000, "5G": 400_000_000}[band]
            if path.stat().st_size != expected:
                problems.append(f"{path.name}: {path.stat().st_size} bytes, expected {expected} (partial download?)")
        if not self.files():
            problems.append(f"no Tampere .bin files in {self.root}")
        return problems


if __name__ == "__main__":
    source = Tampere()
    for problem in source.check():
        print("problem:", problem)
    for rec in source.recordings():
        first = rec.read(0, 4)
        print(
            f"{rec.source_file:<32} {rec.recording_id:<34} {rec.band:<9} {rec.sample_rate_hz / 1e6:>5.0f} MS/s "
            f"{rec.duration_s:.2f} s  label={rec.label} ({rec.label_strength})  first={np.round(first, 4)}"
        )
