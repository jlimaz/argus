"""Triangular-jammed GPS L1 I/Q dumps, https://zenodo.org/records/14859381 (CC BY 4.0).

Facts read from the Zenodo record (https://zenodo.org/api/records/14859381), not
from a dataset file (nothing has been downloaded):

- Five files `LINKS_L1_10Mhz_complex_int8_20250118_<time>_TRI_<power>.gz`, power
  65, 70, 75, 80 and 85 (about 10 GB compressed each).
- "These gzip files contain I-Q data dumps of Jammed GNSS signal"; band L1;
  "Sample frequency: 10 MHz"; "Jamming signal: Triangular"; "Jamming power: 85,
  80, 75, 70, 65" (no unit given; kept as the number in the file name).

INFERRED, not stated by the record (unverified until a file is opened):

- Sample layout: interleaved signed int8, I then Q, inferred only from
  "complex_int8" in the file name. Scaled by 1/128.
- Centre frequency 1575.42 MHz, the nominal GPS L1 carrier, inferred from "L1".
  Every Recording says so in `meta`.

Each file is its own `recording_id` (the file name without `.gz`); the power
level is in `meta["jamming_power"]`. Files the taxonomy excludes are skipped.

Reading, and what it costs:

- If a decompressed sibling exists (same name without `.gz`), it is memory-mapped
  and read like any other interleaved file. This is the fast path; decompress
  first (`gunzip -k`) when there is room (the uncompressed size is not stated,
  but exceeds 4 GB).
- Otherwise the `.gz` is read through `gzip`, which cannot seek. Each Recording
  keeps one stream open: reading forward skips by decompressing and discarding
  (cost proportional to the distance), reading backward reopens the stream and
  decompresses from the start. Read each recording front to back.
- `num_samples` of a `.gz` needs the uncompressed length. The gzip trailer
  (ISIZE) holds it only modulo 2^32, useless above 4 GB, so the first
  `recordings()` or `check()` decompresses the whole file once (minutes per
  10 GB file) and caches the count in a sidecar `<name>.nsamples` next to it,
  together with the `.gz` size so a replaced file is counted again. If the
  folder is read-only the count is simply not cached.
"""

from __future__ import annotations

import gzip
import re
import threading
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np

from mcls.contracts import Recording
from mcls.sources._common import check_range, interleaved_reader, pairs_to_complex
from mcls.sources.base import Source, register

SAMPLE_RATE_HZ = 10e6  # stated by the record
#: Nominal GPS L1 carrier. INFERRED from "L1"; the record states no centre frequency.
CENTRE_HZ = 1575.42e6
#: INFERRED from "complex_int8" in the file name: interleaved signed int8, I then Q.
COMPONENT = np.dtype("i1")
BYTES_PER_SAMPLE = 2
POWERS = (65, 70, 75, 80, 85)

NAME = re.compile(r"^LINKS_L1_10Mhz_complex_int8_(?P<date>\d{8})_(?P<time>[^_]+)_TRI_(?P<power>\d+)$")
CHUNK = 1 << 22  # 4 MB of decompressed bytes per read when skipping or counting
GZIP_MAGIC = b"\x1f\x8b"
CENTRE_NOTE = "nominal GPS L1 carrier, inferred from 'L1'; the dataset does not state a centre frequency"
LAYOUT_NOTE = "interleaved int8 I then Q, inferred from 'complex_int8' in the file name; unverified"


def parse_name(stem: str) -> dict | None:
    """File name without `.gz` -> date, time, power; None when it does not fit."""
    match = NAME.match(stem)
    if not match:
        return None
    return {"date": match["date"], "time": match["time"], "power": int(match["power"])}


def stem_of(path: Path) -> str:
    return path.name[:-3] if path.name.endswith(".gz") else path.name


def sidecar(gz_path: Path) -> Path:
    return gz_path.with_name(gz_path.name + ".nsamples")


def count_gzip_bytes(gz_path: Path) -> int:
    """Uncompressed length of a gzip file, by one streaming pass. Raises EOFError if truncated."""
    total = 0
    with gzip.open(gz_path, "rb") as stream:
        while True:
            block = stream.read(CHUNK)
            if not block:
                return total
            total += len(block)


def gzip_num_samples(gz_path: Path) -> int:
    """Sample count of a `.gz`, from the sidecar when it matches the file, else counted (and cached)."""
    size = gz_path.stat().st_size
    cache = sidecar(gz_path)
    try:
        count, cached_size = (int(v) for v in cache.read_text().split())
        if cached_size == size:
            return count
    except (OSError, ValueError):
        pass
    nbytes = count_gzip_bytes(gz_path)
    if nbytes % BYTES_PER_SAMPLE:
        raise ValueError(f"{gz_path.name}: {nbytes} uncompressed bytes is not a whole number of int8 I/Q pairs")
    count = nbytes // BYTES_PER_SAMPLE
    try:
        cache.write_text(f"{count} {size}\n")
    except OSError:
        pass  # read-only folder: count again next time
    return count


class GzipReader:
    """read(start, count) over a gzip stream of interleaved components, efficient when reading forward.

    Keeps the stream open. A forward jump decompresses and discards the gap; a
    backward jump reopens the file and decompresses from the start.
    """

    def __init__(self, path: Path, component: np.dtype, num_samples: int):
        self.path, self.component, self.num_samples = Path(path), np.dtype(component), num_samples
        self.sample_bytes = 2 * self.component.itemsize
        self._stream: gzip.GzipFile | None = None
        self._pos = 0  # decompressed byte position of _stream
        self.restarts = 0  # times the stream was (re)opened; for tests and tuning
        self._lock = threading.Lock()

    def _open(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = gzip.open(self.path, "rb")
        self._pos = 0
        self.restarts += 1

    def _skip_to(self, offset: int) -> None:
        if self._stream is None or offset < self._pos:
            self._open()
        while self._pos < offset:
            got = len(self._stream.read(min(CHUNK, offset - self._pos)))
            if not got:
                raise EOFError(f"{self.path.name}: stream ended at byte {self._pos}, before {offset}")
            self._pos += got

    def __call__(self, start: int, count: int) -> np.ndarray:
        check_range(self.path.name, start, count, self.num_samples)
        with self._lock:
            self._skip_to(start * self.sample_bytes)
            want = count * self.sample_bytes
            data = self._stream.read(want)
            self._pos += len(data)
        if len(data) != want:
            raise EOFError(f"{self.path.name}: wanted {want} bytes at sample {start}, got {len(data)}")
        return pairs_to_complex(np.frombuffer(data, self.component), self.component)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


@register
class TriangularL1(Source):
    name = "triangular_l1"

    def files(self) -> dict[str, dict[str, Path]]:
        """stem -> {"raw": decompressed file, "gz": gzip file}, whichever exist."""
        found: dict[str, dict[str, Path]] = {}
        for path in sorted(self.root.rglob("LINKS_L1_*")):
            if not path.is_file() or path.name.endswith(".nsamples"):
                continue
            stem = stem_of(path)
            if parse_name(stem):
                found.setdefault(stem, {})["gz" if path.name.endswith(".gz") else "raw"] = path
        return found

    def recordings(self) -> Iterator[Recording]:
        info = self.taxonomy.dataset(self.name)
        for stem, paths in self.files().items():
            parsed = parse_name(stem)
            path = paths.get("raw", paths.get("gz"))
            label = self.taxonomy.label_for(self.name, path.relative_to(self.root).as_posix())
            if label.excluded:
                continue  # before counting: an excluded .gz is never decompressed
            if "raw" in paths:
                path = paths["raw"]
                size = path.stat().st_size
                if size % BYTES_PER_SAMPLE:
                    raise ValueError(f"{path.name}: {size} bytes is not a whole number of int8 I/Q pairs")
                num_samples = size // BYTES_PER_SAMPLE
                read, access = interleaved_reader(path, COMPONENT, num_samples), "memmap"
            else:
                path = paths["gz"]
                num_samples = gzip_num_samples(path)
                read, access = GzipReader(path, COMPONENT, num_samples), "gzip_stream"
            relative = path.relative_to(self.root).as_posix()
            recording = Recording(
                dataset=self.name,
                recording_id=stem,
                source_file=relative,
                sample_rate_hz=SAMPLE_RATE_HZ,
                center_frequency_hz=CENTRE_HZ,
                num_samples=num_samples,
                label=label.label,
                label_strength=label.strength,
                is_real=info.is_real,
                license=info.license,
                read=read,
                emitter_id=label.emitter_id,
                meta={
                    "jamming_power": parsed["power"],
                    "jamming_signal": "triangular",
                    "date": parsed["date"],
                    "time": parsed["time"],
                    "access": access,
                    "center_frequency_source": CENTRE_NOTE,
                    "sample_layout_source": LAYOUT_NOTE,
                    "note": label.note,
                },
            )
            recording.validate()
            yield recording

    def check(self) -> list[str]:
        """Problems, including truncated `.gz` files. Costs one full decompression per uncached `.gz`."""
        problems = super().check()
        if problems:
            return problems
        files = self.files()
        if not files:
            return [f"no LINKS_L1_*_TRI_<power>[.gz] files in {self.root}"]
        powers = {parse_name(stem)["power"] for stem in files}
        problems += [f"no file for jamming power {p}" for p in POWERS if p not in powers]
        for stem, paths in files.items():
            if "raw" in paths:
                size = paths["raw"].stat().st_size
                if size == 0 or size % BYTES_PER_SAMPLE:
                    problems.append(f"{paths['raw'].name}: {size} bytes is not a positive whole number of I/Q pairs (partial decompression?)")
                continue
            gz = paths["gz"]
            with open(gz, "rb") as handle:
                if handle.read(2) != GZIP_MAGIC:
                    problems.append(f"{gz.name}: not a gzip file")
                    continue
            try:
                if gzip_num_samples(gz) <= 0:
                    problems.append(f"{gz.name}: no samples")
            except (EOFError, OSError, zlib.error, ValueError) as err:
                problems.append(f"{gz.name}: unreadable ({err}); partial download?")
        return problems


if __name__ == "__main__":
    source = TriangularL1()
    for problem in source.check():
        print("problem:", problem)
    for rec in source.recordings():
        print(f"{rec.source_file:<70} power={rec.meta['jamming_power']} {rec.duration_s:8.2f} s  {rec.meta['access']}  {rec.label}")
