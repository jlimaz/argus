"""Shard files: 1-D complex64 .npy written incrementally, plus a JSONL rows file beside each.

A shard is a plain NumPy .npy (format 1.0) holding a 1-D little-endian
complex64 array of n_windows x 4096 samples; window i is
shard[i * 4096 : (i + 1) * 4096]. The header is reserved at a fixed 128 bytes
and rewritten with the final shape on close, so windows stream to disk without
holding them in memory. Open with np.load(path, mmap_mode="r").

Each `<name>.npy` has `<name>.rows.jsonl`: one ManifestRow-shaped JSON object per
window, in shard order, without "split" (assigned later by the manifest step).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mcls.contracts import WINDOW_SAMPLES

HEADER_BYTES = 128
MAGIC = b"\x93NUMPY\x01\x00"
#: About 1 GB of complex64 per shard.
MAX_WINDOWS_PER_SHARD = 32768


def npy_header(num_samples: int) -> bytes:
    """A .npy v1.0 header for a 1-D '<c8' array, padded to exactly HEADER_BYTES."""
    text = "{'descr': '<c8', 'fortran_order': False, 'shape': (%d,), }" % num_samples
    room = HEADER_BYTES - len(MAGIC) - 2
    text = text.ljust(room - 1) + "\n"
    if len(text) != room:
        raise ValueError("shape too large for the reserved header")
    return MAGIC + np.uint16(room).astype("<u2").tobytes() + text.encode("latin1")


class ShardWriter:
    """Append normalized windows and their rows; rolls to a new shard every `max_windows`."""

    def __init__(self, out_dir: Path, stem: str, max_windows: int = MAX_WINDOWS_PER_SHARD):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stem = stem
        self.max_windows = max_windows
        self.index = -1
        self.count = 0  # windows in the current shard
        self.total = 0
        self.shards: list[str] = []
        self._data = None
        self._rows = None

    @property
    def shard_name(self) -> str:
        return f"{self.stem}.s{self.index:03d}.npy"

    def _open(self) -> None:
        self._close_current()
        self.index += 1
        self.count = 0
        self.shards.append(self.shard_name)
        self._data = open(self.out_dir / self.shard_name, "wb")
        self._data.write(npy_header(0))
        self._rows = open(self.out_dir / f"{self.stem}.s{self.index:03d}.rows.jsonl", "w")

    def _close_current(self) -> None:
        if self._data is None:
            return
        self._data.seek(0)
        self._data.write(npy_header(self.count * WINDOW_SAMPLES))
        self._data.close()
        self._rows.close()
        self._data = self._rows = None

    def add(self, windows: np.ndarray, rows: list[dict]) -> None:
        """`windows` (n, 4096) complex64; `rows` n dicts without shard/offset (filled here)."""
        if len(windows) != len(rows):
            raise ValueError("one row per window")
        for window, row in zip(windows, rows):
            if self._data is None or self.count >= self.max_windows:
                self._open()
            row = dict(row, shard=self.shard_name, offset=self.count * WINDOW_SAMPLES, length=WINDOW_SAMPLES)
            self._data.write(np.ascontiguousarray(window, dtype="<c8").tobytes())
            self._rows.write(json.dumps(row) + "\n")
            self.count += 1
            self.total += 1

    def close(self) -> None:
        self._close_current()
