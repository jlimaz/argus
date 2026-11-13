"""Window access: memory-map shards and return windows by manifest rows (numpy only)."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, WINDOW_SAMPLES, ManifestRow
from mcls.dataset.rows import label_indices


class WindowStore:
    """Read windows from `.npy` shards without loading whole files.

    A shard is a complex64 `.npy`, either 1-D or 2-D `[k, WINDOW_SAMPLES]`;
    `row.offset` counts samples in the flattened array. Relative shard paths
    resolve against `root` (pass the manifest's directory).
    """

    def __init__(self, root: Path | None = None, max_open: int = 256):
        self.root = Path(root) if root is not None else None
        self.max_open = max_open
        self._open: dict[str, np.ndarray] = {}

    def path_of(self, row: ManifestRow) -> Path:
        path = Path(row.shard)
        if not path.is_absolute() and self.root is not None:
            path = self.root / path
        return path

    def _shard(self, row: ManifestRow) -> np.ndarray:
        key = str(self.path_of(row))
        array = self._open.get(key)
        if array is None:
            array = np.load(key, mmap_mode="r", allow_pickle=False)
            if array.dtype != SAMPLE_DTYPE:
                raise ValueError(f"{key}: shard dtype must be {np.dtype(SAMPLE_DTYPE)}, got {array.dtype}")
            if array.ndim == 2 and array.shape[1] != WINDOW_SAMPLES:
                raise ValueError(f"{key}: 2-D shard must be [k, {WINDOW_SAMPLES}], got {array.shape}")
            if array.ndim not in (1, 2):
                raise ValueError(f"{key}: shard must be 1-D or 2-D, got {array.ndim}-D")
            if not array.flags.c_contiguous:
                raise ValueError(f"{key}: shard must be C-ordered (flattening would copy the file)")
            array = array.reshape(-1)
            if len(self._open) >= self.max_open:
                self._open.pop(next(iter(self._open)))
            self._open[key] = array
        return array

    def window(self, row: ManifestRow) -> np.ndarray:
        """One window, complex64 [WINDOW_SAMPLES], copied out of the map."""
        shard = self._shard(row)
        end = row.offset + row.length
        if row.length != WINDOW_SAMPLES or row.offset < 0 or end > shard.shape[0]:
            raise ValueError(f"{row.window_id}: [{row.offset}, {end}) outside shard of {shard.shape[0]} samples")
        return np.array(shard[row.offset:end], dtype=SAMPLE_DTYPE)

    def windows(self, rows: Sequence[ManifestRow]) -> np.ndarray:
        """complex64 [n, WINDOW_SAMPLES] in row order; reads grouped by shard."""
        out = np.empty((len(rows), WINDOW_SAMPLES), dtype=SAMPLE_DTYPE)
        by_shard: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            by_shard[str(self.path_of(row))].append(i)
        for indices in by_shard.values():
            for i in sorted(indices, key=lambda j: rows[j].offset):
                out[i] = self.window(rows[i])
        return out

    def get(self, rows: Sequence[ManifestRow], classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(windows complex64 [n, WINDOW_SAMPLES], labels int64 [n])."""
        return self.windows(rows), label_indices(rows, classes)


def window_hash(window: np.ndarray) -> str:
    """sha256 of a window's complex64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(window, dtype=SAMPLE_DTYPE).tobytes()).hexdigest()


def audit_window_hashes(
    rows: Sequence[ManifestRow], store: WindowStore, batch: int = 4096
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Find identical windows (byte for byte) that sit in more than one split (PLAN 3.5).

    Returns [(sha256, [(window_id, split), ...]), ...] for every sample hash
    seen in two or more splits, sorted by hash; an empty list means clean.
    Duplicates inside one split are not leakage and are not reported.
    """
    seen: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        for row, window in zip(chunk, store.windows(chunk)):
            seen[window_hash(window)].append((row.window_id, row.split))
    leaks = []
    for digest, members in sorted(seen.items()):
        if len({split for _, split in members}) > 1:
            leaks.append((digest, sorted(members)))
    return leaks
