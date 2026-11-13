"""Helpers shared by the loaders: interleaved-integer readers, bounds checks, tar members.

Kept private (leading underscore): nothing outside `mcls/sources/` should need it.
"""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path
from typing import Callable

import numpy as np

from mcls.contracts import SAMPLE_DTYPE

#: The repository root (argus/), so loaders can import `pipeline.features`.
REPO_ROOT = Path(__file__).resolve().parents[3]


def import_pipeline_features():
    """`pipeline.features` from the repository root, adding the root to sys.path if needed."""
    try:
        from pipeline import features
    except ImportError:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from pipeline import features
    return features


def check_range(name: str, start: int, count: int, num_samples: int) -> None:
    if start < 0 or count < 0 or start + count > num_samples:
        raise IndexError(f"{name}: samples {start}..{start + count} outside 0..{num_samples}")


def int_scale(dtype: np.dtype) -> tuple[float, float]:
    """(offset, scale) that maps an integer dtype to about -1..1: (x - offset) * scale."""
    dtype = np.dtype(dtype)
    if dtype.kind == "i":
        return 0.0, 1.0 / float(2 ** (8 * dtype.itemsize - 1))
    if dtype.kind == "u":
        half = (2 ** (8 * dtype.itemsize) - 1) / 2.0  # cu8: 127.5 is zero
        return half, 1.0 / half
    if dtype.kind == "f":
        return 0.0, 1.0
    raise ValueError(f"unsupported sample component type {dtype}")


def pairs_to_complex(pairs: np.ndarray, component: np.dtype) -> np.ndarray:
    """Interleaved I, Q, I, Q components (any int or float dtype) -> complex64 about -1..1."""
    offset, scale = int_scale(component)
    values = pairs.astype(np.float32)
    if offset:
        values -= np.float32(offset)
    if scale != 1.0:
        values *= np.float32(scale)
    return np.ascontiguousarray(values).view(SAMPLE_DTYPE)


def interleaved_reader(
    path: Path, component: np.dtype, num_samples: int, byte_offset: int = 0
) -> Callable[[int, int], np.ndarray]:
    """read(start, count) over a memory-mapped file of interleaved I/Q components.

    `component` carries the byte order, e.g. ">i2" for big-endian int16; numpy
    swaps on the fly, so a wrong byte order here gives wrong values, not an error.
    """
    component = np.dtype(component)
    raw = np.memmap(path, dtype=component, mode="r", offset=byte_offset, shape=(2 * num_samples,))

    def read(start: int, count: int) -> np.ndarray:
        check_range(Path(path).name, start, count, num_samples)
        return pairs_to_complex(raw[2 * start : 2 * (start + count)], component)

    return read


class SubFile(io.RawIOBase):
    """A read-only, seekable window [offset, offset + size) of a larger file.

    Lets h5py or scipy open one member of an uncompressed tar in place, without
    extracting it.
    """

    def __init__(self, path: Path, offset: int, size: int):
        super().__init__()
        self._handle = open(path, "rb")
        self._offset, self._size, self._pos = offset, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + pos)
        return self._pos

    def readinto(self, buffer) -> int:
        remaining = self._size - self._pos
        if remaining <= 0:
            return 0
        view = memoryview(buffer)[: min(len(buffer), remaining)]
        self._handle.seek(self._offset + self._pos)
        got = self._handle.readinto(view)
        self._pos += got
        return got

    def close(self) -> None:
        self._handle.close()
        super().close()


def tar_member_window(tar_path: Path, member: tarfile.TarInfo) -> SubFile:
    """Open one member of an uncompressed tar as a seekable file, in place."""
    if not member.isfile():
        raise ValueError(f"{member.name} is not a regular file")
    return SubFile(tar_path, member.offset_data, member.size)


#: MATLAB .mat header: 116 bytes of text, 8 of subsystem offset, 2 of version, 2 of endian flag.
MAT_HEADER_BYTES = 128


def mat_version(header: bytes) -> str:
    """'v5' (scipy.io), 'v7.3' (HDF5, h5py) or 'v4/unknown' from the first 128 bytes of a .mat."""
    if len(header) < MAT_HEADER_BYTES or not header.startswith(b"MATLAB"):
        return "v4/unknown"
    endian = header[126:128]
    version = int.from_bytes(header[124:126], "little" if endian == b"IM" else "big")
    if version == 0x0200:
        return "v7.3"
    if version == 0x0100:
        return "v5"
    return "v4/unknown"
