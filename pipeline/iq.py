"""Read IQ recordings and find the bursts in them.

Three kinds of file:

- `.cu8`: what `rtl_sdr` writes. Interleaved unsigned 8-bit I and Q.
- `.cs8`: what `hackrf_transfer` writes. Interleaved signed 8-bit I and Q.
- SigMF: a `.sigmf-data` file of samples with a `.sigmf-meta` JSON file beside it
  that gives the sample format, the sample rate and the centre frequency.

A raw `.cu8` or `.cs8` file carries no sample rate or centre frequency, so they are
passed in. A recording can be gigabytes, so it is memory-mapped and read in
chunks; each chunk goes through `pipeline.features.detect_bursts`, the same code
that measures the training rows. A burst that runs across a chunk boundary is
reported in both chunks and marked `truncated`.
"""

import json
from pathlib import Path

import numpy as np

from pipeline.features import detect_bursts

# About 0.2 s at 20 MS/s. Longer than the 20 ms gap that separates bursts, short
# enough that one chunk and its working arrays fit comfortably in memory.
CHUNK_SAMPLES = 1 << 22

# The fields every burst row carries, as in the training rows (finetune/sources).
FEATURES = (
    "center_frequency_hz", "bandwidth_hz", "snr_db", "duty_cycle", "burst_duration_s",
    "hopping", "pulse_interval_s", "modulation", "truncated",
)

# Sample format: (numpy type of one I or Q value, offset, scale). Values become
# (raw - offset) / scale, so every format lands in roughly -1 to 1.
FORMATS = {
    "cu8": (np.uint8, 127.5, 127.5),
    "ci8": (np.int8, 0.0, 128.0),
    "ci16_le": (np.dtype("<i2"), 0.0, 32768.0),
    "cf32_le": (np.dtype("<f4"), 0.0, 1.0),
}
# File suffix: SigMF datatype name.
SUFFIXES = {".cu8": "cu8", ".cs8": "ci8", ".ci8": "ci8", ".cs16": "ci16_le", ".ci16": "ci16_le", ".cf32": "cf32_le", ".fc32": "cf32_le"}


class IQError(ValueError):
    """A recording that cannot be read, and why."""


def datatype_for(path: Path) -> str:
    """The sample format a raw file's suffix names."""
    suffix = path.suffix.lower()
    if suffix not in SUFFIXES:
        raise IQError(
            f"{path.name}: the suffix {suffix or '(none)'} does not say the sample format. "
            f"Use one of {', '.join(sorted(SUFFIXES))}, or pass the format."
        )
    return SUFFIXES[suffix]


def open_samples(path: Path, datatype: str) -> np.ndarray:
    """Memory-map a file of interleaved I and Q values (one value per entry)."""
    if datatype not in FORMATS:
        raise IQError(
            f"{path.name}: the sample format {datatype!r} is not supported. "
            f"Supported: {', '.join(FORMATS)}."
        )
    kind = np.dtype(FORMATS[datatype][0])
    size = path.stat().st_size
    values = size // kind.itemsize
    if values < 2:
        raise IQError(f"{path.name} has no samples.")
    # An odd trailing value (a cut-off recording) is dropped.
    return np.memmap(path, dtype=kind, mode="r", shape=(values - values % 2,))


def to_complex(values: np.ndarray, datatype: str) -> np.ndarray:
    """Interleaved I, Q values to complex64 samples scaled to about -1..1."""
    _, offset, scale = FORMATS[datatype]
    floats = (np.asarray(values, dtype=np.float32) - offset) / scale
    return (floats[0::2] + 1j * floats[1::2]).astype(np.complex64)


def _bursts(values: np.ndarray, datatype: str, sample_rate_hz: float, center_frequency_hz: float,
            first_sample: int, chunk_samples: int) -> list[dict]:
    """Bursts in one stretch of interleaved values, with start times from the file start."""
    rows = []
    samples = len(values) // 2
    for start in range(0, samples, chunk_samples):
        iq = to_complex(values[2 * start : 2 * min(start + chunk_samples, samples)], datatype)
        for burst in detect_bursts(iq, sample_rate_hz, center_frequency_hz):
            row = {key: burst[key] for key in FEATURES}
            row["center_frequency_hz"] = round(row["center_frequency_hz"])
            row["bandwidth_hz"] = round(row["bandwidth_hz"])
            row["snr_db"] = round(row["snr_db"], 2)
            row["duty_cycle"] = round(row["duty_cycle"], 5)
            row["start_s"] = round((first_sample + start) / sample_rate_hz + burst["start_s"], 6)
            rows.append(row)
    return rows


def read_raw(path: Path, sample_rate_hz: float, center_frequency_hz: float,
             datatype: str | None = None, chunk_samples: int = CHUNK_SAMPLES) -> list[dict]:
    """Bursts in a raw .cu8 / .cs8 (or .ci16, .cf32) recording."""
    path = Path(path)
    if sample_rate_hz is None or sample_rate_hz <= 0:
        raise IQError(f"{path.name}: a raw recording needs its sample rate.")
    if center_frequency_hz is None or center_frequency_hz < 0:
        raise IQError(f"{path.name}: a raw recording needs the centre frequency it was tuned to.")
    datatype = datatype or datatype_for(path)
    values = open_samples(path, datatype)
    rows = _bursts(values, datatype, sample_rate_hz, center_frequency_hz, 0, chunk_samples)
    for row in rows:
        row["source"] = path.name
    return rows


def sigmf_paths(path: Path) -> tuple[Path, Path]:
    """(meta, data) for a .sigmf-meta, a .sigmf-data, or the name they share."""
    path = Path(path)
    base = path.with_suffix("") if path.suffix in (".sigmf-meta", ".sigmf-data", ".sigmf") else path
    # Appended, not swapped in: a base name such as "drone.2437M" keeps its dot.
    meta, data = (base.parent / f"{base.name}{suffix}" for suffix in (".sigmf-meta", ".sigmf-data"))
    for needed in (meta, data):
        if not needed.is_file():
            raise IQError(f"{needed.name} is missing. A SigMF recording is a .sigmf-meta and a .sigmf-data file.")
    return meta, data


def read_sigmf(path: Path, chunk_samples: int = CHUNK_SAMPLES) -> list[dict]:
    """Bursts in a SigMF recording, using the frequency of each capture segment.

    Only single-channel, little-endian complex recordings are read. Annotations
    are not used yet.
    """
    meta_path, data_path = sigmf_paths(path)
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise IQError(f"{meta_path.name} is not valid JSON at line {exc.lineno}: {exc.msg}.") from exc
    info = meta.get("global") or {}
    datatype = info.get("core:datatype")
    if not isinstance(datatype, str):
        raise IQError(f"{meta_path.name} has no core:datatype.")
    datatype = {"ci8_le": "ci8", "cu8_le": "cu8"}.get(datatype, datatype)
    if datatype.startswith("r"):
        raise IQError(f"{meta_path.name}: real-valued recordings ({datatype}) are not supported yet.")
    if (info.get("core:num_channels") or 1) != 1:
        raise IQError(f"{meta_path.name}: only single-channel recordings are supported.")
    sample_rate = info.get("core:sample_rate")
    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise IQError(f"{meta_path.name} has no core:sample_rate.")
    captures = sorted(meta.get("captures") or [], key=lambda c: c.get("core:sample_start", 0))
    if not captures:
        raise IQError(f"{meta_path.name} has no captures, so the centre frequency is unknown.")
    if any(capture.get("core:header_bytes") for capture in captures):
        raise IQError(f"{meta_path.name}: captures with core:header_bytes are not supported.")

    values = open_samples(data_path, datatype)
    total = len(values) // 2
    rows = []
    for index, capture in enumerate(captures):
        frequency = capture.get("core:frequency")
        if not isinstance(frequency, (int, float)):
            raise IQError(f"{meta_path.name}: capture {index + 1} has no core:frequency.")
        start = int(capture.get("core:sample_start", 0))
        end = int(captures[index + 1].get("core:sample_start", total)) if index + 1 < len(captures) else total
        end = min(end, total)
        if end <= start:
            continue
        segment = values[2 * start : 2 * end]
        for row in _bursts(segment, datatype, float(sample_rate), float(frequency), start, chunk_samples):
            row["source"] = data_path.name
            if capture.get("core:datetime"):
                row["capture_datetime"] = capture["core:datetime"]
            rows.append(row)
    return rows
