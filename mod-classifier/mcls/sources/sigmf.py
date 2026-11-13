"""A generic SigMF loader, plus the two datasets that use it (LoRaIQ, occupancy).

SigMF (https://sigmf.org, spec v1.x) stores a headerless `.sigmf-data` file next
to a JSON `.sigmf-meta` file with `global`, `captures` and `annotations`. Read
from the spec (verified against the specification, not a dataset file):

- `global.core:datatype`: `[c|r]<f|i|u><bits>[_le|_be]`, e.g. `cf32_le`,
  `ci16_be`, `ci8`, `cu8`. Only complex types are supported (all datasets here
  are complex); 8-bit types have no byte order. Complex samples are I then Q.
- `global.core:sample_rate`, `global.core:num_channels` (only 1 is supported),
  `global.core:dataset` (a non-conforming data file name; supported if it is a
  plain file next to the meta).
- `captures[i].core:sample_start`, `core:frequency`, `core:header_bytes` (bytes
  of header before that capture's samples, skipped).

One Recording per capture segment: segment i runs from its `sample_start` to the
next capture's (or to the end of the file). Scaling: int types / 2^(bits-1),
unsigned types (x - (2^bits - 1)/2) / ((2^bits - 1)/2) (cu8: 127.5 is zero),
float types as stored.

Split groups: segments of one file share `recording_id` (they are one
continuous capture). Subclasses override `group_id` when files recorded at the
same moment must also share one (LoRaIQ: the four receivers of one measurement).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from mcls.contracts import Recording
from mcls.sources._common import interleaved_reader
from mcls.sources.base import Source, register

DATATYPE = re.compile(r"^(?P<kind>[cr])(?P<fmt>[fiu])(?P<bits>8|16|32|64)(?:_(?P<endian>le|be))?$")


def component_dtype(datatype: str) -> np.dtype:
    """'ci16_be' -> dtype('>i2'), the dtype of one I or Q component."""
    match = DATATYPE.match(datatype)
    if not match:
        raise ValueError(f"unknown SigMF datatype {datatype!r}")
    if match["kind"] != "c":
        raise ValueError(f"SigMF datatype {datatype!r} is real-valued; only complex data is supported")
    fmt, bits, endian = match["fmt"], int(match["bits"]), match["endian"]
    if fmt == "f" and bits not in (32, 64):
        raise ValueError(f"unsupported SigMF float width in {datatype!r}")
    if bits == 8:
        if endian:
            raise ValueError(f"SigMF datatype {datatype!r}: 8-bit types take no byte order")
        return np.dtype(fmt + "1")
    if not endian:
        raise ValueError(f"SigMF datatype {datatype!r} needs _le or _be")
    return np.dtype(("<" if endian == "le" else ">") + fmt + str(bits // 8))


def data_path(meta_path: Path, meta: dict) -> Path:
    dataset = meta.get("global", {}).get("core:dataset")
    if dataset:
        return meta_path.with_name(dataset)
    return meta_path.with_name(meta_path.name[: -len(".sigmf-meta")] + ".sigmf-data")


def segments(meta: dict, data_bytes: int) -> list[dict]:
    """Byte offset and sample count of every capture segment of one file."""
    glob = meta.get("global", {})
    component = component_dtype(glob["core:datatype"])
    channels = int(glob.get("core:num_channels", 1))
    if channels != 1:
        raise ValueError(f"{channels} channels: only single-channel SigMF is supported")
    sample_bytes = 2 * component.itemsize
    captures = sorted(meta.get("captures") or [{"core:sample_start": 0}], key=lambda c: c.get("core:sample_start", 0))
    headers = [int(c.get("core:header_bytes", 0)) for c in captures]
    total_samples = (data_bytes - sum(headers)) // sample_bytes
    out = []
    header_so_far = 0
    for i, capture in enumerate(captures):
        header_so_far += headers[i]
        start = int(capture.get("core:sample_start", 0))
        end = int(captures[i + 1].get("core:sample_start", 0)) if i + 1 < len(captures) else total_samples
        out.append(
            {
                "index": i,
                "sample_start": start,
                "num_samples": end - start,
                "byte_offset": header_so_far + start * sample_bytes,
                "frequency": capture.get("core:frequency"),
                "capture": capture,
            }
        )
    return out


class SigMFSource(Source):
    """Any folder of SigMF recordings. Subclasses set `name` (and maybe `default_frequency_hz`)."""

    #: Used when a capture has no core:frequency.
    default_frequency_hz: float | None = None
    #: Glob for the meta files, relative to root.
    pattern = "**/*.sigmf-meta"

    def meta_files(self) -> list[Path]:
        return sorted(self.root.glob(self.pattern))

    def frequency_hz(self, value: float) -> float:
        """A capture's core:frequency in Hz. SigMF says Hz; a subclass may correct a dataset that is not."""
        return float(value)

    def group_id(self, meta_path: Path) -> str:
        """The split key of a file: its path below root, without the suffix."""
        return meta_path.relative_to(self.root).as_posix()[: -len(".sigmf-meta")].replace("/", "_")

    def recordings(self) -> Iterator[Recording]:
        info = self.taxonomy.dataset(self.name)
        for meta_path in self.meta_files():
            meta = json.loads(meta_path.read_text())
            glob = meta.get("global", {})
            data = data_path(meta_path, meta)
            component = component_dtype(glob["core:datatype"])
            relative = meta_path.relative_to(self.root).as_posix()
            label = self.taxonomy.label_for(self.name, relative)
            if label.excluded:
                continue
            annotations = meta.get("annotations", [])
            for seg in segments(meta, data.stat().st_size):
                if seg["num_samples"] <= 0:
                    continue
                frequency = self.frequency_hz(seg["frequency"]) if seg["frequency"] is not None else self.default_frequency_hz
                if frequency is None:
                    raise ValueError(f"{relative}: capture {seg['index']} has no core:frequency and no default is set")
                lo, hi = seg["sample_start"], seg["sample_start"] + seg["num_samples"]
                inside = [a for a in annotations if lo <= int(a.get("core:sample_start", 0)) < hi]
                recording = Recording(
                    dataset=self.name,
                    recording_id=self.group_id(meta_path),
                    source_file=f"{relative}#{seg['index']}",
                    sample_rate_hz=float(glob["core:sample_rate"]),
                    center_frequency_hz=float(frequency),
                    num_samples=seg["num_samples"],
                    label=label.label,
                    label_strength=label.strength,
                    is_real=info.is_real,
                    license=info.license,
                    read=interleaved_reader(data, component, seg["num_samples"], seg["byte_offset"]),
                    emitter_id=label.emitter_id,
                    meta={
                        "datatype": glob["core:datatype"],
                        "capture_index": seg["index"],
                        "sample_start": seg["sample_start"],
                        "capture": seg["capture"],
                        "annotations": inside,
                        "note": label.note,
                    },
                )
                recording.validate()
                yield recording

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        metas = self.meta_files()
        if not metas:
            return [f"no .sigmf-meta files in {self.root}"]
        for meta_path in metas:
            relative = meta_path.relative_to(self.root).as_posix()
            try:
                meta = json.loads(meta_path.read_text())
                component = component_dtype(meta["global"]["core:datatype"])
            except (ValueError, KeyError) as err:
                problems.append(f"{relative}: bad meta ({err})")
                continue
            data = data_path(meta_path, meta)
            if not data.exists():
                problems.append(f"{relative}: data file {data.name} missing")
                continue
            size = data.stat().st_size
            headers = sum(int(c.get("core:header_bytes", 0)) for c in meta.get("captures", []))
            if (size - headers) % (2 * component.itemsize):
                problems.append(f"{data.name}: {size} bytes is not a whole number of samples (partial download?)")
            starts = [int(c.get("core:sample_start", 0)) for c in meta.get("captures", [])]
            total = (size - headers) // (2 * component.itemsize)
            if starts and max(starts) >= total:
                problems.append(f"{data.name}: a capture starts at sample {max(starts)} but the file holds {total} (partial download?)")
            for a in meta.get("annotations", []):
                end = int(a.get("core:sample_start", 0)) + int(a.get("core:sample_count", 0))
                if end > total:
                    problems.append(f"{data.name}: an annotation ends at sample {end} past the end ({total}); partial download?")
                    break
        return problems


@register
class LoRaIQ(SigMFSource):
    """LoRaIQ (EPFL), https://zenodo.org/records/17708397 (CC BY 4.0).

    Layout from the record page (not a file): `sigmfs/<measurement date-time>/RRH1..RRH4/<n>.sigmf-{data,meta}`
    (the zip itself, read Nov 12, names the receiver folders in lower case, `rrh1`..`rrh4`;
    either case is accepted)
    plus `dataset.csv` describing every frame (sigmf_file, sigmf_file_offset, ...).
    The four RRHs receive the same frames at the same time, so every file of one
    measurement folder shares a recording_id. Datatype and centre come from each
    meta file; 862.5 MHz (the stated transmit frequency) is only the fallback.
    """

    name = "loraiq"
    default_frequency_hz = 862.5e6
    #: Keep every N-th frame file of each receiver folder (MCLS_LORAIQ_STRIDE overrides).
    #: Measured on CRC, Nov 12: a frame file (about 64 ms at 250 kS/s, cf32) gives 64
    #: windows and takes about 0.65 s to prepare, so all ~71,600 would be about 4.6
    #: million windows (~147 GB) and 13 hours. Every 50th keeps ~1,430 frames from
    #: every session and receiver (~92,000 windows, ~3 GB), more than the class cap.
    frame_stride = 50

    def meta_files(self) -> list[Path]:
        import os

        stride = int(os.environ.get("MCLS_LORAIQ_STRIDE", self.frame_stride))
        if stride < 1:
            raise ValueError(f"MCLS_LORAIQ_STRIDE must be 1 or more, got {stride}")
        by_folder: dict[Path, list[Path]] = {}
        for path in super().meta_files():
            by_folder.setdefault(path.parent, []).append(path)
        number = lambda p: (int(p.name.split(".")[0]) if p.name.split(".")[0].isdigit() else 0, p.name)  # noqa: E731
        kept = []
        for folder in sorted(by_folder):
            files = sorted(by_folder[folder], key=number)
            # Only the receiver folders (rrh1..rrh4) of the real layout are thinned.
            kept += files[::stride] if re.fullmatch(r"rrh\d+", folder.name, re.IGNORECASE) else files
        return kept

    def frequency_hz(self, value: float) -> float:
        # The real meta files (read Nov 12) give core:frequency = 862.5: MHz, not the Hz SigMF requires.
        value = float(value)
        return value * 1e6 if value < 1e5 else value

    def group_id(self, meta_path: Path) -> str:
        parts = meta_path.relative_to(self.root).parts
        rrh = next((i for i, p in enumerate(parts) if re.fullmatch(r"rrh\d+", p, re.IGNORECASE)), None)
        if rrh is None or rrh == 0:
            return super().group_id(meta_path)
        return "_".join(parts[:rrh])


@register
class Occupancy(SigMFSource):
    """Multi-environment multi-band occupancy dataset (IEEE DataPort, CC BY 4.0).

    Layout from the dataset page (not a file): `<environment>/iq/<BAND>_t<tile>_wifi0_<UTC>_s<seq>.sigmf-*`,
    ci16_le, 15 or 25 MS/s, about 1.02e6 samples per dwell. Each dwell file is its
    own recording_id (dwells are separate in time).
    """

    name = "occupancy"


if __name__ == "__main__":
    import sys

    source = {"loraiq": LoRaIQ, "occupancy": Occupancy}[sys.argv[1] if len(sys.argv) > 1 else "loraiq"]()
    for problem in source.check():
        print("problem:", problem)
    for rec in source.recordings():
        print(f"{rec.source_file:<60} {rec.recording_id:<30} {rec.band:<9} {rec.num_samples} {rec.label}")
