"""DroneRF (Al-Sa'd et al., 2019), https://data.mendeley.com/datasets/f4c2b4n755/1 (CC BY 4.0).

Three drones (Parrot Bebop, Parrot AR Drone, DJI Phantom) in several flight modes,
plus recordings with no drone. Two receivers each recorded 40 MHz of the 2.4 GHz
band as **real** samples at 40 MS/s: file letter `L` is 2400-2440 MHz, `H` is
2440-2480 MHz. Each recording segment is one CSV of comma-separated real floats,
shipped inside 23 RAR archives and read here without extraction (`unrar.cffi`).
Extracted CSVs placed directly in the root are read too.

File names: `<BUI><L|H>_<segment>.csv`, e.g. `10111H_3.csv`. The 5-digit BUI code:
digit 1 = drone present, digits 2-3 = drone, digits 4-5 = flight mode (`00000`
means no drone).

Each CSV becomes one Recording at **20 MS/s complex baseband** after
`pipeline.features.real_to_iq` (analytic signal, shift by fs/4, decimate by 2,
EDGE_TRIM samples dropped at each end), centred at band start + fs/4: 2410 MHz (L) or 2450 MHz (H), as
`real_to_iq` and the repo loader define it. Note: a real stream at 40 MS/s spans
20 MHz, not the 40 MHz the L/H band names suggest; the RF mapping is unverified.
Split key: `recording_id` = the BUI (drone + flight mode), so every segment of
one BUI, L and H, lands in one split. Why not one id per segment: the data paper
(Al-Sa'd et al., Data in Brief 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC6727013/,
read Nov 12) does not say whether segments are independent captures. It says
"The records are 10.25 seconds of RF background activities and approximately
5.25 seconds of RF drone communications for each flight mode", stored as
"RF data segments" by a LabVIEW program started once per mode ("Start the
LabVIEW programs to fetch, process and store RF data segments"). Inference
(not stated): with 1e7 real samples per file at 40 MS/s (0.25 s), 41 no-drone
segments are 10.25 s and the Bebop's 84 segments over 4 modes are 21 x 0.25 s
= 5.25 s per mode, so the segments of one BUI look like consecutive slices of
one session, and splitting them would leak (PLAN 3.5). The L and H files of one
segment are the two receivers at the same moment and share the id too.

Exception, the no-drone session (`00000`, 41 segments, 10.25 s): it is the only
real background data, and one id would put all of it in one split, so background
could never be validated or tested. It is cut into time blocks of
`BACKGROUND_BLOCK` consecutive segments (`00000_blk0` ... `00000_blk4`), and the
last segment of each block but the final one is dropped as a 0.25 s gap, so no
two blocks touch in time. Adjacent blocks still come from the same session and
room, so a background score measures new moments of one lab, not a new place.

Labels and emitter ids come from the taxonomy (the drones' links are Wi-Fi);
files the taxonomy excludes, including names with an unknown BUI, are skipped.
The parsed drone name is only the fallback emitter id.

Lazy reading and its cost: a CSV inside a RAR cannot be read from an offset, so
the first `read()` of a recording decompresses and parses the whole CSV (about
10 million values) and converts it; the converted IQ (about 40 MB complex64) is
cached for that recording until another recording of the same Source is read
(one entry, so memory stays bounded). Read recordings one at a time, in order.

Verified vs assumed:
- Verified (the repo's own `finetune/sources/dronerf.py`, written against the
  files): CSV of real values inside RAR, 40 MS/s, L/H band starts, the BUI code
  and the name pattern.
- Assumed: every CSV holds `REAL_SAMPLES_PER_FILE` = 10,000,000 values (from the
  same docstring, not counted here). `num_samples` is computed from it and the
  first `read()` raises if a file differs. Override per instance if needed.
- Assumed: the CSV values are already in about -1..1 (receiver float output), so
  no scaling is applied. The front end normalises each window anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from mcls.contracts import SAMPLE_DTYPE, Recording
from mcls.sources._common import check_range, import_pipeline_features
from mcls.sources.base import Source, register

REAL_RATE_HZ = 40e6
#: Start of the 40 MHz slice each receiver recorded, by file letter.
BAND_START_HZ = {"L": 2400e6, "H": 2440e6}
#: Model names from the paper's references (Parrot Bebop, Parrot AR.Drone 2.0, DJI Phantom 3 Standard).
DRONES = {"00": "Parrot_Bebop", "01": "Parrot_AR_Drone_2", "10": "DJI_Phantom_3_Standard"}
MODES = {"00": "on_connected", "01": "hovering", "10": "flying", "11": "video_recording"}
REAL_SAMPLES_PER_FILE = 10_000_000
EXPECTED_ARCHIVES = 23


def parse_name(name: str) -> dict | None:
    """'10111H_3.csv' -> bui, band letter, segment; None when the name does not fit."""
    path = Path(name.replace("\\", "/"))
    if path.suffix.lower() != ".csv":
        return None
    stem = path.stem
    if len(stem) < 8 or not stem[:5].isdigit() or stem[5] not in BAND_START_HZ or stem[6] != "_":
        return None
    try:
        segment = int(stem[7:])
    except ValueError:
        return None
    bui = stem[:5]
    present = bui[0] == "1"
    return {
        "bui": bui,
        "band": stem[5],
        "segment": segment,
        "drone_present": present,
        "drone": DRONES.get(bui[1:3]) if present else None,
        "mode": MODES.get(bui[3:5]) if present else None,
    }


def parse_csv(data: bytes) -> np.ndarray:
    """Comma- or newline-separated real values -> float32 array."""
    text = data.replace(b"\r", b"").replace(b"\n", b",").strip(b", ")
    return np.array(np.fromstring(text, sep=",", dtype=np.float64), dtype=np.float32)


def iq_length(real_samples: int) -> int:
    """Length of `real_to_iq`'s output for `real_samples` real inputs."""
    edge = import_pipeline_features().EDGE_TRIM
    half = (real_samples + 1) // 2
    return half - 2 * edge if half > 2 * edge else half


def open_rar(path: Path):
    try:
        from unrar.cffi import rarfile
    except ImportError as err:  # pragma: no cover - depends on the venv
        raise ImportError("DroneRF archives need unrar-cffi: pip install unrar-cffi") from err
    return rarfile.RarFile(str(path))



#: No-drone segments per background block; see the module docstring.
BACKGROUND_BUI = "00000"
BACKGROUND_BLOCK = 8
BACKGROUND_BLOCKS = 5  # segments 32-40 form the last block (no gap after it)


def recording_id_for(parsed: dict) -> str | None:
    """The split key of one file: the BUI, or a time block of the no-drone session; None for a gap segment."""
    if parsed["bui"] != BACKGROUND_BUI:
        return parsed["bui"]
    block = min(parsed["segment"] // BACKGROUND_BLOCK, BACKGROUND_BLOCKS - 1)
    is_gap = parsed["segment"] % BACKGROUND_BLOCK == BACKGROUND_BLOCK - 1 and block < BACKGROUND_BLOCKS - 1
    return None if is_gap else f"{BACKGROUND_BUI}_blk{block}"


@register
class DroneRF(Source):
    name = "dronerf"
    real_samples_per_file = REAL_SAMPLES_PER_FILE

    #: Opens a RAR archive; returns an object with infolist() (items with .filename) and read(name).
    open_archive: Callable = staticmethod(open_rar)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache: tuple[str, np.ndarray] | None = None

    # -- discovery ---------------------------------------------------------

    def members(self) -> list[tuple[Path, str | None, str]]:
        """(archive or CSV path, member name inside the archive or None, CSV file name)."""
        found = []
        for archive in sorted(self.root.glob("*.rar")):
            store = self.open_archive(archive)
            for info in sorted(store.infolist(), key=lambda i: i.filename):
                if parse_name(info.filename):
                    found.append((archive, info.filename, Path(info.filename.replace("\\", "/")).name))
        for csv in sorted(self.root.glob("*.csv")):
            if parse_name(csv.name):
                found.append((csv, None, csv.name))
        return found

    # -- reading -----------------------------------------------------------

    def _load_real(self, path: Path, member: str | None) -> np.ndarray:
        data = path.read_bytes() if member is None else self.open_archive(path).read(member)
        return parse_csv(data)

    def _make_reader(self, key: str, path: Path, member: str | None, num_samples: int):
        def read(start: int, count: int) -> np.ndarray:
            check_range(key, start, count, num_samples)
            if self._cache is None or self._cache[0] != key:
                self._cache = None  # free the previous recording first
                real = self._load_real(path, member)
                if len(real) != self.real_samples_per_file:
                    raise ValueError(f"{key}: {len(real)} real samples, expected {self.real_samples_per_file}")
                iq, _, _ = import_pipeline_features().real_to_iq(real, REAL_RATE_HZ)
                self._cache = (key, np.ascontiguousarray(iq, dtype=SAMPLE_DTYPE))
            return self._cache[1][start : start + count].copy()

        return read

    def recordings(self) -> Iterator[Recording]:
        info = self.taxonomy.dataset(self.name)
        num_samples = iq_length(self.real_samples_per_file)
        for path, member, file_name in self.members():
            parsed = parse_name(file_name)
            label = self.taxonomy.label_for(self.name, file_name)
            if label.excluded:
                continue
            key = f"{path.name}/{member}" if member else path.name
            recording_id = recording_id_for(parsed)
            if recording_id is None:  # a gap segment between background blocks
                continue
            recording = Recording(
                dataset=self.name,
                recording_id=recording_id,
                source_file=key,
                sample_rate_hz=REAL_RATE_HZ / 2,
                center_frequency_hz=BAND_START_HZ[parsed["band"]] + REAL_RATE_HZ / 4,
                num_samples=num_samples,
                label=label.label,
                label_strength=label.strength,
                is_real=info.is_real,
                license=info.license,
                read=self._make_reader(key, path, member, num_samples),
                emitter_id=label.emitter_id or parsed["drone"],
                meta={
                    "bui": parsed["bui"],
                    "band_letter": parsed["band"],
                    "segment": parsed["segment"],
                    "flight_mode": parsed["mode"],
                    "converted_from_real": True,
                    "note": label.note,
                },
            )
            recording.validate()
            yield recording

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        archives = sorted(self.root.glob("*.rar"))
        if archives and len(archives) != EXPECTED_ARCHIVES:
            problems.append(f"{len(archives)} RAR archives, expected {EXPECTED_ARCHIVES} (partial download?)")
        names: set[str] = set()
        for archive in archives:
            try:
                store = self.open_archive(archive)
            except Exception as err:  # unrar raises its own error types
                problems.append(f"{archive.name}: cannot open ({err}); partial download?")
                continue
            for info in store.infolist():
                base = Path(info.filename.replace("\\", "/")).name
                if base.lower().endswith(".csv"):
                    if not parse_name(base):
                        problems.append(f"{archive.name}/{info.filename}: name does not match the DroneRF pattern")
                    names.add(base)
                    size = getattr(info, "file_size", None)
                    if size is not None and size < self.real_samples_per_file:
                        problems.append(f"{archive.name}/{info.filename}: only {size} bytes uncompressed")
        names |= {p.name for p in self.root.glob("*.csv") if parse_name(p.name)}
        if not names:
            problems.append(f"no DroneRF CSVs or RAR archives in {self.root}")
        for name in sorted(names):
            parsed = parse_name(name)
            if parsed:
                other = "H" if parsed["band"] == "L" else "L"
                twin = f"{parsed['bui']}{other}_{parsed['segment']}.csv"
                if twin not in names:
                    problems.append(f"{name}: its other half {twin} is missing")
        return problems


if __name__ == "__main__":
    source = DroneRF()
    for problem in source.check():
        print("problem:", problem)
    for rec in source.recordings():
        print(f"{rec.source_file:<40} {rec.recording_id:<12} {rec.band:<9} {rec.label} ({rec.label_strength})")
