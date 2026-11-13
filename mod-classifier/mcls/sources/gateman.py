"""GATEMAN GNSS jamming recordings, https://zenodo.org/records/2654322 (CC BY 4.0).

Facts read from the Zenodo record (https://zenodo.org/api/records/2654322), not
from a dataset file (nothing has been downloaded):

- One archive, `rawdata.zip` (48,748,425,742 bytes, md5
  b8ad5a4851eb9ec39784a858c83af378), plus `readBinData.m` at top level.
- Nine scenario folders `<GNSS>+<jammer>`: GNSS one of GALE1, GPSL1, GPSL5;
  jammer one of AMtone, Chirp10MHz, Chirp20MHz. Each holds a GNSS file
  (`GALE1@20MSps.bin`, `GPSL1@20MSps.bin` or `GPSL5@20MSps.bin`) and a jamming
  file (`AMtone@20MSps.bin`, `Chirp10MHz@20MSps.bin` or `Chirp20MHz@20MSps.bin`).
- "Each file stores the received baseband samples as an array of complex,
  16-bit signed integer data"; "The data is stored in Big-endian, network order
  format"; interleaved I, Q, I, Q. So each sample is 4 bytes, `>i2` I then Q.
- "Both signals of a scenario were recorded synchronously with an USRP at an
  I\\Q rate of 20 MS/s."

The loader reads the EXTRACTED layout: it searches below `root` for
`<scenario>/<signal>@20MSps-16bit.bin` (the names inside the real zip, read Nov 12;
the shorter `<signal>@20MSps.bin` of the record page is accepted too) with rglob,
so a top folder inside the zip
(e.g. `rawdata/`) does not matter. It does not read the zip itself.

INFERRED, not stated by the record:

- Centre frequency. The record gives none. The loader uses the nominal carrier
  of the band named in the scenario folder: GPS L1 and Galileo E1 1575.42 MHz,
  GPS L5 1176.45 MHz. Every Recording says so in `meta["center_frequency_source"]`.
- Scaling: int16 / 32768 (the project convention for signed integers).

Split key: the GNSS and jamming files of one scenario were recorded at the same
time, so both share `recording_id` = the scenario folder name (e.g.
`GPSL1+AMtone`). Labels come from the taxonomy, matched against the path below
root (e.g. `rawdata/GPSL1+AMtone/AMtone@20MSps.bin`). Files the taxonomy
excludes (at present the GNSS-only files) are skipped; `check()` still expects
both files of every scenario, since a missing one means a partial extract.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from mcls.contracts import Recording
from mcls.sources._common import interleaved_reader
from mcls.sources.base import Source, register

SAMPLE_RATE_HZ = 20e6
COMPONENT = ">i2"  # big-endian int16, stated by the record
BYTES_PER_SAMPLE = 4  # int16 I + int16 Q

#: Nominal carrier per GNSS band name. INFERRED from the band name; the record states no centre frequency.
NOMINAL_CENTRE_HZ = {"GPSL1": 1575.42e6, "GALE1": 1575.42e6, "GPSL5": 1176.45e6}
JAMMERS = ("AMtone", "Chirp10MHz", "Chirp20MHz")
SCENARIOS = tuple(f"{g}+{j}" for g in ("GALE1", "GPSL1", "GPSL5") for j in JAMMERS)

SCENARIO = re.compile(r"^(?P<gnss>GALE1|GPSL1|GPSL5)\+(?P<jammer>AMtone|Chirp10MHz|Chirp20MHz)$")
FILE = re.compile(r"^(?P<signal>GALE1|GPSL1|GPSL5|AMtone|Chirp10MHz|Chirp20MHz)@20MSps(?:-16bit)?\.bin$")
CENTRE_NOTE = "inferred from the GNSS band in the scenario folder name; the dataset does not state a centre frequency"


def parse(path: Path) -> dict | None:
    """'.../GPSL1+AMtone/AMtone@20MSps.bin' -> scenario, gnss, jammer, role; None when it does not fit."""
    folder, name = SCENARIO.match(path.parent.name), FILE.match(path.name)
    if not folder or not name:
        return None
    signal = name["signal"]
    if signal not in (folder["gnss"], folder["jammer"]):
        return None
    return {
        "scenario": path.parent.name,
        "gnss": folder["gnss"],
        "jammer": folder["jammer"],
        "signal": signal,
        "role": "gnss" if signal == folder["gnss"] else "jammer",
    }


@register
class Gateman(Source):
    name = "gateman"

    def files(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*@20MSps*.bin") if parse(p))

    def recordings(self) -> Iterator[Recording]:
        info = self.taxonomy.dataset(self.name)
        for path in self.files():
            parsed = parse(path)
            size = path.stat().st_size
            if size % BYTES_PER_SAMPLE:
                raise ValueError(f"{path.name}: {size} bytes is not a whole number of int16 I/Q pairs")
            num_samples = size // BYTES_PER_SAMPLE
            relative = path.relative_to(self.root).as_posix()
            label = self.taxonomy.label_for(self.name, relative)
            if label.excluded:
                continue
            recording = Recording(
                dataset=self.name,
                recording_id=parsed["scenario"],
                source_file=relative,
                sample_rate_hz=SAMPLE_RATE_HZ,
                center_frequency_hz=NOMINAL_CENTRE_HZ[parsed["gnss"]],
                num_samples=num_samples,
                label=label.label,
                label_strength=label.strength,
                is_real=info.is_real,
                license=info.license,
                read=interleaved_reader(path, COMPONENT, num_samples),
                emitter_id=label.emitter_id,
                meta={
                    "scenario": parsed["scenario"],
                    "gnss_band": parsed["gnss"],
                    "jammer": parsed["jammer"],
                    "role": parsed["role"],
                    "center_frequency_source": CENTRE_NOTE,
                    "note": label.note,
                },
            )
            recording.validate()
            yield recording

    def check(self) -> list[str]:
        problems = super().check()
        if problems:
            return problems
        files = self.files()
        if not files:
            zipped = list(self.root.rglob("rawdata.zip"))
            hint = " (rawdata.zip is there: extract it first)" if zipped else ""
            return [f"no GATEMAN <scenario>/<signal>@20MSps.bin files in {self.root}{hint}"]
        found: dict[str, set[str]] = {}
        for path in files:
            parsed = parse(path)
            found.setdefault(parsed["scenario"], set()).add(parsed["role"])
            size = path.stat().st_size
            if size == 0 or size % BYTES_PER_SAMPLE:
                problems.append(
                    f"{path.relative_to(self.root).as_posix()}: {size} bytes is not a positive multiple of "
                    f"{BYTES_PER_SAMPLE} (partial download?)"
                )
        for scenario in SCENARIOS:
            roles = found.get(scenario, set())
            for role in ("gnss", "jammer"):
                if role not in roles:
                    problems.append(f"{scenario}: {role} file missing")
        return problems


if __name__ == "__main__":
    source = Gateman()
    for problem in source.check():
        print("problem:", problem)
    for rec in source.recordings():
        print(f"{rec.source_file:<50} {rec.recording_id:<20} {rec.band:<9} {rec.duration_s:8.2f} s  {rec.label}")
