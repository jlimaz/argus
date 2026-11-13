"""Turn the DroneRF dataset (Al-Sa'd et al., 2019, Mendeley Data) into training rows.

DroneRF is 227 recordings of the 2.4 GHz band with three consumer drones (Parrot
Bebop, Parrot AR Drone, DJI Phantom) in several flight modes, plus 41 recordings
of the band with no drone. Two receivers recorded 40 MHz each: file suffix L is
2400-2440 MHz, H is 2440-2480 MHz. Each recording is a CSV of 10 million real
samples at 40 MS/s. The files come in 23 RAR archives, read here without
extraction. Every burst the detector finds in a drone recording becomes a row
labelled `drone_link`; the no-drone recordings are skipped (whatever they hold is
other 2.4 GHz traffic, not a class we can name).

Licence: CC BY 4.0. https://data.mendeley.com/datasets/f4c2b4n755/1
"""

import argparse
import collections
import statistics
from pathlib import Path

import numpy as np
from unrar.cffi import rarfile

from finetune.sources.common import RAW, TRAIN, write_jsonl
from finetune.sources.nist_radar import histogram
from pipeline.features import detect_bursts, real_to_iq

ARCHIVES = RAW / "dronerf"
OUTPUT = TRAIN / "dronerf.jsonl"
SOURCE = "dronerf_mendeley_f4c2b4n755"
LABEL_SERVICE = "drone_link"
SAMPLE_RATE_HZ = 40e6
# Start of the 40 MHz slice each receiver recorded, by file letter.
BAND_START_HZ = {"L": 2400e6, "H": 2440e6}
# BUI code: digit 1 = drone present, digits 2-3 = drone, digits 4-5 = flight mode.
DRONES = {"00": "Parrot Bebop", "01": "Parrot AR Drone", "10": "DJI Phantom"}
MODES = {"00": "on_connected", "01": "hovering", "10": "flying", "11": "video_recording"}
FEATURES = (
    "center_frequency_hz", "bandwidth_hz", "snr_db", "duty_cycle", "burst_duration_s",
    "hopping", "pulse_interval_s", "modulation", "truncated",
)


def archives(pattern: str = "*.rar") -> list[Path]:
    found = sorted(ARCHIVES.glob(pattern))
    if not found:
        raise SystemExit(f"no archives in {ARCHIVES}. Download them from https://data.mendeley.com/datasets/f4c2b4n755/1")
    return found


def parse_name(name: str) -> tuple[str, str, int] | None:
    """'10111H_3.csv' -> (BUI, band letter, segment); None when the name does not fit."""
    stem = Path(name).stem
    if len(stem) < 7 or not stem[:5].isdigit() or stem[5] not in BAND_START_HZ:
        return None
    try:
        return stem[:5], stem[5], int(stem.split("_")[-1])
    except ValueError:
        return None


def segments(limit: int | None = None, pattern: str = "*.rar"):
    """Yield (archive, csv name, BUI, band, segment, real samples) from every archive."""
    count = 0
    for archive in archives(pattern):
        store = rarfile.RarFile(str(archive))
        for info in sorted(store.infolist(), key=lambda i: i.filename):
            parsed = parse_name(info.filename)
            if parsed is None:
                continue
            if limit is not None and count >= limit:
                return
            count += 1
            samples = np.fromstring(store.read(info.filename), sep=",", dtype=np.float32)
            yield archive.name, info.filename, *parsed, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="only process the first N recordings")
    parser.add_argument("--hist", action="store_true", help="print text histograms of the features")
    parser.add_argument("--only", default="*.rar", help="glob of archive names to process, e.g. '*11000*'")
    args = parser.parse_args()

    rows = []
    seen = collections.Counter()
    found = collections.Counter()
    bursts_per = []
    for archive, name, bui, band, segment, samples in segments(args.limit, args.only):
        present, drone, mode = bui[0], bui[1:3], bui[3:5]
        key = "background" if present == "0" else f"{DRONES[drone]} / {MODES[mode]}"
        seen[key] += 1
        if present == "0":
            continue
        iq, sample_rate, offset = real_to_iq(samples, SAMPLE_RATE_HZ)
        bursts = detect_bursts(iq, sample_rate, BAND_START_HZ[band] + offset)
        bursts_per.append(len(bursts))
        found[key] += len(bursts)
        for burst in bursts:
            row = {key: burst[key] for key in FEATURES}
            row["center_frequency_hz"] = round(row["center_frequency_hz"])
            row["bandwidth_hz"] = round(row["bandwidth_hz"])
            row["snr_db"] = round(row["snr_db"], 2)
            row["duty_cycle"] = round(row["duty_cycle"], 5)
            row.update(
                label_service=LABEL_SERVICE,
                label_signal=DRONES[drone],
                flight_mode=MODES[mode],
                band=band,
                source=SOURCE,
                file=f"{archive}/{name}",
                simulated=False,
                frequency_assigned=True,
            )
            rows.append(row)
        print(f"{name:<14} {key:<34} {len(bursts):>3} bursts", flush=True)

    print(f"\n{'recording':<34} {'files':>5} {'bursts':>6}")
    for key in sorted(seen):
        print(f"{key:<34} {seen[key]:>5} {found[key]:>6}")
    if bursts_per:
        print(f"bursts per drone recording: min {min(bursts_per)}, median {statistics.median(bursts_per)}, max {max(bursts_per)}, none in {bursts_per.count(0)}")

    if args.hist and rows:
        report = []
        for key in ("bandwidth_hz", "snr_db", "duty_cycle", "burst_duration_s", "pulse_interval_s"):
            values = [row[key] for row in rows if row[key] is not None]
            if values:
                report.append(histogram(key, values))
        report.append(f"hopping: {sum(r['hopping'] for r in rows)} of {len(rows)}   truncated: {sum(r['truncated'] for r in rows)} of {len(rows)}")
        text = "\n\n".join(report)
        print("\n" + text)
        out = TRAIN.parent.parent / "finetune" / "out" / "dronerf_hist.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")

    if args.limit is None and args.only == "*.rar":
        write_jsonl(OUTPUT, rows)
        print(f"wrote {len(rows)} rows to {OUTPUT}")
    else:
        print(f"--limit or --only set: not writing {OUTPUT}")


if __name__ == "__main__":
    main()
