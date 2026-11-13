"""Shared pieces for turning an open source into training observations."""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TRAIN = ROOT / "data" / "train"

# How an observation differs from the catalogue entry: bandwidth is the listed value
# times roughly N(1, 0.1), and one modulation in five is reported as unknown. The
# figures were measured on an earlier set of simulated observations.
BANDWIDTH_SPREAD = 0.1
UNKNOWN_MODULATION_RATE = 0.2
SNR_DB_RANGE = (6.0, 35.0)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        row["id"] = index
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def observe(rng: random.Random, frequencies: list[int], bandwidths: list[int], modulations: list[str]) -> dict:
    """Draw one observation of a catalogued signal."""
    low, high = min(frequencies), max(frequencies)
    bandwidth = rng.choice(bandwidths) * rng.gauss(1.0, BANDWIDTH_SPREAD)
    modulation = rng.choice(modulations)
    if rng.random() < UNKNOWN_MODULATION_RATE:
        modulation = "unknown"
    return {
        "center_frequency_hz": round(rng.uniform(low, high)),
        "bandwidth_hz": max(1, round(bandwidth)),
        "modulation": modulation,
        "snr_db": round(rng.uniform(*SNR_DB_RANGE), 1),
    }
