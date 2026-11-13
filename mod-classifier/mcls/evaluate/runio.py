"""Read a run folder written to the run-output contract.

    runs/<name>/
      meta.json                 {"git_commit", "manifest_sha256", "seed", "classes", "model", "created"}
      predictions_<split>.npz   window_id (str), recording_id (str), label (int64),
                                logits (float32, N x K), snr_db_est (float32, NaN if unknown)

Nothing here trains or loads a model: evaluation works from saved logits only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mcls import manifest as manifest_mod
from mcls.contracts import ManifestRow

PREDICTION_KEYS = ("window_id", "recording_id", "label", "logits", "snr_db_est")


@dataclass
class Predictions:
    split: str
    window_id: np.ndarray
    recording_id: np.ndarray
    label: np.ndarray
    logits: np.ndarray
    snr_db_est: np.ndarray

    def __len__(self) -> int:
        return len(self.label)

    def subset(self, mask: np.ndarray) -> "Predictions":
        return Predictions(self.split, self.window_id[mask], self.recording_id[mask], self.label[mask],
                           self.logits[mask], self.snr_db_est[mask])


def load_meta(run: Path) -> dict[str, Any]:
    path = run / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"{run}: meta.json missing (run-output contract)")
    meta = json.loads(path.read_text())
    if not isinstance(meta.get("classes"), list) or not meta["classes"]:
        raise ValueError(f"{path}: 'classes' must be a non-empty list")
    return meta


def load_predictions(path: Path, num_classes: int) -> Predictions:
    """Load one predictions_<split>.npz and check it against the contract."""
    split = path.stem.removeprefix("predictions_")
    with np.load(path, allow_pickle=False) as data:
        missing = [k for k in PREDICTION_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{path.name}: missing arrays {missing}")
        arrays = {k: data[k] for k in PREDICTION_KEYS}
    n = len(arrays["label"])
    logits = arrays["logits"]
    if logits.ndim != 2 or logits.shape != (n, num_classes):
        raise ValueError(f"{path.name}: logits shape {logits.shape}, expected ({n}, {num_classes})")
    for key in ("window_id", "recording_id", "snr_db_est"):
        if arrays[key].shape != (n,):
            raise ValueError(f"{path.name}: {key} shape {arrays[key].shape}, expected ({n},)")
    if arrays["window_id"].dtype.kind not in "US" or arrays["recording_id"].dtype.kind not in "US":
        raise ValueError(f"{path.name}: window_id and recording_id must be string arrays")
    label = arrays["label"]
    if label.dtype.kind not in "iu":
        raise ValueError(f"{path.name}: label must be integer, got {label.dtype}")
    if n and (label.min() < 0 or label.max() >= num_classes):
        raise ValueError(f"{path.name}: label outside 0..{num_classes - 1}")
    if not np.isfinite(logits).all():
        raise ValueError(f"{path.name}: logits contain NaN or infinity")
    if len(np.unique(arrays["window_id"])) != n:
        raise ValueError(f"{path.name}: duplicate window_id")
    return Predictions(split, arrays["window_id"].astype(str), arrays["recording_id"].astype(str),
                       label.astype(np.int64), logits.astype(np.float64), arrays["snr_db_est"].astype(np.float64))


def prediction_files(run: Path) -> dict[str, Path]:
    return {p.stem.removeprefix("predictions_"): p for p in sorted(run.glob("predictions_*.npz"))}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hashes_match(a: str | None, b: str | None) -> bool:
    """Compare a full sha256 with a full or short one (mcls.manifest.digest gives 16 hex)."""
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    short = min(len(a), len(b))
    return short >= 16 and a[:short] == b[:short]


def load_manifest(path: Path, meta: dict[str, Any]) -> tuple[dict[str, ManifestRow], dict[str, Any]]:
    """Read the manifest and check it is the one the run was trained on.

    Raises ValueError when meta.json names a different manifest hash.
    """
    sha = file_sha256(path)
    expected = meta.get("manifest_sha256")
    if expected and not hashes_match(sha, expected):
        raise ValueError(f"{path}: sha256 {sha[:16]} does not match the run's manifest_sha256 {str(expected)[:16]}")
    rows = manifest_mod.read(path)
    info: dict[str, Any] = {"path": str(path), "sha256": sha, "matches_meta": bool(expected), "rows": len(rows)}
    train = [r for r in rows if r.split == "train"]
    info["train_rows"] = {"real": sum(r.is_real for r in train), "synthetic": sum(not r.is_real for r in train)}
    try:
        manifest_mod.check_leakage(rows)
        info["leakage"] = {"ok": True, "error": None, "checks": "recording_id and window_id across splits"}
    except ValueError as exc:
        info["leakage"] = {"ok": False, "error": str(exc), "checks": "recording_id and window_id across splits"}
    return {r.window_id: r for r in rows}, info


def burst_keys(rows: list[ManifestRow]) -> tuple[dict[str, str], dict[str, int]]:
    """Map window_id -> burst key, and count how each window was grouped.

    A row with `burst_id` is grouped by (dataset, recording_id, burst_id).
    Only rows whose burst_id is None fall back to inference: the front end
    (Engineer 4) writes a burst's windows consecutively in its shard and gives
    each the stats of that burst (snr_db_est, bandwidth_hz, duty_cycle), so a
    burst is a run of consecutive windows (by offset) in one shard with
    identical stats. Windows with no stats (idle background) are one-window
    groups. Two adjacent bursts with bit-identical stats would merge; that is
    the known limit of the fallback.
    """
    keys: dict[str, str] = {}
    counts = {"burst_id": 0, "inferred_equal_stats": 0, "single_window_no_stats": 0}
    groups: dict[tuple[str, str, str], list[ManifestRow]] = {}
    for row in rows:
        if row.burst_id is not None:
            keys[row.window_id] = f"{row.dataset}|{row.recording_id}|id:{row.burst_id}"
            counts["burst_id"] += 1
        else:
            groups.setdefault((row.dataset, row.recording_id, row.shard), []).append(row)
    for (dataset, recording, shard), members in groups.items():
        members.sort(key=lambda r: r.offset)
        count, previous = -1, None
        for row in members:
            stats = (row.snr_db_est, row.bandwidth_hz, row.duty_cycle)
            empty = all(v is None for v in stats)
            if empty or stats != previous:
                count += 1
            previous = None if empty else stats
            counts["single_window_no_stats" if empty else "inferred_equal_stats"] += 1
            keys[row.window_id] = f"{dataset}|{recording}|{shard}|{count}"
    return keys, counts
