"""The run folder (orchestrator's run-output contract, PLAN.md 5.3). No torch.

    runs/<name>/
      config.toml               the effective run config (after command-line overrides)
      meta.json                 {"git_commit", "manifest_sha256", "seed", "classes", "model", "created"}
      metrics.json              written by the evaluator, never here
      predictions_<split>.npz   window_id (str), recording_id (str), label (int64), logits (float32 N x K),
                                snr_db_est (float32, NaN if unknown)
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import subprocess
import tomllib
from pathlib import Path

import numpy as np

META_KEYS = ("git_commit", "manifest_sha256", "seed", "classes", "model", "created")
PREDICTION_KEYS = ("window_id", "recording_id", "label", "logits", "snr_db_est")


def apply_overrides(config: dict, seed: int | None = None, device: str | None = None,
                    out_dir: str | None = None, name: str | None = None) -> dict:
    """A copy of `config` with command-line overrides written into [run].

    A seed override also suffixes the run name with `_s<seed>`, so the 3 seeds
    of one config land in 3 folders.
    """
    config = copy.deepcopy(config)
    run = config["run"]
    if name is not None:
        run["name"] = name
    if seed is not None:
        run["seed"] = seed
        run["name"] = f"{run['name']}_s{seed}"
    if device is not None:
        run["device"] = device
    if out_dir is not None:
        run["out_dir"] = out_dir
    return config


def run_dir_for(config: dict) -> Path:
    return Path(config["run"]["out_dir"]) / config["run"]["name"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(cwd: Path | None = None) -> str:
    """HEAD's hash, or "unknown" outside a git checkout.

    Same rule as mcls.baselines.run.git_commit, so meta.json is identical
    whichever of the two writes it (Engineer 6 checks an existing meta.json).
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return head.stdout.strip() if head.returncode == 0 else "unknown"


def class_list(config: dict) -> list[str]:
    from mcls import taxonomy

    return list(taxonomy.load().classes(config["data"]["classes"]))


def build_meta(config: dict, classes: list[str]) -> dict:
    manifest = Path(config["data"]["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    return {
        "git_commit": git_commit(Path(__file__).resolve().parent),
        "manifest_sha256": sha256_file(manifest),
        "seed": int(config["run"]["seed"]),
        "classes": list(classes),
        "model": config["model"]["name"],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def prepare(config: dict, run_dir: Path, classes: list[str], resume: bool = False) -> dict:
    """Create the run folder and write config.toml and meta.json; return meta.

    A fresh run refuses a folder that already holds meta.json (no silent
    overwrite). A resumed run requires the folder and the same config, and
    keeps its original meta.json.
    """
    config_path, meta_path = run_dir / "config.toml", run_dir / "meta.json"
    if resume:
        if not meta_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"cannot resume: {run_dir} has no config.toml and meta.json")
        with open(config_path, "rb") as handle:
            stored = tomllib.load(handle)
        if stored != config:
            raise ValueError(f"cannot resume: config differs from {config_path}")
        return json.loads(meta_path.read_text())
    if meta_path.exists():
        raise FileExistsError(f"{run_dir} already holds a run; pass --resume or choose another name")
    meta = build_meta(config, classes)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dumps_toml(config))
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def write_predictions(path: Path, window_id, recording_id, label, logits, snr_db_est, num_classes: int) -> None:
    """Write one predictions_<split>.npz, checked against the contract. No pickled objects."""
    window_id = np.asarray(window_id, dtype=str)
    recording_id = np.asarray(recording_id, dtype=str)
    label = np.asarray(label, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    snr = np.asarray([np.nan if v is None else v for v in snr_db_est], dtype=np.float32)
    n = len(window_id)
    if not (len(recording_id) == len(label) == len(snr) == n) or logits.shape != (n, num_classes):
        raise ValueError(f"predictions: inconsistent shapes for {n} windows and {num_classes} classes")
    if n and (label.min() < 0 or label.max() >= num_classes):
        raise ValueError("predictions: label out of range")
    if not np.isfinite(logits).all():
        raise ValueError("predictions: logits are not finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, window_id=window_id, recording_id=recording_id, label=label, logits=logits, snr_db_est=snr)


def dumps_toml(config: dict) -> str:
    """Write a config dict (sections of scalars and flat lists) back to TOML that tomllib reads identically."""
    lines = []
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"top-level key {section!r} must be a section")
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("config floats must be finite")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # JSON's escapes are all valid TOML basic-string escapes
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"cannot write {type(value).__name__} to TOML")
