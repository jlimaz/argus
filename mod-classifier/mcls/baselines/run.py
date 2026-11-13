"""Fit a baseline on a manifest and write its run folder (PROTOCOL run-output contract).

    runs/<name>/
      config.toml               the effective run config
      meta.json                 git_commit, manifest_sha256, seed, classes, model, created
      predictions_<split>.npz   window_id, recording_id, label, logits, snr_db_est
                                (val, test, cross_test, generalization: each split that has rows)
      model.pkl                 the fitted FeatureClassifier (mcls.baselines.load)

metrics.json is the evaluator's.

A run folder is never reused silently: results must not be credited to another
run's manifest or seed. `train` refuses a folder that already holds results
(predictions, model.pkl or metrics.json), or a meta.json whose manifest_sha256,
seed, model or classes differ from this run. A meta.json that matches exactly is
kept: Engineer 7's `python -m mcls.train` writes config.toml and meta.json
before calling `train`. `overwrite=True` (CLI `--overwrite`) first deletes the run
files of a folder that is recognisably a run folder (see `clear_run_dir`), and
refuses any other folder.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from mcls import config as config_module
from mcls import manifest as manifest_module
from mcls import taxonomy
from mcls.contracts import SPLITS, WINDOW_SAMPLES, ManifestRow
from mcls.baselines import features
from mcls.baselines.classifier import FeatureClassifier
from mcls.baselines.models import MODELS, build

#: Splits that get a predictions file, when they have rows.
PREDICT_SPLITS = ("val", "test", "cross_test", "generalization")
#: Files that mean a folder already holds a finished (or partly written) run.
RESULT_FILES = ("model.pkl", "metrics.json")
#: The only files --overwrite may delete: the run-output contract plus what e6, e7 and e8 add.
RUN_FILES = frozenset({
    "config.toml", "meta.json", "metrics.json", "model.pkl", "calibration.json",
    "checkpoint.pt", "last.pt", "history.jsonl",
})
#: A folder is a run folder only if it holds at least one of these.
RUN_MARKERS = frozenset({"config.toml", "meta.json"})
PREDICTIONS_FILE = re.compile(r"predictions_[A-Za-z0-9_]+\.npz")
WEAK_WEIGHT = 0.5
#: Windows read and featurized at a time (32 MB of samples).
CHUNK = 1024


def train(
    config: dict,
    run_dir: Path,
    manifest: Path | None = None,
    classes: tuple[str, ...] | None = None,
    overwrite: bool = False,
) -> None:
    """Fit config["model"]["name"] on the train split, select on val, write run_dir.

    `classes` overrides the taxonomy's class list (tests use it). Raises
    FileExistsError when run_dir holds another run (see the module docstring).
    """
    config_module.validate(config)
    name = config["model"]["name"]
    if name not in MODELS:
        raise ValueError(f"model.name {name!r} is not a baseline (choose from {sorted(MODELS)})")
    run_dir = Path(run_dir)
    manifest_path = Path(manifest if manifest is not None else config["data"]["manifest"])
    seed = int(config["run"]["seed"])
    if classes is None:
        classes = taxonomy.load().classes(config["data"]["classes"])
    classes = tuple(classes)
    expected = {
        "manifest_sha256": sha256_file(manifest_path),
        "seed": seed,
        "model": name,
        "classes": list(classes),
    }
    if overwrite:
        clear_run_dir(run_dir)
    meta_matches = check_run_dir(run_dir, expected)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not meta_matches:
        effective = copy.deepcopy(config)
        effective["data"]["manifest"] = str(manifest_path)
        (run_dir / "config.toml").write_text(to_toml(effective))
        write_meta(run_dir, expected)

    splits = select_rows(manifest_module.read(manifest_path), classes, config["data"], seed)
    if not splits["train"]:
        raise ValueError(f"{manifest_path}: no train rows with a label in {classes}")
    store = window_store(manifest_path.parent)
    X = {split: featurize(rows, store) for split, rows in splits.items()}
    y = {split: label_indices(rows, classes) for split, rows in splits.items()}

    mean = X["train"].astype(np.float64).mean(axis=0)
    std = X["train"].astype(np.float64).std(axis=0)
    std[std < 1e-9] = 1.0
    classifier = FeatureClassifier(None, classes, mean, std)
    model = build(name, len(classes), seed=seed, params=config.get("optim", {}))
    weights = np.array([WEAK_WEIGHT if r.label_strength == "weak" else 1.0 for r in splits["train"]])
    model.fit(
        classifier.standardize(X["train"]),
        y["train"],
        classifier.standardize(X["val"]),
        y["val"],
        sample_weight=weights,
    )
    classifier.model = model
    classifier.save(run_dir)

    for split in PREDICT_SPLITS:
        rows = splits[split]
        if rows:
            write_predictions(run_dir / f"predictions_{split}.npz", rows, y[split], classifier.logits_from_features(X[split]))


# ---------------------------------------------------------------------------
# Rows and windows
# ---------------------------------------------------------------------------


def _is_run_file(name: str) -> bool:
    return name in RUN_FILES or PREDICTIONS_FILE.fullmatch(name) is not None


def clear_run_dir(run_dir: Path) -> list[str]:
    """Delete the run files in run_dir (for --overwrite); return their names.

    Refuses (FileExistsError) unless run_dir is a real directory (not a symlink)
    holding meta.json, config.toml or predictions_*.npz, and nothing but regular
    files named in RUN_FILES or predictions_<split>.npz. Only those files are
    unlinked; the folder itself stays, and no link is followed. A missing or
    empty folder is left as is.
    """
    run_dir = Path(run_dir)
    if run_dir.is_symlink():
        raise FileExistsError(f"--overwrite refused: {run_dir} is a symlink")
    if not run_dir.exists():
        return []
    if not run_dir.is_dir():
        raise FileExistsError(f"--overwrite refused: {run_dir} is not a folder")
    entries = sorted(run_dir.iterdir(), key=lambda p: p.name)
    if not entries:
        return []
    unknown = [p.name for p in entries if p.is_symlink() or not p.is_file() or not _is_run_file(p.name)]
    if unknown:
        raise FileExistsError(
            f"--overwrite refused: {run_dir} holds files that are not run files ({', '.join(unknown)}); "
            "remove them yourself or choose another folder"
        )
    names = [p.name for p in entries]
    if not any(name in RUN_MARKERS or PREDICTIONS_FILE.fullmatch(name) for name in names):
        raise FileExistsError(f"--overwrite refused: {run_dir} does not look like a run folder (no meta.json, "
                              "config.toml or predictions_*.npz)")
    for path in entries:
        path.unlink()
    return names


def check_run_dir(run_dir: Path, expected: dict) -> bool:
    """True when run_dir holds a meta.json matching `expected` (keep it), False when it holds none.

    Raises FileExistsError when run_dir already holds results or another run's meta.json.
    """
    if not run_dir.exists():
        return False
    results = sorted(p.name for p in run_dir.glob("predictions_*.npz"))
    results += [name for name in RESULT_FILES if (run_dir / name).exists()]
    if results:
        raise FileExistsError(f"{run_dir} already holds results ({', '.join(results)}); pass --overwrite or choose another folder")
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except ValueError as exc:
        raise FileExistsError(f"{meta_path} is not valid JSON ({exc}); pass --overwrite") from exc
    differ = [key for key, value in expected.items() if not isinstance(meta, dict) or meta.get(key) != value]
    if differ:
        raise FileExistsError(f"{meta_path} belongs to another run ({', '.join(differ)} differ); pass --overwrite or choose another folder")
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_rows(rows: list[ManifestRow], classes: tuple[str, ...], data: dict, seed: int) -> dict[str, list[ManifestRow]]:
    """Rows per split: labels in `classes` only; real rows only outside train unless
    data.use_simulated (and then simulated rows only in train); train capped per class."""
    use_simulated = bool(data.get("use_simulated", False))
    splits: dict[str, list[ManifestRow]] = {split: [] for split in SPLITS}
    for row in rows:
        if row.label not in classes or row.split not in splits:
            continue
        if not row.is_real and not (use_simulated and row.split == "train"):
            continue
        splits[row.split].append(row)
    cap = data.get("max_windows_per_class")
    if cap:
        splits["train"] = cap_per_class(splits["train"], int(cap), seed)
    return splits


def cap_per_class(rows: list[ManifestRow], cap: int, seed: int) -> list[ManifestRow]:
    """At most `cap` rows per label, chosen by a seeded draw; manifest order kept."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_label.setdefault(row.label, []).append(i)
    keep: list[int] = []
    for label in sorted(by_label):
        index = by_label[label]
        keep += index if len(index) <= cap else sorted(rng.choice(index, cap, replace=False).tolist())
    return [rows[i] for i in sorted(keep)]


def label_indices(rows: list[ManifestRow], classes: tuple[str, ...]) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(classes)}
    return np.array([lookup[r.label] for r in rows], dtype=np.int64)


class ShardReader:
    """Reads windows straight from shards (np.load mmap); offset indexes the flattened shard.

    The same behaviour as Engineer 5's mcls.dataset.WindowStore, used until that is merged.
    """

    def __init__(self, root: Path | None = None):
        self.root = None if root is None else Path(root)
        self._open: dict[str, np.ndarray] = {}

    def _shard(self, name: str) -> np.ndarray:
        if name not in self._open:
            path = Path(name)
            if not path.is_absolute() and self.root is not None:
                path = self.root / path
            self._open[name] = np.load(path, mmap_mode="r").reshape(-1)
        return self._open[name]

    def windows(self, rows: list[ManifestRow]) -> np.ndarray:
        out = np.empty((len(rows), WINDOW_SAMPLES), dtype=np.complex64)
        for i, row in enumerate(rows):
            out[i] = self._shard(row.shard)[row.offset : row.offset + row.length]
        return out


def window_store(root: Path):
    """Engineer 5's WindowStore when present, else ShardReader."""
    try:
        from mcls.dataset import WindowStore
    except ImportError:
        return ShardReader(root)
    return WindowStore(root)


def featurize(rows: list[ManifestRow], store) -> np.ndarray:
    blocks = [features.extract(store.windows(rows[i : i + CHUNK])) for i in range(0, len(rows), CHUNK)]
    if not blocks:
        return np.zeros((0, len(features.FEATURE_NAMES)), dtype=np.float32)
    return np.concatenate(blocks)


# ---------------------------------------------------------------------------
# Run-folder files
# ---------------------------------------------------------------------------


def write_predictions(path: Path, rows: list[ManifestRow], labels: np.ndarray, logits: np.ndarray) -> None:
    snr = np.array([np.nan if r.snr_db_est is None else r.snr_db_est for r in rows], dtype=np.float32)
    np.savez(
        path,
        window_id=np.array([r.window_id for r in rows], dtype=str),
        recording_id=np.array([r.recording_id for r in rows], dtype=str),
        label=np.asarray(labels, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32),
        snr_db_est=snr,
    )


def write_meta(run_dir: Path, expected: dict) -> None:
    meta = {
        "git_commit": git_commit(),
        "manifest_sha256": expected["manifest_sha256"],
        "seed": expected["seed"],
        "classes": list(expected["classes"]),
        "model": expected["model"],
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    (Path(run_dir) / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def to_toml(config: dict) -> str:
    """A config dict (sections of scalars and lists) as TOML that tomllib reads back unchanged."""
    lines = []
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"top-level key {section!r} must be a section")
        lines.append(f"[{section}]")
        lines += [f"{key} = {_toml_value(value)}" for key, value in values.items()]
        lines.append("")
    return "\n".join(lines)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ValueError(f"cannot write {type(value).__name__} to TOML")
