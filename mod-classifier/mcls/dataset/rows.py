"""Collect per-shard rows, cap classes, hold out emitters, and build a split manifest.

Builds on `mcls.manifest` (orchestrator-owned): splits come from
`assign_splits`, leakage from `check_leakage`. Everything here is a pure
function of its inputs and a seed, so the same shards and seed always give the
same manifest file and sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mcls import manifest
from mcls.contracts import LABEL_STRENGTHS, SPLITS, ManifestRow

#: PLAN 3.6: cap each class at about 30 thousand windows per dataset.
DEFAULT_CAP_PER_CLASS = 30_000
#: PLAN 5.3: weak-label windows get weight 0.5.
WEAK_WEIGHT = 0.5
#: Held-out emitters (PLAN 3.5) go here, never into the headline test.
HOLDOUT_SPLIT = "generalization"


def seeded_rng(seed: int, *keys: str) -> np.random.Generator:
    """A generator that depends only on `seed` and the string `keys` (never on Python's hash())."""
    material = hashlib.sha256(json.dumps([seed, *keys]).encode()).digest()
    return np.random.default_rng(int.from_bytes(material[:16], "big"))


#: Front-end row files (Engineer 4): `<stem>.s000.rows.jsonl`, next to their shards.
ROWS_GLOB = "*rows.jsonl"


def collect_rows(shard_dirs: Iterable[Path]) -> list[ManifestRow]:
    """Read every `*rows.jsonl` under each directory (recursively), one row dict per line.

    Rows follow ManifestRow without "split" (the front end leaves it out; it is
    set to "train" here and overwritten by the split step). A relative `shard`
    path is resolved against the directory of the rows file that names it;
    returned rows carry absolute shard paths. Rows are returned sorted by
    window_id. A window_id seen twice is an error.
    """
    rows: list[ManifestRow] = []
    seen: dict[str, Path] = {}
    for directory in shard_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"shard directory not found: {directory}")
        for path in sorted(directory.rglob(ROWS_GLOB)):
            # Hidden files and folders (e.g. `.<key>.partial.*` of killed CRC tasks) are never read.
            if any(part.startswith(".") for part in path.relative_to(directory).parts):
                continue
            with open(path) as source:
                for number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        row = ManifestRow.from_dict({"split": "train", **json.loads(line)})
                        row.validate()
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f"{path}, line {number}: {exc}") from exc
                    shard = Path(row.shard)
                    if not shard.is_absolute():
                        row.shard = str((path.parent / shard).resolve())
                    if row.window_id in seen:
                        raise ValueError(f"window_id {row.window_id!r} appears in {seen[row.window_id]} and {path}")
                    seen[row.window_id] = path
                    rows.append(row)
    rows.sort(key=lambda r: r.window_id)
    return rows


def cap_per_class(rows: Sequence[ManifestRow], cap: int, seed: int) -> list[ManifestRow]:
    """Keep at most `cap` windows per (dataset, label); a seeded random choice. cap <= 0 keeps all.

    The choice depends only on the window ids in each group and the seed, not on
    input order. Output is sorted by window_id.
    """
    if cap <= 0:
        return sorted(rows, key=lambda r: r.window_id)
    groups: dict[tuple[str, str], list[ManifestRow]] = defaultdict(list)
    for row in rows:
        groups[(row.dataset, row.label)].append(row)
    kept: list[ManifestRow] = []
    for (dataset, label), group in sorted(groups.items()):
        group.sort(key=lambda r: r.window_id)
        if len(group) <= cap:
            kept.extend(group)
            continue
        choice = seeded_rng(seed, "cap", dataset, label).permutation(len(group))[:cap]
        kept.extend(group[i] for i in choice)
    kept.sort(key=lambda r: r.window_id)
    return kept


def holdout_emitters(
    rows: Iterable[ManifestRow],
    emitter_ids: Iterable[str],
    split: str = HOLDOUT_SPLIT,
    match: str = "emitter",
) -> int:
    """Move every row of the named emitters (drone models) to `split`, in place (PLAN 3.5).

    `match="emitter"` compares `emitter_id` exactly; `match="recording"` takes
    any row whose recording_id contains one of the names (for rows that carry
    the model only in their recording id). Returns how many rows moved.

    Every recording of a held-out emitter goes to `split` whole, so after this
    the emitter appears in `split` and nowhere else.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if match not in ("emitter", "recording"):
        raise ValueError("match must be 'emitter' or 'recording'")
    names = set(emitter_ids)
    moved = 0
    for row in rows:
        if match == "emitter":
            hit = row.emitter_id is not None and row.emitter_id in names
        else:
            hit = any(name in row.recording_id for name in names)
        if hit:
            row.split = split
            moved += 1
    return moved


def check_emitter_holdout(rows: Iterable[ManifestRow], emitter_ids: Iterable[str], split: str = HOLDOUT_SPLIT) -> None:
    """Raise ValueError if a held-out emitter appears outside `split`."""
    names = set(emitter_ids)
    for row in rows:
        if row.emitter_id in names and row.split != split:
            raise ValueError(f"held-out emitter {row.emitter_id!r} in {row.split}: {row.window_id}")


def filter_classes(rows: Iterable[ManifestRow], classes: Sequence[str] | None) -> tuple[list[ManifestRow], dict[str, int]]:
    """Keep rows whose label is in `classes` (None keeps all); return (kept, dropped count per label)."""
    if classes is None:
        return list(rows), {}
    allowed = set(classes)
    kept: list[ManifestRow] = []
    dropped: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.label in allowed:
            kept.append(row)
        else:
            dropped[row.label] += 1
    return kept, dict(sorted(dropped.items()))


@dataclass
class BuildReport:
    """What `build_with_report` did besides producing rows."""

    dropped_labels: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    held_out: int = 0


def build_with_report(
    shard_dirs: Iterable[Path],
    seed: int,
    cross_test: Iterable[str] = (),
    cap: int = DEFAULT_CAP_PER_CLASS,
    holdout: Iterable[str] = (),
    holdout_split: str = HOLDOUT_SPLIT,
    holdout_match: str = "emitter",
    classes: Sequence[str] | None = None,
) -> tuple[list[ManifestRow], BuildReport]:
    """Collect, keep `classes`, cap, hold out emitters, split stratified by class, check leakage.

    Held-out emitters are moved first (to `generalization` by default) so the
    stratified split of train, val and test is made over the remaining
    recordings only.
    """
    report = BuildReport()
    rows, report.dropped_labels = filter_classes(collect_rows(shard_dirs), classes)
    rows = cap_per_class(rows, cap, seed)
    holdout = tuple(holdout)
    for row in rows:
        row.split = "train"
    if holdout:
        if holdout_split in ("train", "val"):
            raise ValueError(f"held-out emitters cannot go to {holdout_split!r}")
        report.held_out = holdout_emitters(rows, holdout, holdout_split, holdout_match)
        if report.held_out == 0:
            raise ValueError(f"no rows match held-out emitters {list(holdout)}")
    skip = (holdout_split,) if holdout else ()
    report.warnings = manifest.assign_splits_stratified(rows, seed, cross_test, skip_splits=skip)
    manifest.check_leakage(rows)
    if holdout and holdout_match == "emitter":
        check_emitter_holdout(rows, holdout, holdout_split)
    return rows, report


def build(
    shard_dirs: Iterable[Path],
    seed: int,
    cross_test: Iterable[str] = (),
    cap: int = DEFAULT_CAP_PER_CLASS,
    holdout: Iterable[str] = (),
    holdout_split: str = HOLDOUT_SPLIT,
    holdout_match: str = "emitter",
    classes: Sequence[str] | None = None,
) -> list[ManifestRow]:
    """`build_with_report` without the report."""
    return build_with_report(shard_dirs, seed, cross_test, cap, holdout, holdout_split, holdout_match, classes)[0]


def relative_to(rows: Iterable[ManifestRow], base: Path) -> list[ManifestRow]:
    """Copies of rows whose shard paths are relative to `base` (the manifest's directory)."""
    base = Path(base).resolve()
    out = []
    for row in rows:
        data = row.to_dict()
        data["shard"] = Path(os.path.relpath(Path(row.shard).resolve(), base)).as_posix()
        out.append(ManifestRow.from_dict(data))
    return out


def label_indices(rows: Sequence[ManifestRow], classes: Sequence[str]) -> np.ndarray:
    """int64 index of each row's label in `classes`; raises on a label outside them."""
    index = {name: i for i, name in enumerate(classes)}
    try:
        return np.array([index[row.label] for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]!r} is not one of {tuple(classes)}") from None


def row_weights(rows: Sequence[ManifestRow], weak_weight: float = WEAK_WEIGHT) -> np.ndarray:
    """float32 loss weight per row: 1.0 for strong labels, `weak_weight` for weak (PLAN 5.3)."""
    weights = {"strong": 1.0, "weak": weak_weight}
    assert set(weights) == set(LABEL_STRENGTHS)
    return np.array([weights[row.label_strength] for row in rows], dtype=np.float32)


def split_rows(rows: Iterable[ManifestRow], split: str) -> list[ManifestRow]:
    """The rows of one split, in manifest order."""
    return [row for row in rows if row.split == split]
