"""Decide which model to trust for which kind of signal, from validation predictions only.

    python scripts/build_routing.py [--runs RUN ...] [--manifest data/manifest_w5.jsonl]
                                    [--out runs/crc/routing_w5.json]

Inputs, all written by training: each run's predictions_val.npz / predictions_test.npz
(window ids, labels, logits before temperature), calibration.json (temperature)
and meta.json (class order), plus the manifest, which gives each window's band and
measured bandwidth: things known before any model runs.

Routing: windows are grouped by band (`band` in the manifest) and, separately, by
bandwidth bucket. In each group the model with the highest validation accuracy is
chosen; a group with fewer than MIN_RECORDINGS validation recordings or
MIN_WINDOWS validation windows falls back to the model that is best on all of
validation. Test data is never used to choose anything: it only scores the
strategies afterwards.

Strategies scored on test: every model alone, the plain average of calibrated
probabilities, routing by band, routing by bandwidth, and the "oracle" (right
whenever any model is right: a ceiling no real strategy can reach, because the true
class is not known when classifying).

Output: one JSON with the routing tables, the scores and the per-class F1, read by
the UI (mod-classifier/ui) and printed as a short report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = [ROOT / "runs/crc" / name for name in ("cnn2d_v1_w5", "baseline_knn_v1_w5", "baseline_lightgbm_v1_w5")]
MIN_RECORDINGS = 2
MIN_WINDOWS = 200
#: Bandwidth buckets (upper edges in Hz) for routing by signal width.
BANDWIDTH_EDGES = [(100e3, "narrow (<100 kHz)"), (1e6, "medium (0.1-1 MHz)"), (5e6, "wide (1-5 MHz)"),
                   (float("inf"), "very wide (>5 MHz)")]


def bandwidth_bucket(bandwidth_hz: float | None) -> str:
    if bandwidth_hz is None or not np.isfinite(bandwidth_hz):
        return "unknown"
    for edge, name in BANDWIDTH_EDGES:
        if bandwidth_hz < edge:
            return name
    return BANDWIDTH_EDGES[-1][1]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def scores(labels: np.ndarray, predicted: np.ndarray, classes: list[str]) -> dict:
    """Accuracy, macro-F1 over the classes present in `labels`, and F1 per present class."""
    per_class = {}
    for k, name in enumerate(classes):
        support = int((labels == k).sum())
        if support == 0:
            continue
        tp = int(((predicted == k) & (labels == k)).sum())
        fp = int(((predicted == k) & (labels != k)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support
        per_class[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": int(len(labels)),
        "accuracy": float((labels == predicted).mean()) if len(labels) else 0.0,
        "macro_f1": float(np.mean(list(per_class.values()))) if per_class else 0.0,
        "f1": per_class,
    }


def load_run(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text())
    calibration = json.loads((run / "calibration.json").read_text())
    out = {"name": run.name, "classes": list(meta["classes"]), "temperature": float(calibration["temperature"]),
           "unknown_threshold": float(calibration["unknown_threshold"]["value"]
                                      if isinstance(calibration.get("unknown_threshold"), dict)
                                      else calibration["unknown_threshold"])}
    for split in ("val", "test"):
        data = np.load(run / f"predictions_{split}.npz", allow_pickle=True)
        out[split] = {"window_id": np.asarray(data["window_id"]).astype(str),
                      "label": np.asarray(data["label"]).astype(int),
                      "probs": softmax(np.asarray(data["logits"], dtype=np.float64) / out["temperature"])}
    return out


def read_groups(manifest: Path, wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    """window_id -> (band, bandwidth bucket, dataset/recording_id) for the windows in `wanted`."""
    groups = {}
    with open(manifest) as handle:
        for line in handle:
            row = json.loads(line)
            if row["window_id"] in wanted:
                groups[row["window_id"]] = (row.get("band") or "unknown", bandwidth_bucket(row.get("bandwidth_hz")),
                                            f"{row['dataset']}/{row['recording_id']}")
    return groups


def aligned(runs: list[dict], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window ids, labels, probs [model, window, class]) over the windows every model predicted."""
    common = set(runs[0][split]["window_id"])
    for run in runs[1:]:
        common &= set(run[split]["window_id"])
    ids = np.array(sorted(common))
    probs, labels = [], None
    for run in runs:
        index = {w: i for i, w in enumerate(run[split]["window_id"])}
        order = np.array([index[w] for w in ids])
        probs.append(run[split]["probs"][order])
        these = run[split]["label"][order]
        if labels is not None and not np.array_equal(labels, these):
            raise ValueError(f"{run['name']}: labels differ from the other models on {split}")
        labels = these
    return ids, labels, np.stack(probs)


def choose_routes(keys: np.ndarray, recordings: np.ndarray, labels: np.ndarray, correct: np.ndarray,
                  names: list[str], fallback: int) -> dict:
    """Per group key: the model with the best validation accuracy, or the fallback when the group is thin."""
    table = {}
    for key in sorted(set(keys)):
        sel = keys == key
        n_rec = len(set(recordings[sel]))
        acc = [float(correct[m][sel].mean()) for m in range(len(names))]
        thin = n_rec < MIN_RECORDINGS or sel.sum() < MIN_WINDOWS
        pick = fallback if thin else int(np.argmax(acc))
        table[key] = {"model": names[pick], "val_accuracy": {n: round(a, 4) for n, a in zip(names, acc)},
                      "val_windows": int(sel.sum()), "val_recordings": n_rec,
                      "reason": ("too little validation data: overall best model" if thin
                                 else "best validation accuracy in this group"),
                      "classes_seen": sorted(set(int(c) for c in labels[sel]))}
    return table


def route(keys: np.ndarray, table: dict, names: list[str], fallback: int) -> np.ndarray:
    lookup = {k: names.index(v["model"]) for k, v in table.items()}
    return np.array([lookup.get(k, fallback) for k in keys])


def build(runs: list[Path], manifest: Path) -> dict:
    loaded = [load_run(r) for r in runs]
    classes = loaded[0]["classes"]
    for run in loaded[1:]:
        if run["classes"] != classes:
            raise ValueError(f"{run['name']}: class order differs from {loaded[0]['name']}")
    names = [r["name"] for r in loaded]
    val_ids, val_labels, val_probs = aligned(loaded, "val")
    test_ids, test_labels, test_probs = aligned(loaded, "test")
    groups = read_groups(manifest, set(val_ids) | set(test_ids))

    def columns(ids):
        rows = [groups.get(w, ("unknown", "unknown", "unknown")) for w in ids]
        return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows]))

    val_band, val_bw, val_rec = columns(val_ids)
    test_band, test_bw, _ = columns(test_ids)
    val_pred = val_probs.argmax(axis=2)
    val_correct = val_pred == val_labels[None, :]
    overall_val = [scores(val_labels, val_pred[m], classes) for m in range(len(names))]
    fallback = int(np.argmax([s["accuracy"] for s in overall_val]))

    tables = {"band": choose_routes(val_band, val_rec, val_labels, val_correct, names, fallback),
              "bandwidth": choose_routes(val_bw, val_rec, val_labels, val_correct, names, fallback)}
    for table in tables.values():
        for entry in table.values():
            entry["classes_seen"] = [classes[c] for c in entry["classes_seen"]]

    test_pred = test_probs.argmax(axis=2)
    strategies = {f"{n} alone": test_pred[m] for m, n in enumerate(names)}
    strategies["average of all models"] = test_probs.mean(axis=0).argmax(axis=1)
    for name, keys in (("route by band", test_band), ("route by bandwidth", test_bw)):
        chosen = route(keys, tables[name.split()[-1]], names, fallback)
        strategies[name] = test_pred[chosen, np.arange(len(test_labels))]
    oracle = np.where((test_pred == test_labels[None, :]).any(axis=0), test_labels, test_pred[fallback])
    strategies["oracle (ceiling, not achievable)"] = oracle

    return {
        "classes": classes,
        "models": [{"name": r["name"], "temperature": r["temperature"], "unknown_threshold": r["unknown_threshold"],
                    "val": overall_val[i]} for i, r in enumerate(loaded)],
        "fallback_model": names[fallback],
        "min_recordings": MIN_RECORDINGS,
        "min_windows": MIN_WINDOWS,
        "bandwidth_buckets": [name for _, name in BANDWIDTH_EDGES],
        "routes": tables,
        "test": {name: scores(test_labels, pred, classes) for name, pred in strategies.items()},
        "note": ("Routes are chosen on validation only; test scores them. Bands mostly come from one dataset each, "
                 "so a route may reflect that dataset more than the signal type. pulsed_radar has no real "
                 "validation or test data (simulated only)."),
    }


def report(result: dict) -> str:
    lines = ["Test scores (windows the models never trained on; routes chosen on validation only)", ""]
    lines.append(f"{'strategy':40s} {'accuracy':>9s} {'macro-F1':>9s}")
    for name, s in result["test"].items():
        lines.append(f"{name:40s} {s['accuracy']:9.3f} {s['macro_f1']:9.3f}")
    for kind, table in result["routes"].items():
        lines += ["", f"Routing by {kind} (fallback: {result['fallback_model']})"]
        for key, entry in table.items():
            lines.append(f"  {key:28s} -> {entry['model']:26s} val windows {entry['val_windows']:6d}, "
                         f"recordings {entry['val_recordings']:3d}  ({entry['reason']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest_w5.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/crc/routing_w5.json")
    args = parser.parse_args(argv)
    result = build(args.runs, args.manifest)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(report(result))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
