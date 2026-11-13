"""Evaluate saved predictions of one or more runs (PLAN 6.1).

    python -m mcls.evaluate --run runs/<name> [--run runs/<other> ...] [--manifest PATH] [--no-plots]

Per run, writes runs/<name>/metrics.json (and plots/ when matplotlib is
importable). With several runs, also writes a comparison table to
--compare-out (default: <first run>/comparison.json).

Reads only the run-output contract (meta.json, predictions_<split>.npz,
optional calibration.json from `python -m mcls.evaluate.calibrate`). The
manifest is required whenever a reported split (anything but train and val)
exists, and must match the run's manifest_sha256; it supplies
is_real (so test numbers are real rows only), burst grouping, synthetic
training-row counts and the split leakage check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mcls.evaluate import gates as gates_mod
from mcls.evaluate import metrics as M
from mcls.evaluate.runio import Predictions, burst_keys, load_manifest, load_meta, load_predictions, prediction_files

#: Splits never reported as results (fit or model selection happen on them).
NOT_REPORTED = ("train", "val")
HEADLINE = "test"


#: How a burst becomes unknown, identical to mcls.infer.core.aggregate minus the band prior.
BURST_RULE = ("p = softmax(mean(window logits) / temperature); unknown if max(p) < unknown_threshold; "
              "band prior NOT applied (it lives in mcls.infer.priors, used by mcls.infer.core.aggregate)")


def _aggregate(pred: Predictions, keys: np.ndarray, classes: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    agg = M.burst_logits(pred.logits, pred.label, keys)
    out = M.summarize(agg["labels"], agg["logits"].argmax(axis=1), classes)
    out["mixed_label_groups"] = agg["mixed_label_bursts"]
    out["windows_per_group_mean"] = float(agg["windows"].mean()) if len(agg["windows"]) else None
    return out, agg


def burst_after_threshold(agg: dict[str, Any], classes: list[str], temperature: float,
                          threshold: float) -> dict[str, Any]:
    """Burst results after the unknown threshold, by the inference rule (BURST_RULE)."""
    from mcls.evaluate.calibrate import coverage_accuracy

    labels = agg["labels"]
    p = M.softmax(agg["logits"], temperature)
    top, confidence = p.argmax(axis=1), p.max(axis=1)
    kept = confidence >= threshold
    k = len(classes)
    confusion = np.zeros((k, k + 1), dtype=np.int64)
    np.add.at(confusion, (labels, np.where(kept, top, k)), 1)
    unknown_share = {}
    for i, name in enumerate(classes):
        n = int((labels == i).sum())
        unknown_share[name] = float((~kept[labels == i]).mean()) if n else None
    out = {
        "rule": BURST_RULE,
        "temperature": temperature,
        "threshold": threshold,
        "n_bursts": int(len(labels)),
        "n_unknown": int((~kept).sum()),
        "kept_fraction": float(kept.mean()) if len(labels) else None,
        "kept": M.summarize(labels[kept], top[kept], classes),
        "unknown_share_by_true_class": unknown_share,
        "confusion_with_unknown": {"columns": [*classes, "unknown"], "matrix": confusion.tolist()},
        "coverage_accuracy_max_softmax": coverage_accuracy(confidence, top == labels) if len(labels) else [],
    }
    if "background" in classes:
        b = classes.index("background")
        bg = labels == b
        alarms = (top != b) & kept & bg
        out["false_alarm_background"] = {"n_background": int(bg.sum()),
                                         "rate": float(alarms.sum() / bg.sum()) if bg.any() else None}
    return out


def evaluate_split(pred: Predictions, classes: list[str], rows: dict, keys_by_window: dict,
                   calibration: dict | None) -> dict[str, Any]:
    result: dict[str, Any] = {"n_windows_file": len(pred)}
    in_manifest = np.array([w in rows for w in pred.window_id], dtype=bool)
    result["n_not_in_manifest"] = int((~in_manifest).sum())
    pred = pred.subset(in_manifest)
    real = np.array([rows[w].is_real for w in pred.window_id], dtype=bool)
    result["n_dropped_not_real"] = int((~real).sum())
    pred = pred.subset(real)
    result["real_only_verified"] = True
    result["n_windows"] = len(pred)
    if not len(pred):
        return result
    predicted = pred.logits.argmax(axis=1)
    result["window"] = M.summarize(pred.label, predicted, classes)
    result["recording"], _ = _aggregate(pred, pred.recording_id, classes)
    with_id = np.array([rows[w].burst_id is not None for w in pred.window_id], dtype=bool)
    burst, agg = _aggregate(pred, np.array([keys_by_window[w] for w in pred.window_id]), classes)
    burst["grouping"] = {
        "windows_by_burst_id": int(with_id.sum()),
        "windows_by_fallback": int((~with_id).sum()),
        "fallback_rule": "burst_id is None: consecutive windows of one shard with equal (snr_db_est, bandwidth_hz, "
                         "duty_cycle); windows with no stats (idle background) are one-window groups",
    }
    if calibration:
        burst["after_unknown_threshold"] = burst_after_threshold(
            agg, classes, calibration["temperature"], calibration["unknown_threshold"]["value"])
    result["burst"] = burst
    result["accuracy_vs_snr"] = M.accuracy_by_snr(pred.label, predicted, pred.snr_db_est)
    if "background" in classes:
        b = classes.index("background")
        fa = {"argmax": M.false_alarm_rate(pred.label, pred.logits, b)}
        if calibration:
            fa["at_unknown_threshold"] = M.false_alarm_rate(
                pred.label, pred.logits, b, calibration["temperature"], calibration["unknown_threshold"]["value"])
        result["false_alarm_background"] = fa
    return result


#: Role of each reported split. Only `test` is the headline; the others are shown next to it, never merged into it.
ROLES = {"test": "headline", "cross_test": "cross_dataset", "generalization": "held_out_emitter"}


def _row(split: str, r: dict[str, Any]) -> dict[str, Any]:
    return {
        "split": split,
        "role": ROLES.get(split, "other"),
        "n_windows": r["n_windows"],
        "window_macro_f1": (r.get("window") or {}).get("macro_f1"),
        "window_accuracy": (r.get("window") or {}).get("accuracy"),
        "burst_macro_f1": (r.get("burst") or {}).get("macro_f1"),
        "recording_macro_f1": (r.get("recording") or {}).get("macro_f1"),
    }


def evaluate_run(run: Path, manifest_path: Path | None = None, plots: bool = True) -> dict[str, Any]:
    from mcls.evaluate.calibrate import load_for_run

    meta = load_meta(run)
    classes = list(meta["classes"])
    notes: list[str] = []
    if manifest_path is None:
        reported = [s for s in prediction_files(run) if s not in NOT_REPORTED]
        raise ValueError(
            f"--manifest is required to report {', '.join(reported) or 'any split'}: without it is_real cannot be "
            "checked, and reported numbers must come from real rows only (PLAN 3.4)")
    rows, manifest_info = load_manifest(manifest_path, meta)
    if not manifest_info["matches_meta"]:
        notes.append("meta.json has no manifest_sha256; manifest identity not checked")
    keys, grouping = burst_keys(list(rows.values()))
    manifest_info["burst_grouping"] = grouping
    if grouping["inferred_equal_stats"]:
        notes.append(f"{grouping['inferred_equal_stats']} manifest windows have no burst_id: bursts inferred "
                     "from equal stats for those")
    calibration = load_for_run(run, meta)  # refuses one from another run, class list or manifest
    if calibration is None:
        notes.append("no calibration.json: false alarm at argmax only, no burst results after the threshold")
    else:
        notes.append("burst results after the threshold skip the band prior (mcls.infer.priors)")

    splits = {}
    for split, path in prediction_files(run).items():
        if split in NOT_REPORTED:
            continue
        splits[split] = evaluate_split(load_predictions(path, len(classes)), classes, rows, keys, calibration)
    if HEADLINE not in splits:
        notes.append(f"no predictions_{HEADLINE}.npz: no headline result")

    side_by_side = [_row(split, r) for split, r in splits.items()]
    headline = next((r for r in side_by_side if r["split"] == HEADLINE), None)
    tr = manifest_info["train_rows"]
    training = {**tr, "synthetic_in_training": tr["synthetic"] > 0}

    held = (calibration or {}).get("held_out", {})
    test = splits.get(HEADLINE, {})
    fa = (test.get("false_alarm_background") or {}).get("at_unknown_threshold")
    gen = splits.get("generalization") or {}
    cross = splits.get("cross_test") or {}
    gate_inputs = {
        "ece_after": (held.get(HEADLINE) or {}).get("ece_after"),
        "background_false_alarm": fa["rate"] if fa else None,
        "leakage_ok": manifest_info["leakage"]["ok"],
        "cross_test_reported": bool(cross.get("n_windows")),
        "generalization_reported": bool(gen.get("n_windows")),
        "cross_test_macro_f1": (cross.get("window") or {}).get("macro_f1"),
        "generalization_macro_f1": (gen.get("window") or {}).get("macro_f1"),
    }
    metrics = {
        "run": str(run),
        "meta": meta,
        "classes": classes,
        "manifest": manifest_info,
        "training_rows": training,
        "calibration": None if calibration is None else {
            "temperature": calibration["temperature"],
            "unknown_threshold": calibration["unknown_threshold"]["value"],
            "ece_before_val": calibration["ece_before"], "ece_after_val": calibration["ece_after"],
            "held_out": {s: {"ece_before": h["ece_before"], "ece_after": h["ece_after"]} for s, h in held.items()},
        },
        "headline": headline,
        "splits": splits,
        "side_by_side": side_by_side,
        "gate_inputs": gate_inputs,
        "gates": gates_mod.check_gates(gate_inputs),
        "notes": notes,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics["plots"] = []
    if plots:
        from mcls.evaluate import plots as plots_mod
        metrics["plots"] = plots_mod.write(run, metrics, calibration)
        if not metrics["plots"]:
            notes.append("matplotlib not importable: plots skipped")
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default) + "\n")
    return metrics


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def compare(all_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Baseline comparison table on identical splits, one row per run."""
    hashes = {str(m["meta"].get("manifest_sha256")) for m in all_metrics}
    table = []
    for m in all_metrics:
        s = m["splits"]
        row = {
            "run": m["run"], "model": m["meta"].get("model"), "seed": m["meta"].get("seed"),
            "synthetic_train_rows": (m["training_rows"] or {}).get("synthetic"),
        }
        for split in ("test", "cross_test"):
            r = s.get(split) or {}
            row[f"{split}_window_macro_f1"] = (r.get("window") or {}).get("macro_f1")
            row[f"{split}_burst_macro_f1"] = (r.get("burst") or {}).get("macro_f1")
        for split in s:
            if split.startswith("generalization"):
                row[f"{split}_window_macro_f1"] = (s[split].get("window") or {}).get("macro_f1")
        row["ece_after_test"] = ((m["calibration"] or {}).get("held_out", {}).get("test") or {}).get("ece_after")
        table.append(row)
    with_syn = [r["run"] for r in table if r["synthetic_train_rows"]]
    without = [r["run"] for r in table if r["synthetic_train_rows"] == 0]
    return {
        "same_manifest": len(hashes) == 1 and "None" not in hashes,
        "table": table,
        "with_synthetic_training_rows": with_syn,
        "without_synthetic_training_rows": without,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mcls.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--compare-out", type=Path, default=None)
    args = parser.parse_args(argv)
    results = []
    for run in args.run:
        try:
            m = evaluate_run(run, args.manifest, plots=not args.no_plots)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {run}: {exc}", file=sys.stderr)
            return 2
        results.append(m)
        print(f"{run}: metrics.json written")
        for row in m["side_by_side"]:
            print(f"  {row['split']:<16} n={row['n_windows']:<8} window macro-F1={_fmt(row['window_macro_f1'])} "
                  f"burst macro-F1={_fmt(row['burst_macro_f1'])}")
        for note in m["notes"]:
            print(f"  note: {note}")
    if len(results) > 1:
        out = args.compare_out or args.run[0] / "comparison.json"
        table = compare(results)
        out.write_text(json.dumps(table, indent=2) + "\n")
        print(f"comparison: {out} (same manifest: {table['same_manifest']})")
    return 0


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    sys.exit(main())
