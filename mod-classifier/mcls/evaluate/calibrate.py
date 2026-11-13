"""Temperature scaling, calibration error and the "unknown" threshold (PLAN 5.5).

    python -m mcls.evaluate.calibrate --run runs/<name> [--keep 0.95] [--bins 15]

Fits on predictions_val.npz only. Writes runs/<name>/calibration.json, which
Engineer 10 reads at inference:

    p = softmax(logits / temperature)
    unknown  if  max(p) < unknown_threshold["value"]

All in NumPy. The 1-D fit uses the fact that the mean negative log-likelihood
is convex in beta = 1/T, so its derivative is monotone and bisection finds the
minimum without scipy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mcls.evaluate.metrics import false_alarm_rate, logsumexp, softmax
from mcls.evaluate.runio import load_meta, load_predictions

BETA_RANGE = (1e-3, 1e3)  # T between 0.001 and 1000


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = np.asarray(logits, dtype=np.float64) / temperature
    return float(np.mean(logsumexp(z) - z[np.arange(len(labels)), labels]))


def _nll_slope(logits: np.ndarray, labels: np.ndarray, beta: float) -> float:
    """d/d(beta) of mean NLL at logits*beta: E_p[z] - z_y, averaged."""
    p = softmax(logits * beta)
    return float(np.mean((p * logits).sum(axis=1) - logits[np.arange(len(labels)), labels]))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, tol: float = 1e-10, max_iter: int = 200) -> float:
    """Temperature T > 0 minimising the mean NLL of softmax(logits / T).

    Bisection on log(beta) over BETA_RANGE. If the slope never changes sign
    (for example every window is already right with a growing margin), the
    bound is returned; callers see it in calibration.json.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(labels):
        raise ValueError("cannot fit a temperature on zero windows")
    lo, hi = math.log(BETA_RANGE[0]), math.log(BETA_RANGE[1])
    if _nll_slope(logits, labels, math.exp(lo)) >= 0:
        return 1.0 / math.exp(lo)
    if _nll_slope(logits, labels, math.exp(hi)) <= 0:
        return 1.0 / math.exp(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _nll_slope(logits, labels, math.exp(mid)) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 1.0 / math.exp(0.5 * (lo + hi))


def reliability(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    """Equal-width confidence bins; ECE = sum over bins of (n_b / n) * |accuracy_b - confidence_b|."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    bins, ece, n = [], 0.0, len(labels)
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        acc = float(correct[mask].mean()) if count else None
        conf = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / n * abs(acc - conf)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": count, "accuracy": acc, "confidence": conf})
    return {"ece": float(ece) if n else None, "bins": bins}


def ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float | None:
    return reliability(probabilities, labels, n_bins)["ece"]


def keep_threshold(scores: np.ndarray, keep: float) -> float:
    """Largest threshold t such that at least `keep` of `scores` satisfy score >= t."""
    if not 0 < keep <= 1:
        raise ValueError("keep must be in (0, 1]")
    ordered = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    k = max(1, math.ceil(keep * len(ordered) - 1e-9))
    return float(ordered[k - 1])


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Negative free energy T * logsumexp(z / T); higher means more in-distribution."""
    return temperature * logsumexp(np.asarray(logits, dtype=np.float64) / temperature)


def coverage_accuracy(scores: np.ndarray, correct: np.ndarray, points: int = 20) -> list[list[float]]:
    """Accuracy on the most confident fraction of windows, for coverage 1/points .. 1."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    hits = np.asarray(correct, dtype=np.float64)[order]
    n = len(hits)
    out = []
    for i in range(1, points + 1):
        k = max(1, round(n * i / points))
        out.append([k / n, float(hits[:k].mean())])
    return out


def calibrate(val_logits: np.ndarray, val_labels: np.ndarray, keep: float = 0.95, n_bins: int = 15) -> dict[str, Any]:
    """Everything calibration.json holds that depends on val only."""
    temperature = fit_temperature(val_logits, val_labels)
    before = softmax(val_logits)
    after = softmax(val_logits, temperature)
    rel_before = reliability(before, val_labels, n_bins)
    rel_after = reliability(after, val_labels, n_bins)
    msp = after.max(axis=1)
    correct = after.argmax(axis=1) == val_labels
    energy = energy_score(val_logits)
    msp_t = keep_threshold(msp, keep)
    energy_t = keep_threshold(energy, keep)
    return {
        "temperature": temperature,
        "temperature_at_bound": temperature in (1 / BETA_RANGE[0], 1 / BETA_RANGE[1]),
        "fit_split": "val",
        "n_val": int(len(val_labels)),
        "nll_before": nll(val_logits, val_labels, 1.0),
        "nll_after": nll(val_logits, val_labels, temperature),
        "n_bins": n_bins,
        "ece_before": rel_before["ece"],
        "ece_after": rel_after["ece"],
        "reliability": {"before": rel_before["bins"], "after": rel_after["bins"]},
        "unknown_threshold": {
            "score": "max_softmax", "value": msp_t, "keep_fraction": keep, "applied_after_temperature": True,
            "val_kept": float((msp >= msp_t).mean()),
            "val_accuracy_kept": float(correct[msp >= msp_t].mean()),
        },
        "energy_threshold": {
            "score": "neg_energy_T1", "value": energy_t, "keep_fraction": keep,
            "definition": "logsumexp(logits) at T=1; unknown if below value (ablation only)",
            "val_kept": float((energy >= energy_t).mean()),
            "val_accuracy_kept": float(correct[energy >= energy_t].mean()),
        },
        "coverage_accuracy": {
            "val_max_softmax": coverage_accuracy(msp, correct),
            "val_energy": coverage_accuracy(energy, correct),
        },
    }


def evaluate_split(logits: np.ndarray, labels: np.ndarray, result: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    """Apply a fitted calibration to a held-out split (reporting only, nothing is refit)."""
    t = result["temperature"]
    n_bins = result["n_bins"]
    p_after = softmax(logits, t)
    msp = p_after.max(axis=1)
    correct = p_after.argmax(axis=1) == labels
    thr = result["unknown_threshold"]["value"]
    energy = energy_score(logits)
    out = {
        "n": int(len(labels)),
        "ece_before": ece(softmax(logits), labels, n_bins),
        "ece_after": ece(p_after, labels, n_bins),
        "reliability_after": reliability(p_after, labels, n_bins)["bins"],
        "kept_at_threshold": float((msp >= thr).mean()),
        "accuracy_kept": float(correct[msp >= thr].mean()) if (msp >= thr).any() else None,
        "kept_at_energy_threshold": float((energy >= result["energy_threshold"]["value"]).mean()),
        "coverage_accuracy_max_softmax": coverage_accuracy(msp, correct),
        "coverage_accuracy_energy": coverage_accuracy(energy, correct),
    }
    if "background" in classes:
        out["false_alarm_at_threshold"] = false_alarm_rate(labels, logits, classes.index("background"), t, thr)
    return out


def check_calibration(calibration: dict[str, Any], classes: list[str] | tuple[str, ...],
                      manifest_sha256: str | None = None, run: str | None = None) -> None:
    """Raise ValueError unless `calibration` belongs to this run and class list.

    Every reader of calibration.json should call this: a temperature or
    threshold fitted for another model, class order or manifest is wrong in a
    way no later check can see. `classes` must match exactly, in order.
    `manifest_sha256` (full or 16-hex short) and `run` (folder name) are
    checked when given; a calibration.json missing a field that is checked is
    refused too.
    """
    from mcls.evaluate.runio import hashes_match

    where = calibration.get("source_run", "calibration.json")
    if list(calibration.get("classes") or []) != list(classes):
        raise ValueError(f"{where}: calibration classes {calibration.get('classes')} differ from {list(classes)}")
    if manifest_sha256 is not None and not hashes_match(calibration.get("manifest_sha256"), manifest_sha256):
        raise ValueError(f"{where}: calibration manifest_sha256 {str(calibration.get('manifest_sha256'))[:16]} "
                         f"differs from {manifest_sha256[:16]}")
    if run is not None and calibration.get("run") != run:
        raise ValueError(f"{where}: calibration is for run {calibration.get('run')!r}, not {run!r}")
    for key in ("temperature", "unknown_threshold"):
        if key not in calibration:
            raise ValueError(f"{where}: calibration.json has no {key}")


def load_for_run(run_dir: Path, meta: dict[str, Any]) -> dict[str, Any] | None:
    """calibration.json of `run_dir`, checked against its meta.json; None when absent."""
    path = run_dir / "calibration.json"
    if not path.is_file():
        return None
    calibration = json.loads(path.read_text())
    check_calibration(calibration, meta["classes"], meta.get("manifest_sha256"), run_dir.resolve().name)
    return calibration


def run(run_dir: Path, keep: float = 0.95, n_bins: int = 15) -> dict[str, Any]:
    meta = load_meta(run_dir)
    classes = list(meta["classes"])
    val_path = run_dir / "predictions_val.npz"
    if not val_path.is_file():
        raise FileNotFoundError(f"{run_dir}: predictions_val.npz missing; calibration fits on val only")
    val = load_predictions(val_path, len(classes))
    result = calibrate(val.logits, val.label, keep, n_bins)
    result["held_out"] = {}
    for split in ("test", "cross_test", "generalization"):
        path = run_dir / f"predictions_{split}.npz"
        if path.is_file():
            pred = load_predictions(path, len(classes))
            if len(pred):
                result["held_out"][split] = evaluate_split(pred.logits, pred.label, result, classes)
    result.update({
        "classes": classes,
        "run": run_dir.resolve().name,
        "source_run": str(run_dir),
        "manifest_sha256": meta.get("manifest_sha256"),
        "git_commit": meta.get("git_commit"),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    (run_dir / "calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mcls.evaluate.calibrate", description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--keep", type=float, default=0.95, help="share of val windows the unknown threshold keeps")
    parser.add_argument("--bins", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        result = run(args.run, args.keep, args.bins)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"temperature={result['temperature']:.4f} ece_before={result['ece_before']:.4f} "
          f"ece_after={result['ece_after']:.4f} unknown_threshold={result['unknown_threshold']['value']:.4f} "
          f"-> {args.run / 'calibration.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
