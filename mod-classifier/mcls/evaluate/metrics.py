"""Classification metrics in plain NumPy (no scikit-learn on the laptop).

Conventions, stated once:
- Labels and predictions are int indices into the run's `classes`.
- Macro-F1 averages over classes that have at least one true window in the
  evaluated set. A class absent from the set cannot be scored; which classes
  were counted is returned with the score.
- Precision is None (not 0) when a class was never predicted; its F1 is then 0
  if it has support.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def logsumexp(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, k: int) -> np.ndarray:
    """Rows are true classes, columns predicted classes."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions differ in shape")
    if labels.size and (labels.min() < 0 or labels.max() >= k or predictions.min() < 0 or predictions.max() >= k):
        raise ValueError(f"indices must lie in 0..{k - 1}")
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    return cm


def per_class(cm: np.ndarray, classes: Sequence[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        support = int(cm[i].sum())
        predicted = int(cm[:, i].sum())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if precision and recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0 if support else None
        out[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "predicted": predicted}
    return out


def macro_f1(cm: np.ndarray) -> float | None:
    """Mean F1 over classes with support; None for an empty set."""
    scores = [s["f1"] for s in per_class(cm, [str(i) for i in range(len(cm))]).values() if s["support"]]
    return float(np.mean(scores)) if scores else None


def summarize(labels: np.ndarray, predictions: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class precision/recall/F1 and the confusion matrix."""
    cm = confusion_matrix(labels, predictions, len(classes))
    n = int(cm.sum())
    pc = per_class(cm, classes)
    return {
        "n": n,
        "accuracy": float(np.trace(cm) / n) if n else None,
        "macro_f1": macro_f1(cm),
        "macro_f1_classes": [c for c in classes if pc[c]["support"]],
        "per_class": pc,
        "confusion": cm.tolist(),
    }


def burst_logits(logits: np.ndarray, labels: np.ndarray, keys: Sequence[str]) -> dict[str, Any]:
    """Average logits over the windows of each burst.

    Returns burst keys (sorted), mean logits, the burst label (that of its
    windows) and how many bursts mixed labels (should be 0; reported, and the
    most common label is used).
    """
    keys = np.asarray(keys)
    unique, inverse = np.unique(keys, return_inverse=True)
    k = logits.shape[1]
    sums = np.zeros((len(unique), k), dtype=np.float64)
    np.add.at(sums, inverse, np.asarray(logits, dtype=np.float64))
    counts = np.bincount(inverse, minlength=len(unique))
    label_counts = np.zeros((len(unique), k), dtype=np.int64)
    np.add.at(label_counts, (inverse, np.asarray(labels, dtype=np.int64)), 1)
    mixed = int(((label_counts > 0).sum(axis=1) > 1).sum())
    return {
        "keys": unique,
        "logits": sums / counts[:, None],
        "labels": label_counts.argmax(axis=1),
        "windows": counts,
        "mixed_label_bursts": mixed,
    }


def accuracy_by_snr(
    labels: np.ndarray, predictions: np.ndarray, snr_db: np.ndarray, edges: Sequence[float] | None = None
) -> dict[str, Any]:
    """Accuracy per SNR bin [lo, hi). NaN SNR is counted apart, never binned."""
    if edges is None:
        edges = list(range(-20, 41, 5))
    edges = [-np.inf, *edges, np.inf]
    snr = np.asarray(snr_db, dtype=np.float64)
    correct = np.asarray(labels) == np.asarray(predictions)
    known = np.isfinite(snr)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = known & (snr >= lo) & (snr < hi)
        n = int(mask.sum())
        bins.append({
            "lo": None if np.isinf(lo) else float(lo),
            "hi": None if np.isinf(hi) else float(hi),
            "n": n,
            "accuracy": float(correct[mask].mean()) if n else None,
        })
    return {"bins": bins, "n_snr_unknown": int((~known).sum())}


def false_alarm_rate(
    labels: np.ndarray,
    logits: np.ndarray,
    background_index: int,
    temperature: float = 1.0,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Share of true-`background` windows called as some other class.

    With `threshold`, a call counts only when its max softmax probability (at
    `temperature`) is at or above the threshold; below it the window is
    `unknown`, which is not an alarm.
    """
    labels = np.asarray(labels)
    mask = labels == background_index
    n = int(mask.sum())
    if not n:
        return {"n_background": 0, "rate": None, "threshold": threshold}
    p = softmax(logits[mask], temperature)
    alarm = p.argmax(axis=1) != background_index
    if threshold is not None:
        alarm &= p.max(axis=1) >= threshold
    return {"n_background": n, "rate": float(alarm.mean()), "threshold": threshold, "temperature": temperature}
