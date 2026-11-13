"""Early stopping on validation macro-F1 (PLAN.md 5.3). No torch.

`macro_f1` here only drives early stopping and checkpoint choice; the reported
numbers come from the evaluator (Engineer 8).
"""

from __future__ import annotations

import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Unweighted mean F1 over the classes present in y_true or y_pred.

    Classes absent from both are left out, as scikit-learn's f1_score does by
    default with average="macro". Returns 0.0 for empty input.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        return 0.0
    scores = []
    for k in range(num_classes):
        true_k, pred_k = y_true == k, y_pred == k
        if not true_k.any() and not pred_k.any():
            continue
        tp = int(np.sum(true_k & pred_k))
        denom = int(true_k.sum()) + int(pred_k.sum())
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores)) if scores else 0.0


class EarlyStopper:
    """Track the best score; `step(score)` returns True when this epoch is a new best.

    `should_stop` turns True after `patience` epochs without an improvement
    larger than `min_delta`. patience 0 disables stopping.
    """

    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        if score > self.best + self.min_delta:
            self.best = float(score)
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict) -> None:
        self.best = float(state["best"])
        self.bad_epochs = int(state["bad_epochs"])
