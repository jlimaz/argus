"""Class-balanced sampling (PLAN 5.3), numpy only and seeded."""

from __future__ import annotations

import numpy as np


class BalancedSampler:
    """Draw indices so every class present is equally likely, each epoch.

    Each draw picks a class uniformly among the classes present in `labels`,
    then a row of that class uniformly (with replacement). `epoch(e)` depends
    only on (seed, e), so a resumed run replays the same order. `num_samples`
    defaults to len(labels): one "epoch" sees as many windows as the split has.
    """

    def __init__(self, labels: np.ndarray, seed: int, num_samples: int | None = None):
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("labels must be a non-empty 1-D array")
        self.labels = labels
        self.seed = int(seed)
        self.num_samples = int(num_samples) if num_samples is not None else int(labels.size)
        self.classes = np.unique(labels)
        self._members = [np.flatnonzero(labels == c) for c in self.classes]

    def __len__(self) -> int:
        return self.num_samples

    def epoch(self, epoch: int) -> np.ndarray:
        """int64 indices into `labels` for one epoch."""
        rng = np.random.default_rng([self.seed, int(epoch)])
        picks = rng.integers(0, len(self.classes), size=self.num_samples)
        out = np.empty(self.num_samples, dtype=np.int64)
        for k, members in enumerate(self._members):
            where = np.flatnonzero(picks == k)
            out[where] = members[rng.integers(0, members.size, size=where.size)]
        return out


def class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """float32 inverse-frequency weights, mean 1 over present classes; 0 for absent ones.

    The alternative to the balanced sampler for a weighted loss (PLAN 5.3).
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights[present] *= present.sum() / weights[present].sum()
    return weights.astype(np.float32)
