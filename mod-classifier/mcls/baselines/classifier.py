"""A fitted baseline as a `contracts.Classifier`: windows in, probabilities out."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from mcls.contracts import check_windows
from mcls.baselines import features
from mcls.baselines.models import softmax

MODEL_FILE = "model.pkl"


class FeatureClassifier:
    """Feature extraction, standardization and a baseline model, in one object."""

    def __init__(self, model, classes: tuple[str, ...], mean: np.ndarray, std: np.ndarray):
        self.model = model
        self.classes = tuple(classes)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.feature_names = features.FEATURE_NAMES

    def standardize(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def logits_from_features(self, X: np.ndarray) -> np.ndarray:
        return self.model.logits(self.standardize(X))

    def logits(self, windows: np.ndarray) -> np.ndarray:
        """(n, K) float32 scores for unit-RMS complex64 windows shaped (n, 4096)."""
        check_windows(windows)
        return self.logits_from_features(features.extract(windows))

    def classify(self, windows: np.ndarray) -> np.ndarray:
        """(n, K) probabilities whose rows sum to 1."""
        return softmax(self.logits(windows).astype(np.float64))

    def save(self, run_dir: Path) -> Path:
        path = Path(run_dir) / MODEL_FILE
        with open(path, "wb") as handle:
            pickle.dump(self, handle)
        return path


def load(run_dir: Path) -> FeatureClassifier:
    """The fitted classifier saved in a baseline run folder (a pickle: load only run folders you made)."""
    with open(Path(run_dir) / MODEL_FILE, "rb") as handle:
        return pickle.load(handle)
