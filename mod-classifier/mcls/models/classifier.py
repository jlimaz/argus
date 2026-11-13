"""A trained checkpoint behind the `mcls.contracts.Classifier` protocol (for Engineer 10).

    clf = load_classifier("runs/cnn1d_v1/checkpoint.pt", calibration="runs/cnn1d_v1/calibration.json")
    probs = clf.classify(windows)      # complex64 (n, 4096) -> (n, len(clf.classes)), rows sum to 1
    raw = clf.logits(windows)          # always before any temperature

The temperature has one home: Engineer 8's calibration.json. With
`calibration` given, classify() returns softmax(logits / its temperature);
without it, softmax(logits) (temperature 1). The checkpoint stores none.
Torch is imported when a classifier is loaded, not when this module is imported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mcls.contracts import check_windows
from mcls.models import iq_array


class CheckpointClassifier:
    def __init__(self, model, classes, temperature: float = 1.0, device="cpu", batch_size: int = 512):
        self.model = model.to(device).eval()
        self.classes: tuple[str, ...] = tuple(classes)
        self.temperature = float(temperature)
        self.device = device
        self.batch_size = batch_size

    def logits(self, windows: np.ndarray) -> np.ndarray:
        """Float32 (n, K) raw logits, never divided by a temperature."""
        import torch

        from mcls.models.heads import main_logits

        check_windows(windows)
        out = np.empty((len(windows), len(self.classes)), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(windows), self.batch_size):
                x = torch.from_numpy(iq_array(windows[start:start + self.batch_size])).to(self.device)
                out[start:start + len(x)] = main_logits(self.model(x)).float().cpu().numpy()
        return out

    def classify(self, windows: np.ndarray) -> np.ndarray:
        return softmax(self.logits(windows) / self.temperature)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def read_temperature(calibration, classes) -> float:
    """The temperature in Engineer 8's calibration.json, checked against the checkpoint's classes. No torch."""
    path = Path(calibration)
    data = json.loads(path.read_text())
    temperature = data.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
            or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"{path}: temperature must be a positive finite number, got {temperature!r}")
    if "classes" in data and list(data["classes"]) != list(classes):
        raise ValueError(f"{path}: classes {data['classes']} do not match the checkpoint's {list(classes)}")
    return float(temperature)


def load_classifier(path, device="cpu", batch_size: int = 512, calibration=None) -> CheckpointClassifier:
    """Load checkpoint.pt; `calibration` is an optional path to calibration.json (its temperature is applied)."""
    from mcls.train.checkpoint import build_from, load_best

    payload = load_best(Path(path))
    temperature = 1.0 if calibration is None else read_temperature(calibration, payload["classes"])
    return CheckpointClassifier(build_from(payload), payload["classes"], temperature, device, batch_size)
