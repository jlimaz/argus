"""mcls.infer with Engineer 6's real baseline classifier (numpy only), saved and loaded from a run folder."""

from __future__ import annotations

import json

import numpy as np

from mcls.baselines import features
from mcls.baselines.classifier import FeatureClassifier
from mcls.baselines.models import SoftmaxRegression
from mcls.contracts import WINDOW_SAMPLES
from mcls.infer import UNKNOWN, classify_bursts, load_calibration, load_classifier
from mcls.frontend.windows import normalize

CLASSES = ("drone_link", "wifi", "bluetooth", "background")
FS = 20e6


def _windows(kind: str, n: int, rng) -> np.ndarray:
    noise = (rng.standard_normal((n, WINDOW_SAMPLES)) + 1j * rng.standard_normal((n, WINDOW_SAMPLES))) * 0.05
    if kind == "tone":
        t = np.arange(WINDOW_SAMPLES) / FS
        noise = noise + np.exp(2j * np.pi * rng.uniform(-3e6, 3e6, (n, 1)) * t)
    return normalize(noise.astype(np.complex64))


def test_baseline_run_folder_classifies_bursts(tmp_path):
    rng = np.random.default_rng(1)
    X = np.concatenate([features.extract(_windows("tone", 24, rng)), features.extract(_windows("noise", 24, rng))])
    y = np.array([0] * 24 + [3] * 24)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    clf = FeatureClassifier(None, CLASSES, mean, std)
    clf.model = SoftmaxRegression(len(CLASSES), params={"epochs": 30, "lr": 0.1}).fit(clf.standardize(X), y)
    clf.save(tmp_path)
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"classes": list(CLASSES), "temperature": 1.0, "unknown_threshold": {"score": "max_softmax", "value": 0.5, "applied_after_temperature": True}}
    ))

    loaded = load_classifier(tmp_path)
    assert loaded.classes == CLASSES

    t = np.arange(200_000) / FS
    iq = (0.01 * (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000))).astype(np.complex64)
    iq[20_000:80_000] += np.exp(2j * np.pi * 1e6 * t[20_000:80_000]).astype(np.complex64)
    bursts = [
        {"start_s": 20_000 / FS, "end_s": 80_000 / FS, "center_frequency_hz": 2.441e9},
        {"start_s": 120_000 / FS, "end_s": 180_000 / FS, "center_frequency_hz": 2.44e9},
    ]
    out = classify_bursts(iq, FS, 2.44e9, bursts, loaded, calibration=load_calibration(tmp_path, loaded.classes))
    assert out[0].signal_class == "drone_link"
    assert out[1].signal_class in ("background", UNKNOWN)
    assert all(abs(sum(r.probs.values()) - 1) < 1e-9 for r in out)
