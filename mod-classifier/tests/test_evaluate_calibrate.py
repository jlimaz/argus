"""Temperature scaling, ECE and the unknown threshold (synthetic logits, tests only)."""

import numpy as np
import pytest

from mcls.evaluate import calibrate as C
from mcls.evaluate.metrics import softmax


def _sample(rng, n, k, true_scale=2.0):
    """Logits z and labels drawn from softmax(z), so softmax(z) is calibrated by construction."""
    z = rng.normal(size=(n, k)) * true_scale
    p = softmax(z)
    labels = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return z, labels


@pytest.mark.parametrize("t_true", [0.5, 1.0, 2.5])
def test_temperature_recovers_known_value(t_true):
    rng = np.random.default_rng(7)
    z, labels = _sample(rng, 40_000, 4)
    # The "model" reports z * T_true; dividing by T_true recovers calibrated logits.
    t = C.fit_temperature(z * t_true, labels)
    assert t == pytest.approx(t_true, rel=0.05)


def test_fitted_temperature_minimises_nll():
    rng = np.random.default_rng(1)
    z, labels = _sample(rng, 5_000, 3)
    logits = z * 3.0
    t = C.fit_temperature(logits, labels)
    best = C.nll(logits, labels, t)
    for other in (t * 0.9, t * 1.1):
        assert best <= C.nll(logits, labels, other)


def test_ece_exactly_zero_on_hand_case():
    # 10 windows at confidence 0.8, 8 of them right: accuracy equals confidence in the only bin.
    p = np.tile([0.8, 0.2], (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert C.ece(p, labels) == pytest.approx(0.0, abs=1e-12)


def test_ece_by_hand_two_bins():
    # bin of 0.9 (4 windows, 2 right): gap 0.4; bin of 0.6 (4 windows, all right): gap 0.4 -> ECE 0.4
    p = np.array([[0.9, 0.1]] * 4 + [[0.6, 0.4]] * 4)
    labels = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    assert C.ece(p, labels, n_bins=10) == pytest.approx(0.4)


def test_ece_of_calibrated_toy_is_near_zero():
    rng = np.random.default_rng(3)
    z, labels = _sample(rng, 100_000, 4)
    assert C.ece(softmax(z), labels) < 0.01
    # The same logits sharpened 3x are badly calibrated, and scaling fixes it.
    result = C.calibrate(z * 3.0, labels)
    assert result["ece_before"] > 0.05
    assert result["ece_after"] < 0.01
    assert result["temperature"] == pytest.approx(3.0, rel=0.05)


def test_reliability_bins_cover_all_windows():
    rng = np.random.default_rng(0)
    z, labels = _sample(rng, 1000, 3)
    rel = C.reliability(softmax(z), labels, n_bins=15)
    assert len(rel["bins"]) == 15
    assert sum(b["n"] for b in rel["bins"]) == 1000


@pytest.mark.parametrize("n", [1, 7, 100, 1001])
def test_threshold_keeps_95_percent(n):
    rng = np.random.default_rng(n)
    scores = rng.random(n)
    t = C.keep_threshold(scores, 0.95)
    kept = (scores >= t).mean()
    assert kept >= 0.95
    # and it is the tightest such threshold: one step stricter drops below 95 %
    assert (scores > t).mean() < 0.95


def test_threshold_with_ties_never_keeps_less():
    scores = np.array([0.5] * 50 + [0.9] * 50)
    t = C.keep_threshold(scores, 0.95)
    assert t == 0.5 and (scores >= t).mean() == 1.0


def test_calibrate_threshold_keeps_95_percent_of_val_after_temperature():
    rng = np.random.default_rng(5)
    z, labels = _sample(rng, 2_000, 4)
    r = C.calibrate(z * 2.0, labels, keep=0.95)
    msp = softmax(z * 2.0, r["temperature"]).max(axis=1)
    assert (msp >= r["unknown_threshold"]["value"]).mean() >= 0.95
    assert r["unknown_threshold"]["val_kept"] >= 0.95
    assert r["energy_threshold"]["val_kept"] >= 0.95


def test_energy_score_is_logsumexp():
    z = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert np.allclose(C.energy_score(z), [np.log(2), 10 + np.log1p(np.exp(-10))])


def test_coverage_accuracy_ends_at_full_coverage():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 0])
    curve = C.coverage_accuracy(scores, correct, points=4)
    assert curve == [[0.25, 1.0], [0.5, 1.0], [0.75, pytest.approx(2 / 3)], [1.0, 0.5]]


def test_temperature_needs_windows():
    with pytest.raises(ValueError):
        C.fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=int))
