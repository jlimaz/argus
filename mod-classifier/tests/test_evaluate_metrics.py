"""Metrics against small hand-computed cases."""

import numpy as np
import pytest

from mcls.evaluate import metrics as M

CLASSES = ["a", "b", "c"]
LABELS = np.array([0, 0, 1, 1, 2, 2])
PREDS = np.array([0, 1, 1, 1, 2, 0])


def test_confusion_matrix_by_hand():
    cm = M.confusion_matrix(LABELS, PREDS, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_confusion_rejects_out_of_range():
    with pytest.raises(ValueError):
        M.confusion_matrix(np.array([0, 3]), np.array([0, 0]), 3)


def test_per_class_and_macro_f1_by_hand():
    s = M.summarize(LABELS, PREDS, CLASSES)
    pc = s["per_class"]
    # a: tp 1, predicted 2, support 2 -> P 0.5, R 0.5, F1 0.5
    assert pc["a"]["precision"] == pytest.approx(0.5) and pc["a"]["recall"] == pytest.approx(0.5)
    assert pc["a"]["f1"] == pytest.approx(0.5)
    # b: tp 2, predicted 3, support 2 -> P 2/3, R 1, F1 0.8
    assert pc["b"]["precision"] == pytest.approx(2 / 3) and pc["b"]["recall"] == 1.0
    assert pc["b"]["f1"] == pytest.approx(0.8)
    # c: tp 1, predicted 1, support 2 -> P 1, R 0.5, F1 2/3
    assert pc["c"]["f1"] == pytest.approx(2 / 3)
    assert s["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert s["accuracy"] == pytest.approx(4 / 6)


def test_never_predicted_class_has_no_precision_and_zero_f1():
    s = M.summarize(np.array([0, 1]), np.array([0, 0]), ["a", "b"])
    assert s["per_class"]["b"]["precision"] is None
    assert s["per_class"]["b"]["f1"] == 0.0
    # a: P 0.5, R 1 -> F1 2/3; b: 0 -> macro 1/3
    assert s["macro_f1"] == pytest.approx(1 / 3)


def test_macro_f1_skips_absent_classes():
    s = M.summarize(np.array([0, 0]), np.array([0, 0]), ["a", "b", "c"])
    assert s["macro_f1"] == 1.0
    assert s["macro_f1_classes"] == ["a"]


def test_empty_set():
    s = M.summarize(np.array([], dtype=int), np.array([], dtype=int), CLASSES)
    assert s["accuracy"] is None and s["macro_f1"] is None


def test_burst_averages_logits_not_votes():
    # Burst x: two windows; one weakly wrong, one strongly right -> average is right.
    logits = np.array([[0.0, 0.1], [5.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    agg = M.burst_logits(logits, labels, ["x", "x", "y"])
    assert agg["keys"].tolist() == ["x", "y"]
    assert np.allclose(agg["logits"][0], [2.5, 0.05])
    assert agg["labels"].tolist() == [0, 1]
    assert agg["windows"].tolist() == [2, 1]
    assert agg["mixed_label_bursts"] == 0


def test_burst_counts_mixed_labels():
    agg = M.burst_logits(np.zeros((3, 2)), np.array([0, 1, 1]), ["x", "x", "x"])
    assert agg["mixed_label_bursts"] == 1
    assert agg["labels"].tolist() == [1]


def test_accuracy_by_snr_bins_and_nan():
    labels = np.array([0, 0, 0, 0, 0])
    preds = np.array([0, 1, 0, 0, 1])
    snr = np.array([-3.0, -1.0, 2.0, 4.0, np.nan])
    r = M.accuracy_by_snr(labels, preds, snr, edges=[0.0])
    assert r["n_snr_unknown"] == 1
    below, above = r["bins"]
    assert below == {"lo": None, "hi": 0.0, "n": 2, "accuracy": 0.5}
    assert above == {"lo": 0.0, "hi": None, "n": 2, "accuracy": 1.0}


def test_false_alarm_with_and_without_threshold():
    # background is index 0. Three background windows: one right, one wrong
    # confidently, one wrong with low confidence.
    logits = np.array([[5.0, 0.0], [0.0, 5.0], [0.0, 0.1], [0.0, 5.0]])
    labels = np.array([0, 0, 0, 1])
    raw = M.false_alarm_rate(labels, logits, 0)
    assert raw["n_background"] == 3 and raw["rate"] == pytest.approx(2 / 3)
    gated = M.false_alarm_rate(labels, logits, 0, threshold=0.9)
    assert gated["rate"] == pytest.approx(1 / 3)
    assert M.false_alarm_rate(np.array([1]), logits[:1], 0)["rate"] is None


def test_softmax_rows_sum_to_one_and_temperature_flattens():
    z = np.array([[1.0, 2.0, 3.0]])
    p1, p2 = M.softmax(z), M.softmax(z, 2.0)
    assert np.allclose(p1.sum(), 1) and p2.max() < p1.max()
    assert np.allclose(M.logsumexp(np.array([[1000.0, 1000.0]])), 1000 + np.log(2))
