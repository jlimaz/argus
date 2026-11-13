import numpy as np
import pytest

from mcls.baselines import models


def blobs(seed=0, n=40, gap=4.0, dims=5, classes=2):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % classes
    X = rng.normal(size=(n, dims))
    X[:, 0] += gap * y
    return X, y


def test_softmax_matches_definition():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 0.0]])
    p = models.softmax(z)
    assert np.allclose(p.sum(axis=1), 1)
    assert np.allclose(p[0], np.exp(z[0]) / np.exp(z[0]).sum())
    assert np.isfinite(p).all()


def test_macro_f1():
    y = np.array([0, 0, 1, 1])
    assert models.macro_f1(y, y, 3) == 1.0
    assert models.macro_f1(y, np.array([0, 0, 0, 0]), 2) == pytest.approx((2 * 2 / 6 + 0) / 2)


def test_softmax_regression_gradient_matches_finite_differences():
    X, y = blobs(seed=1, n=12, dims=3, classes=3)
    model = models.SoftmaxRegression(3, params={"weight_decay": 0.1, "label_smoothing": 0.1})
    rng = np.random.default_rng(0)
    W, b, w = rng.normal(size=(3, 3)), rng.normal(size=3), rng.uniform(0.5, 1.0, size=12)
    _, gW, gb = model._loss_grad(X, y, w, W, b)
    eps = 1e-6
    for i, j in [(0, 0), (1, 2), (2, 1)]:
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += eps
        Wm[i, j] -= eps
        numeric = (model._loss_grad(X, y, w, Wp, b)[0] - model._loss_grad(X, y, w, Wm, b)[0]) / (2 * eps)
        assert gW[i, j] == pytest.approx(numeric, rel=1e-4, abs=1e-8)
    bp = b.copy()
    bp[1] += eps
    numeric = (model._loss_grad(X, y, w, W, bp)[0] - model._loss_grad(X, y, w, W, b)[0]) / eps
    assert gb[1] == pytest.approx(numeric, rel=1e-3, abs=1e-6)


@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_softmax_regression_reduces_loss_and_separates_two_classes(optimizer):
    X, y = blobs(seed=2, gap=10.0)
    X_val, y_val = blobs(seed=3, gap=10.0)
    params = {"name": optimizer, "lr": 0.1 if optimizer == "sgd" else 0.05, "epochs": 30, "batch_size": 8}
    model = models.SoftmaxRegression(2, seed=0, params=params)
    start = model._loss_grad(X, y, np.ones(len(y)), np.zeros((5, 2)), np.zeros(2))[0]
    model.fit(X, y, X_val, y_val)
    assert model.loss(X, y) < 0.5 * start
    losses = [r["train_loss"] for r in model.history]
    assert losses[-1] < losses[0]
    assert (np.argmax(model.logits(X_val), axis=1) == y_val).all()


def test_softmax_regression_early_stops_and_keeps_best_epoch():
    X, y = blobs(seed=4)
    model = models.SoftmaxRegression(2, params={"epochs": 200, "early_stop_patience": 3, "batch_size": 8})
    model.fit(X, y, X, y)
    assert len(model.history) < 200
    scores = [r["val_macro_f1"] for r in model.history]
    assert model.best_epoch == int(np.argmax(scores))
    assert models.macro_f1(y, np.argmax(model.logits(X), axis=1), 2) == max(scores)


def test_softmax_regression_is_deterministic_given_seed():
    X, y = blobs(seed=5, classes=3, n=30)
    params = {"epochs": 5, "batch_size": 4}
    a = models.SoftmaxRegression(3, seed=7, params=params).fit(X, y, X, y).logits(X)
    b = models.SoftmaxRegression(3, seed=7, params=params).fit(X, y, X, y).logits(X)
    c = models.SoftmaxRegression(3, seed=8, params=params).fit(X, y, X, y).logits(X)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_majority_predicts_the_most_frequent_class():
    y = np.array([2, 2, 2, 0, 1])
    model = models.MajorityClass(4).fit(np.zeros((5, 1)), y)
    logits = model.logits(np.zeros((3, 1)))
    assert logits.shape == (3, 4) and logits.dtype == np.float32
    assert (np.argmax(logits, axis=1) == 2).all()
    assert np.isfinite(logits).all()


def test_build_rejects_unknown_models():
    with pytest.raises(ValueError):
        models.build("cnn1d", 2)


def test_balanced_subsample_is_equal_per_class_and_seeded():
    y = np.array([0] * 10 + [1] * 3 + [2] * 5)
    keep = models.balanced_subsample(y, seed=1)
    assert np.bincount(y[keep]).tolist() == [3, 3, 3]
    assert np.array_equal(keep, models.balanced_subsample(y, seed=1))
    assert np.all(np.diff(keep) > 0)


def test_knn_balanced_option_fits_on_equal_classes():
    pytest.importorskip("sklearn")
    X, y = blobs(seed=10, n=40)
    X, y = X[: 25], y[: 25]  # 13 of class 0, 12 of class 1
    model = models.KNN(2, seed=0, params={"balanced_sampler": True}).fit(X, y)
    assert model.model.n_samples_fit_ == 24


def test_knn_picks_k_and_covers_absent_classes():
    pytest.importorskip("sklearn")
    X, y = blobs(seed=6)
    model = models.KNN(3).fit(X, y, *blobs(seed=7))
    logits = model.logits(X)
    assert logits.shape == (len(X), 3) and np.isfinite(logits).all()
    assert model.k in models.KNN.K_CHOICES
    assert (np.argmax(logits, axis=1) != 2).all()


def test_lightgbm_separates_two_classes():
    pytest.importorskip("lightgbm")
    X, y = blobs(seed=8, n=60)
    model = models.LightGBM(3, seed=0).fit(X, y, *blobs(seed=9, n=60))
    logits = model.logits(X)
    assert logits.shape == (len(X), 3)
    assert (np.argmax(logits, axis=1) == y).mean() > 0.9
