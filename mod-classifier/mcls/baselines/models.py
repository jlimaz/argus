"""Baseline models on window features (PLAN.md section 5.1).

Every model has `fit(X, y, X_val, y_val, sample_weight=None)` and `logits(X)`,
which returns (n, num_classes) float32 scores whose softmax is the class
probability. X is already standardized by the caller.

- MajorityClass: the floor. Logits are the log class frequencies of train.
- SoftmaxRegression: multinomial logistic regression trained by minibatch
  gradient descent in numpy (cross-entropy, L2, early stopping on validation
  macro-F1).
- KNN and LightGBM: scikit-learn and lightgbm, imported only when fitted, so
  this module imports on a machine that has neither.

Weights and class balance differ by model:

- SoftmaxRegression weighs each window by its sample weight (weak labels 0.5)
  times a class-balancing weight, so every class carries the same total.
- LightGBM uses the sample weights and class_weight="balanced".
- KNN IGNORES both: every neighbour votes equally, weak-label windows count
  as much as strong ones, and a large class outvotes a small one near a
  boundary. With optim.balanced_sampler = true, KNN instead fits on a seeded
  subsample with the same number of windows per class (the smallest class's
  count); weak labels are still not down-weighted.
- MajorityClass predicts the most frequent train class by count.
"""

from __future__ import annotations

import math

import numpy as np

#: Logit given to a class the model never saw in training (softmax weight about e-30 of the others).
ABSENT_LOGIT = -30.0


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-F1 over the classes present in y_true or y_pred."""
    if len(y_true) == 0:
        return 0.0
    scores = []
    for k in range(num_classes):
        true_k, pred_k = y_true == k, y_pred == k
        if not true_k.any() and not pred_k.any():
            continue
        tp = np.count_nonzero(true_k & pred_k)
        denominator = np.count_nonzero(true_k) + np.count_nonzero(pred_k)
        scores.append(2 * tp / denominator)
    return float(np.mean(scores)) if scores else 0.0


def balanced_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-sample weight n / (K_present * count(class)): each present class carries the same total."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    present = np.count_nonzero(counts)
    per_class = np.where(counts > 0, len(y) / (present * np.maximum(counts, 1)), 0.0)
    return per_class[y]


def balanced_subsample(y: np.ndarray, seed: int) -> np.ndarray:
    """Sorted indices with the same number of rows per present class (the smallest class's count), seeded."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    size = int(counts.min()) if len(counts) else 0
    picked = [rng.choice(np.flatnonzero(y == label), size, replace=False) for label in labels]
    return np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)


def _expand(scores: np.ndarray, present: np.ndarray, num_classes: int) -> np.ndarray:
    """Place scores for the `present` class indices into a full (n, K) array."""
    out = np.full((len(scores), num_classes), ABSENT_LOGIT, dtype=np.float64)
    out[:, present] = scores
    return out.astype(np.float32)


class MajorityClass:
    name = "majority"

    def __init__(self, num_classes: int, seed: int = 0, params: dict | None = None):
        self.num_classes = num_classes
        self.log_prior = np.zeros(num_classes)

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        counts = np.bincount(y, minlength=self.num_classes).astype(np.float64)
        # Laplace smoothing keeps every logit finite; the largest count still wins.
        self.log_prior = np.log((counts + 1) / (counts.sum() + self.num_classes))
        return self

    def logits(self, X) -> np.ndarray:
        return np.tile(self.log_prior, (len(X), 1)).astype(np.float32)


class SoftmaxRegression:
    """Multinomial logistic regression, p = softmax(X W + b), trained by minibatch gradient descent.

    Loss: weighted cross-entropy (optionally label-smoothed) plus (weight_decay / 2) * ||W||^2.
    Optimizer: SGD with momentum (Nesterov) or AdamW. The learning rate follows the
    schedule over all steps. After each epoch the validation macro-F1 is measured;
    the best epoch's weights are kept, and training stops after `early_stop_patience`
    epochs without improvement.
    """

    name = "softmax_regression"

    def __init__(self, num_classes: int, seed: int = 0, params: dict | None = None):
        p = params or {}
        self.num_classes = num_classes
        self.seed = seed
        self.optimizer = p.get("name", "sgd")
        self.lr = float(p.get("lr", 0.1))
        self.momentum = float(p.get("momentum", 0.9))
        self.weight_decay = float(p.get("weight_decay", 1e-4))
        self.schedule = p.get("schedule", "cosine")
        self.epochs = int(p.get("epochs", 50))
        self.batch_size = int(p.get("batch_size", 1024))
        self.label_smoothing = float(p.get("label_smoothing", 0.0))
        self.patience = int(p.get("early_stop_patience", 10))
        self.balanced = bool(p.get("balanced", True))
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.history: list[dict] = []
        self.best_epoch = -1

    # -- maths ---------------------------------------------------------------

    def _scores(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        return X @ W + b

    def loss(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        """Mean weighted cross-entropy plus the L2 term, at the current weights."""
        return self._loss_grad(X, y, self._weights(y, sample_weight), self.W, self.b)[0]

    def _targets(self, y: np.ndarray) -> np.ndarray:
        K = self.num_classes
        t = np.full((len(y), K), self.label_smoothing / K)
        t[np.arange(len(y)), y] += 1.0 - self.label_smoothing
        return t

    def _loss_grad(self, X, y, w, W, b):
        scores = self._scores(X, W, b)
        z = scores - scores.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        t = self._targets(y)
        w = w / w.sum()
        loss = float(-(w * (t * log_p).sum(axis=1)).sum() + 0.5 * self.weight_decay * np.sum(W * W))
        delta = (np.exp(log_p) - t) * w[:, None]
        grad_W = X.T @ delta + self.weight_decay * W
        grad_b = delta.sum(axis=0)
        return loss, grad_W, grad_b

    def _weights(self, y, sample_weight):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if self.balanced:
            w = w * balanced_weights(y, self.num_classes)
        return w

    def _lr_at(self, step: int, total: int) -> float:
        if self.schedule == "constant" or total <= 1:
            return self.lr
        fraction = step / (total - 1)
        if self.schedule == "onecycle":
            warm = 0.1
            if fraction < warm:
                return self.lr * (0.1 + 0.9 * fraction / warm)
            fraction = (fraction - warm) / (1 - warm)
        return self.lr * 0.5 * (1 + math.cos(math.pi * fraction))

    # -- training ------------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        W = np.zeros((d, self.num_classes))
        b = np.zeros(self.num_classes)
        w_all = self._weights(y, sample_weight)
        velocity_W, velocity_b = np.zeros_like(W), np.zeros_like(b)
        m_W, m_b, v_W, v_b = np.zeros_like(W), np.zeros_like(b), np.zeros_like(W), np.zeros_like(b)
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.int64)
        batch = min(self.batch_size, n)
        steps_per_epoch = math.ceil(n / batch)
        total = steps_per_epoch * self.epochs
        best = (-1.0, W.copy(), b.copy())
        stale = 0
        step = 0
        self.history = []
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                _, gW, gb = self._loss_grad(X[idx], y[idx], w_all[idx], W, b)
                lr = self._lr_at(step, total)
                step += 1
                if self.optimizer == "adamw":
                    # Adam on the cross-entropy part; the L2 term is applied as decoupled decay.
                    gW = gW - self.weight_decay * W
                    beta1, beta2, eps = 0.9, 0.999, 1e-8
                    m_W = beta1 * m_W + (1 - beta1) * gW
                    m_b = beta1 * m_b + (1 - beta1) * gb
                    v_W = beta2 * v_W + (1 - beta2) * gW**2
                    v_b = beta2 * v_b + (1 - beta2) * gb**2
                    c1, c2 = 1 - beta1**step, 1 - beta2**step
                    W = W - lr * (m_W / c1 / (np.sqrt(v_W / c2) + eps) + self.weight_decay * W)
                    b = b - lr * (m_b / c1 / (np.sqrt(v_b / c2) + eps))
                else:
                    velocity_W = self.momentum * velocity_W + gW
                    velocity_b = self.momentum * velocity_b + gb
                    W = W - lr * (gW + self.momentum * velocity_W)
                    b = b - lr * (gb + self.momentum * velocity_b)
            train_loss = self._loss_grad(X, y, w_all, W, b)[0]
            record = {"epoch": epoch, "train_loss": train_loss}
            if has_val:
                score = macro_f1(y_val, np.argmax(self._scores(X_val, W, b), axis=1), self.num_classes)
                record["val_macro_f1"] = score
                if score > best[0]:
                    best, stale = (score, W.copy(), b.copy()), 0
                else:
                    stale += 1
            self.history.append(record)
            if has_val and stale >= self.patience:
                break
        if has_val:
            self.best_epoch = int(np.argmax([r["val_macro_f1"] for r in self.history]))
            self.W, self.b = best[1], best[2]
        else:
            self.best_epoch = len(self.history) - 1
            self.W, self.b = W, b
        return self

    def logits(self, X) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("fit the model first")
        return self._scores(np.asarray(X, dtype=np.float64), self.W, self.b).astype(np.float32)


class KNN:
    """k-nearest neighbours (scikit-learn); k picked on validation macro-F1."""

    name = "knn"
    K_CHOICES = (1, 3, 5, 9, 15, 25)

    def __init__(self, num_classes: int, seed: int = 0, params: dict | None = None):
        self.num_classes = num_classes
        self.seed = seed
        self.balanced = bool((params or {}).get("balanced_sampler", False))
        self.model = None
        self.k = None
        self.scores: dict[int, float] = {}

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        """sample_weight is ignored (neighbour votes are unweighted); see the module docstring."""
        from sklearn.neighbors import KNeighborsClassifier

        if self.balanced:
            keep = balanced_subsample(y, self.seed)
            X, y = np.asarray(X)[keep], np.asarray(y)[keep]

        choices = [k for k in self.K_CHOICES if k <= len(y)] or [1]
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if not has_val:
            choices = [min(5, len(y))]
        best = None
        for k in choices:
            model = KNeighborsClassifier(n_neighbors=k).fit(X, y)
            if has_val:
                self.scores[k] = macro_f1(y_val, model.predict(X_val), self.num_classes)
            if best is None or self.scores.get(k, 0.0) > self.scores.get(best[0], 0.0):
                best = (k, model)
        self.k, self.model = best
        return self

    def logits(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        # Neighbour votes can be exactly 0; a floor keeps the logits finite.
        return _expand(np.log(np.clip(proba, 1e-6, None)), self.model.classes_, self.num_classes)


class LightGBM:
    """Gradient-boosted trees (lightgbm); number of trees picked by early stopping on validation."""

    name = "lightgbm"

    def __init__(self, num_classes: int, seed: int = 0, params: dict | None = None):
        self.num_classes = num_classes
        self.seed = seed
        p = params or {}
        self.learning_rate = float(p.get("lr", 0.05))
        self.n_estimators = 2000
        self.patience = int(p.get("early_stop_patience", 50))
        self.model = None
        self.present = None

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        import lightgbm

        self.present = np.unique(y)
        if len(self.present) < 2:
            raise ValueError("lightgbm needs at least two classes in train")
        remap = {int(k): i for i, k in enumerate(self.present)}
        y_local = np.array([remap[int(k)] for k in y])
        self.model = lightgbm.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=31,
            class_weight="balanced",
            random_state=self.seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
        kwargs = {"sample_weight": sample_weight}
        keep = None if y_val is None else np.isin(y_val, self.present)
        if y_val is not None and keep.any():
            y_val_local = np.array([remap[int(k)] for k in y_val[keep]])
            kwargs["eval_set"] = [(X_val[keep], y_val_local)]
            kwargs["callbacks"] = [lightgbm.early_stopping(self.patience, verbose=False)]
        self.model.fit(X, y_local, **kwargs)
        return self

    def logits(self, X) -> np.ndarray:
        raw = np.asarray(self.model.predict_proba(X, raw_score=True), dtype=np.float64)
        if raw.ndim == 1:  # binary: lightgbm returns one score, the logit of the second class
            raw = np.stack([np.zeros_like(raw), raw], axis=1)
        return _expand(raw, self.present, self.num_classes)


MODELS = {cls.name: cls for cls in (MajorityClass, SoftmaxRegression, KNN, LightGBM)}


def build(name: str, num_classes: int, seed: int = 0, params: dict | None = None):
    if name not in MODELS:
        raise ValueError(f"not a baseline model: {name!r} (choose from {sorted(MODELS)})")
    return MODELS[name](num_classes, seed=seed, params=params)
