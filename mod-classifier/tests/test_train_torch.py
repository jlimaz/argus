"""Training tests that need torch (skipped on the laptop, run on CRC). Tiny CPU runs only."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from mcls import models  # noqa: E402
from mcls.contracts import WINDOW_SAMPLES  # noqa: E402
from mcls.train import checkpoint as ckpt  # noqa: E402
from mcls.train import loop, runs, schedules  # noqa: E402
from mcls.train.adabn import adapt_batchnorm  # noqa: E402
from mcls.train.ema import EMA  # noqa: E402
from mcls.train.losses import classification_loss, family_loss  # noqa: E402
from mcls.train.optimizers import make_optimizer, make_scheduler  # noqa: E402

CLASSES = ["tone_lo", "tone_hi", "noise"]


# --- losses ----------------------------------------------------------------


def test_ce_matches_torch_with_smoothing_and_weights():
    torch.manual_seed(0)
    logits, target = torch.randn(6, 3), torch.tensor([0, 1, 2, 0, 1, 2])
    weight = torch.tensor([1.0, 0.5, 1.0, 0.5, 1.0, 1.0])
    per_row = F.cross_entropy(logits, target, reduction="none", label_smoothing=0.1)
    expected = (per_row * weight).sum() / weight.sum()
    assert torch.allclose(classification_loss(logits, target, weight, 0.1), expected)
    assert torch.allclose(classification_loss(logits, target, None, 0.1), per_row.mean())


def test_focal_gamma_zero_is_cross_entropy_and_downweights_easy_rows():
    torch.manual_seed(1)
    logits, target = torch.randn(8, 4), torch.randint(0, 4, (8,))
    ce = classification_loss(logits, target, label_smoothing=0.1, kind="cross_entropy")
    assert torch.allclose(classification_loss(logits, target, label_smoothing=0.1, kind="focal", focal_gamma=0.0), ce,
                          atol=1e-6)
    assert classification_loss(logits, target, label_smoothing=0.1, kind="focal", focal_gamma=2.0) < ce
    with pytest.raises(ValueError):
        classification_loss(logits, target, kind="hinge")


def test_loss_names_match_config_schema():
    from mcls import config as config_module
    from mcls.train import losses

    assert losses.LOSSES == config_module.LOSSES


def test_family_loss_is_masked():
    logits = torch.randn(4, 5, requires_grad=True)
    none_known = family_loss(logits, torch.full((4,), -1))
    assert none_known.item() == 0.0
    none_known.backward()
    family = torch.tensor([2, -1, 0, -1])
    expected = F.cross_entropy(logits[[0, 2]], family[[0, 2]])
    assert torch.allclose(family_loss(logits, family), expected)


# --- optimizers, schedules, EMA --------------------------------------------


def test_optimizers_follow_plan_and_skip_decay_on_norms():
    model = models.build("cnn1d", 3, width=4)
    adamw = make_optimizer(model, {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-2})
    assert isinstance(adamw, torch.optim.AdamW)
    decay, no_decay = adamw.param_groups
    assert decay["weight_decay"] == 1e-2 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim < 2 for p in no_decay["params"])
    sgd = make_optimizer(model, {"name": "sgd", "lr": 0.1, "momentum": 0.9})
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["nesterov"] and sgd.param_groups[0]["momentum"] == 0.9


@pytest.mark.parametrize("kind", schedules.SCHEDULES)
def test_scheduler_follows_lr_at(kind):
    model = torch.nn.Linear(2, 2)
    optim = {"name": "adamw", "lr": 1e-3, "schedule": kind}
    optimizer = make_optimizer(model, optim)
    scheduler = make_scheduler(optimizer, optim, total_steps=20)
    for step in range(20):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(schedules.lr_at(step, 20, 1e-3, kind))
        optimizer.step()
        scheduler.step()


def test_ema_update():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    ema = EMA(model, decay=0.5)
    before = ema.module[0].weight.clone()
    with torch.no_grad():
        model[0].weight.add_(2.0)
    model(torch.randn(4, 2))  # bumps num_batches_tracked
    ema.update(model)
    assert torch.allclose(ema.module[0].weight, 0.5 * before + 0.5 * model[0].weight)
    assert ema.module[1].num_batches_tracked.item() == model[1].num_batches_tracked.item()
    assert not any(p.requires_grad for p in ema.module.parameters())


# --- AdaBN -----------------------------------------------------------------


def test_adabn_reestimates_batchnorm_statistics_only():
    model = models.build("cnn1d", 3, width=4).eval()
    weights = {k: v.clone() for k, v in model.state_dict().items() if "running" not in k and "num_batches" not in k}
    stem_bn = model.backbone.stem[1]
    assert torch.all(stem_bn.running_mean == 0)
    batches = [np.random.default_rng(i).standard_normal((4, 2, WINDOW_SAMPLES)).astype(np.float32) + 3.0
               for i in range(3)]
    adapt_batchnorm(model, batches)
    assert not model.training
    assert not torch.all(stem_bn.running_mean == 0)
    assert stem_bn.momentum == 0.1  # restored
    for key, value in model.state_dict().items():
        if key in weights:
            assert torch.equal(value, weights[key]), key
    with pytest.raises(ValueError):
        adapt_batchnorm(model, [np.zeros((1, 2, WINDOW_SAMPLES), dtype=np.float32)])


# --- end to end on fake data -----------------------------------------------


class ToyDataset(torch.utils.data.Dataset):
    """Three separable classes, in Engineer 5's item format."""

    def __init__(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        self.labels = np.arange(n) % 3
        t = np.arange(WINDOW_SAMPLES)
        self.items = []
        for i, label in enumerate(self.labels):
            noise = (rng.standard_normal(WINDOW_SAMPLES) + 1j * rng.standard_normal(WINDOW_SAMPLES)) * 0.3
            if label == 2:
                z = noise / 0.3
            else:
                z = np.exp(2j * np.pi * (0.05 if label == 0 else 0.3) * t + 1j * rng.uniform(0, 6.28)) + noise
            z = (z / np.sqrt(np.mean(np.abs(z) ** 2))).astype(np.complex64)
            self.items.append(models.iq_array(z[None])[0])
        self.rows = [SimpleNamespace(window_id=f"w{seed}_{i}", recording_id=f"r{i % 4}",
                                     snr_db_est=None if i % 2 else 10.0) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {"iq": self.items[i], "label": np.int64(self.labels[i]),
                "weight": np.float32(1.0 if i % 2 else 0.5), "index": i}


class FamilyToyDataset(ToyDataset):
    """Adds a family label to every third row; the rest are unknown (-1)."""

    def __getitem__(self, i):
        item = super().__getitem__(i)
        item["family"] = np.int64(1 if i % 3 == 0 else -1)
        return item


class ToyBalancedSampler:
    def __init__(self, labels, seed):
        self.n, self.seed = len(labels), seed

    def epoch(self, e):
        return np.random.default_rng([self.seed, e, 99]).permutation(self.n)


def _toy_data():
    sets = {"train": ToyDataset(24, 0), "val": ToyDataset(9, 1), "test": ToyDataset(6, 2),
            "generalization": ToyDataset(4, 3)}
    return loop.RunData(datasets=sets, rows={k: v.rows for k, v in sets.items()},
                        train_labels=sets["train"].labels, sampler_factory=ToyBalancedSampler)


def _toy_run(tmp_path, run_name: str, model: str = "cnn1d", epochs: int = 2, **optim):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n")
    config = {
        "run": {"name": run_name, "seed": 5, "out_dir": str(tmp_path), "device": "cpu"},
        "data": {"manifest": str(manifest), "classes": "v1"},
        "model": {"name": model, "width": 4, "dropout": 0.1},
        "optim": {"name": "adamw", "lr": 3e-3, "epochs": epochs, "batch_size": 8, "ema_decay": 0.9,
                  "early_stop_patience": 0, **optim},
    }
    run_dir = tmp_path / run_name
    runs.prepare(config, run_dir, CLASSES)
    return config, run_dir


@pytest.mark.parametrize("model", models.NEURAL_MODELS)
def test_fit_writes_the_run_folder(tmp_path, model):
    config, run_dir = _toy_run(tmp_path, f"toy_{model}", model)
    loop.fit(config, run_dir, data=_toy_data())
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [h["epoch"] for h in history] == [0, 1]
    payload = ckpt.load_best(run_dir / ckpt.BEST)
    assert payload["classes"] == CLASSES and "temperature" not in payload and payload["model"] == model
    assert payload["manifest_sha256"] == json.loads((run_dir / "meta.json").read_text())["manifest_sha256"]
    assert (run_dir / ckpt.LAST).is_file()
    for split, n, seed in (("val", 9, 1), ("test", 6, 2), ("generalization", 4, 3)):
        with np.load(run_dir / f"predictions_{split}.npz", allow_pickle=False) as data:
            assert set(data.files) == set(runs.PREDICTION_KEYS)
            assert data["logits"].shape == (n, 3) and data["label"].dtype == np.int64
            assert data["window_id"][0] == f"w{seed}_0"
            assert np.isnan(data["snr_db_est"][1]) and data["snr_db_est"][0] == 10.0
    assert not (run_dir / "predictions_cross_test.npz").exists()
    assert not (run_dir / "metrics.json").exists()


def test_fit_with_aux_head_sgd_and_focal(tmp_path):
    config, run_dir = _toy_run(tmp_path, "toy_aux", epochs=1, name="sgd", lr=0.01, momentum=0.9,
                               schedule="cosine", loss="focal")
    config["model"]["aux_head"] = True
    data = _toy_data()
    data.datasets["train"] = FamilyToyDataset(24, 0)
    loop.fit(config, run_dir, data=data)
    payload = ckpt.load_best(run_dir / ckpt.BEST)
    assert payload["model_kwargs"]["num_families"] == len(models.FAMILIES)
    assert payload["family_classes"] == list(models.FAMILIES)


def _best_state(run_dir):
    return ckpt.load_best(run_dir / ckpt.BEST)["state_dict"]


def _assert_same_state(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key].float(), b[key].float(), atol=1e-6), key


def test_fit_is_seed_stable(tmp_path):
    config_a, dir_a = _toy_run(tmp_path, "seed_a")
    config_b, dir_b = _toy_run(tmp_path, "seed_b")
    loop.fit(config_a, dir_a, data=_toy_data())
    loop.fit(config_b, dir_b, data=_toy_data())
    _assert_same_state(_best_state(dir_a), _best_state(dir_b))
    with np.load(dir_a / "predictions_val.npz") as a, np.load(dir_b / "predictions_val.npz") as b:
        np.testing.assert_allclose(a["logits"], b["logits"], atol=1e-5)


def test_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    config_full, dir_full = _toy_run(tmp_path, "full", epochs=3)
    loop.fit(config_full, dir_full, data=_toy_data())

    config_cut, dir_cut = _toy_run(tmp_path, "cut", epochs=3)
    real_save = ckpt.save_last

    class Stop(Exception):
        pass

    def save_then_die(path, state):
        real_save(path, state)
        if state["epoch"] == 0:
            raise Stop

    monkeypatch.setattr(ckpt, "save_last", save_then_die)
    with pytest.raises(Stop):
        loop.fit(config_cut, dir_cut, data=_toy_data())
    monkeypatch.setattr(ckpt, "save_last", real_save)
    loop.fit(config_cut, dir_cut, resume=True, data=_toy_data())

    epochs = [json.loads(line)["epoch"] for line in (dir_cut / "history.jsonl").read_text().splitlines()]
    assert epochs == [0, 1, 2]
    full_last = ckpt.load_last(dir_full / ckpt.LAST)
    cut_last = ckpt.load_last(dir_cut / ckpt.LAST)
    _assert_same_state(full_last["model"], cut_last["model"])
    _assert_same_state(full_last["ema"], cut_last["ema"])


def test_early_stopping_stops(tmp_path, monkeypatch):
    config, run_dir = _toy_run(tmp_path, "stop", epochs=10, early_stop_patience=1)
    monkeypatch.setattr(loop, "macro_f1", lambda *args: 0.5)  # never improves after epoch 0
    loop.fit(config, run_dir, data=_toy_data())
    history = (run_dir / "history.jsonl").read_text().splitlines()
    assert len(history) == 2
    assert ckpt.load_last(run_dir / ckpt.LAST)["finished"] is True
