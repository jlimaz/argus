"""Training pieces that need no torch: schedules, early stopping, device choice, run folder, CLI dispatch."""

import json
import sys
import tomllib
import types
from pathlib import Path

import numpy as np
import pytest

from mcls import config as config_module
from mcls.train import runs, schedules
from mcls.train.__main__ import main
from mcls.train.device import pick_device
from mcls.train.stopping import EarlyStopper, macro_f1

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# --- schedules -------------------------------------------------------------


def test_onecycle_warms_up_then_decays():
    total, peak = 1000, 1e-3
    lrs = [schedules.lr_at(s, total, peak, "onecycle") for s in range(total)]
    assert lrs[0] == pytest.approx(peak / 25)
    top = int(np.argmax(lrs))
    assert top == 300 and lrs[top] == pytest.approx(peak)
    assert all(a <= b for a, b in zip(lrs[:top], lrs[1:top + 1]))
    assert all(a >= b for a, b in zip(lrs[top:], lrs[top + 1:]))
    assert lrs[-1] == pytest.approx(peak / 25 / 1e4)


def test_cosine_and_constant():
    assert schedules.lr_at(0, 100, 0.1, "cosine") == pytest.approx(0.1)
    assert schedules.lr_at(99, 100, 0.1, "cosine") == pytest.approx(0.0, abs=1e-12)
    assert schedules.lr_at(50, 101, 0.1, "cosine") == pytest.approx(0.05)
    assert {schedules.lr_at(s, 10, 0.3, "constant") for s in range(10)} == {0.3}


def test_schedule_clamps_steps_and_rejects_bad_input():
    assert schedules.lr_at(10_000, 100, 0.1, "cosine") == pytest.approx(0.0, abs=1e-12)
    assert schedules.lr_at(0, 1, 0.1, "onecycle") > 0
    with pytest.raises(ValueError):
        schedules.lr_at(0, 10, 0.1, "step")
    with pytest.raises(ValueError):
        schedules.lr_at(0, 0, 0.1, "cosine")


def test_lambda_is_ratio_to_peak():
    fn = schedules.lambda_for(100, 2e-3, "onecycle")
    assert fn(30) == pytest.approx(1.0)
    assert fn(0) == pytest.approx(1 / 25)


def test_schedule_names_match_config_schema():
    assert schedules.SCHEDULES == config_module.SCHEDULES


# --- early stopping --------------------------------------------------------


def test_macro_f1_by_hand():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    # class 0: tp 1, support 2, predicted 2 -> 0.5; class 1: tp 2, 2, 3 -> 0.8; class 2: 0
    assert macro_f1(y_true, y_pred, 4) == pytest.approx((0.5 + 0.8 + 0.0) / 3)
    assert macro_f1(y_true, y_true, 3) == 1.0
    assert macro_f1(np.array([]), np.array([]), 3) == 0.0


def test_early_stopper_patience():
    stopper = EarlyStopper(patience=2)
    assert stopper.step(0.5) and not stopper.should_stop
    assert not stopper.step(0.4)
    assert stopper.step(0.6)
    assert not stopper.step(0.6) and not stopper.should_stop
    assert not stopper.step(0.1) and stopper.should_stop
    restored = EarlyStopper(patience=2)
    restored.load_state_dict(stopper.state_dict())
    assert restored.best == 0.6 and restored.should_stop


def test_early_stopper_zero_patience_never_stops():
    stopper = EarlyStopper(patience=0)
    for _ in range(20):
        stopper.step(0.0)
    assert not stopper.should_stop


# --- device ----------------------------------------------------------------


def test_pick_device():
    assert pick_device("auto", True, True) == "cuda"
    assert pick_device("auto", False, True) == "mps"
    assert pick_device("auto", False, False) == "cpu"
    assert pick_device("cpu", True, True) == "cpu"
    with pytest.raises(RuntimeError):
        pick_device("cuda", False, True)
    with pytest.raises(RuntimeError):
        pick_device("mps", True, False)
    with pytest.raises(ValueError):
        pick_device("tpu", True, True)


# --- run folder ------------------------------------------------------------


def test_toml_round_trip_of_every_shipped_config():
    for path in sorted(CONFIGS.glob("*.toml")):
        if path.name in ("taxonomy.toml", "catalog.toml", "band_priors.toml"):
            continue
        config = config_module.load(path)
        assert tomllib.loads(runs.dumps_toml(config)) == config, path.name


def test_toml_writer_escapes_strings():
    config = {"run": {"name": 'a "quoted"\\ name\tö', "seed": 3, "x": 1e-3, "flag": False, "l": [0.0, 30.0]}}
    assert tomllib.loads(runs.dumps_toml(config)) == config


def test_neural_configs_validate():
    for name in ("cnn1d", "cnn2d", "fused"):
        config = config_module.load(CONFIGS / f"{name}.toml")
        assert config["model"]["name"] == name
        assert config["optim"]["batch_size"] == 256 and config["optim"]["label_smoothing"] == 0.1


def test_overrides_and_run_dir():
    config = config_module.load(CONFIGS / "cnn1d.toml")
    changed = runs.apply_overrides(config, seed=3, device="cpu", out_dir="/tmp/x")
    assert changed["run"]["name"] == "cnn1d_v1_s3" and changed["run"]["seed"] == 3
    assert runs.run_dir_for(changed) == Path("/tmp/x/cnn1d_v1_s3")
    assert config["run"]["name"] == "cnn1d_v1"  # original untouched


def test_write_predictions_contract(tmp_path):
    path = tmp_path / "predictions_val.npz"
    runs.write_predictions(path, ["w0", "w1"], ["r0", "r0"], [1, 0],
                           np.array([[0.1, 0.9], [2.0, -1.0]]), [None, 12.5], num_classes=2)
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == set(runs.PREDICTION_KEYS)
        assert data["window_id"].tolist() == ["w0", "w1"]
        assert data["label"].dtype == np.int64 and data["logits"].dtype == np.float32
        assert data["logits"].shape == (2, 2)
        assert np.isnan(data["snr_db_est"][0]) and data["snr_db_est"][1] == np.float32(12.5)


def test_write_predictions_rejects_bad_shapes(tmp_path):
    with pytest.raises(ValueError):
        runs.write_predictions(tmp_path / "p.npz", ["w0"], ["r0"], [0], np.zeros((1, 3)), [None], num_classes=2)
    with pytest.raises(ValueError):
        runs.write_predictions(tmp_path / "p.npz", ["w0"], ["r0"], [5], np.zeros((1, 2)), [None], num_classes=2)


# --- CLI dispatch ----------------------------------------------------------


def _config_file(tmp_path, model: str, **optim) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"fake": 1}\n')
    config = {
        "run": {"name": f"{model}_test", "seed": 1, "out_dir": str(tmp_path / "runs"), "device": "cpu"},
        "data": {"manifest": str(manifest), "classes": "v1"},
        "model": {"name": model},
    }
    if optim:
        config["optim"] = optim
    path = tmp_path / f"{model}.toml"
    path.write_text(runs.dumps_toml(config))
    return path


@pytest.fixture
def fake_targets(monkeypatch):
    calls = []
    baselines = types.ModuleType("mcls.baselines")
    baselines.train = lambda config, run_dir: calls.append(("baseline", config, run_dir))
    loop = types.ModuleType("mcls.train.loop")
    loop.fit = lambda config, run_dir, resume=False: calls.append(("neural", config, run_dir, resume))
    monkeypatch.setitem(sys.modules, "mcls.baselines", baselines)
    monkeypatch.setitem(sys.modules, "mcls.train.loop", loop)
    return calls


@pytest.mark.parametrize("model", ["softmax_regression", "lightgbm", "knn", "majority"])
def test_baselines_go_to_engineer_6(tmp_path, fake_targets, model):
    assert main(["--config", str(_config_file(tmp_path, model))]) == 0
    kind, config, run_dir = fake_targets[0]
    assert kind == "baseline" and config["model"]["name"] == model
    assert run_dir == tmp_path / "runs" / f"{model}_test"


@pytest.mark.parametrize("model", ["cnn1d", "cnn2d", "fused"])
def test_neural_models_go_to_the_loop(tmp_path, fake_targets, model):
    assert main(["--config", str(_config_file(tmp_path, model)), "--seed", "7"]) == 0
    kind, config, run_dir, resume = fake_targets[0]
    assert kind == "neural" and not resume
    assert config["run"]["seed"] == 7 and run_dir.name == f"{model}_test_s7"


def test_run_folder_contents(tmp_path, fake_targets):
    path = _config_file(tmp_path, "cnn1d", lr=0.001, epochs=2)
    assert main(["--config", str(path)]) == 0
    run_dir = tmp_path / "runs" / "cnn1d_test"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert set(meta) == set(runs.META_KEYS)
    assert meta["classes"] == ["drone_link", "wifi", "bluetooth", "background"]
    assert meta["model"] == "cnn1d" and meta["seed"] == 1
    assert meta["manifest_sha256"] == runs.sha256_file(tmp_path / "manifest.jsonl")
    assert len(meta["manifest_sha256"]) == 64
    stored = tomllib.loads((run_dir / "config.toml").read_text())
    assert stored == config_module.load(path)
    config_module.validate(stored)


def test_refuses_to_overwrite_and_resumes(tmp_path, fake_targets):
    path = _config_file(tmp_path, "cnn1d")
    assert main(["--config", str(path)]) == 0
    assert main(["--config", str(path)]) == 2
    assert main(["--config", str(path), "--resume"]) == 0
    assert fake_targets[-1][3] is True


def test_resume_refuses_a_changed_config(tmp_path, fake_targets):
    path = _config_file(tmp_path, "cnn1d")
    assert main(["--config", str(path)]) == 0
    assert main(["--config", str(path), "--resume", "--device", "auto"]) == 2


def test_resume_without_a_run_and_for_baselines(tmp_path, fake_targets):
    assert main(["--config", str(_config_file(tmp_path, "cnn1d")), "--resume"]) == 2
    assert main(["--config", str(_config_file(tmp_path, "knn")), "--resume"]) == 2
    assert fake_targets == []


def test_bad_config_and_missing_manifest_exit_2(tmp_path, fake_targets):
    path = _config_file(tmp_path, "cnn1d")
    (tmp_path / "manifest.jsonl").unlink()
    assert main(["--config", str(path)]) == 2
    bad = tmp_path / "bad.toml"
    bad.write_text(path.read_text().replace('name = "cnn1d"', 'name = "resnet"'))
    assert main(["--config", str(bad)]) == 2
    assert fake_targets == []
