"""Config keys, the single temperature location, the generalization split, meta shared with Engineer 6. No torch."""

import ast
import json
import tomllib
from pathlib import Path

import pytest

from mcls import config as config_module
from mcls.baselines.run import check_run_dir, sha256_file, write_meta  # Engineer 6
from mcls.models.classifier import read_temperature
from mcls.train import runs
from mcls.train.__main__ import main
from mcls.train.options import options

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def test_options_defaults_follow_plan():
    assert options({"run": {}, "model": {"name": "cnn1d"}}) == {
        "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-2, "momentum": 0.9, "nesterov": True,
        "schedule": "onecycle", "epochs": 40, "batch_size": 256, "label_smoothing": 0.1, "grad_clip": 1.0,
        "ema_decay": 0.999, "early_stop_patience": 8, "loss": "cross_entropy", "focal_gamma": 2.0,
        "balanced_sampler": True, "aux_weight": 0.5, "aux_head": False, "width": 32, "dropout": 0.2,
    }


def test_options_read_every_new_config_key():
    config = config_module.load(CONFIGS / "cnn1d.toml")
    config["model"]["aux_head"] = True
    config["optim"].update({"name": "sgd", "loss": "focal", "focal_gamma": 1, "nesterov": False,
                            "balanced_sampler": False, "aux_weight": 0.25})
    config_module.validate(config)
    assert tomllib.loads(runs.dumps_toml(config)) == config
    opts = options(config)
    assert opts["aux_head"] is True and opts["optimizer"] == "sgd" and opts["loss"] == "focal"
    assert opts["focal_gamma"] == 1.0 and isinstance(opts["focal_gamma"], float)
    assert opts["nesterov"] is False and opts["balanced_sampler"] is False and opts["aux_weight"] == 0.25


def test_option_values_are_ones_the_schema_accepts():
    defaults = options({})
    assert defaults["loss"] in config_module.LOSSES
    assert defaults["optimizer"] in config_module.OPTIMIZERS
    assert defaults["schedule"] in config_module.SCHEDULES


def test_loop_predicts_the_generalization_split():
    tree = ast.parse((ROOT / "mcls" / "train" / "loop.py").read_text())
    value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", None) == "PREDICTION_SPLITS" for t in node.targets))
    assert ast.literal_eval(value) == ("val", "test", "cross_test", "generalization")


def test_checkpoint_module_has_no_temperature_writer():
    source = (ROOT / "mcls" / "train" / "checkpoint.py").read_text()
    assert "def set_temperature" not in source and '"temperature":' not in source


def test_read_temperature_from_calibration(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"temperature": 1.7, "classes": ["a", "b"]}))
    assert read_temperature(path, ["a", "b"]) == 1.7
    with pytest.raises(ValueError):
        read_temperature(path, ["b", "a"])
    for bad in (0, -1, "2", True, None):
        path.write_text(json.dumps({"temperature": bad}))
        with pytest.raises(ValueError):
            read_temperature(path, ["a", "b"])


def test_dispatcher_meta_matches_engineer_6_exactly(tmp_path, monkeypatch):
    import sys
    import types

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"fake": 1}\n')
    config = {
        "run": {"name": "n", "seed": 4, "out_dir": str(tmp_path / "runs"), "device": "cpu"},
        "data": {"manifest": str(manifest), "classes": "v1"},
        "model": {"name": "cnn1d"},
    }
    path = tmp_path / "c.toml"
    path.write_text(runs.dumps_toml(config))
    loop = types.ModuleType("mcls.train.loop")
    loop.fit = lambda config, run_dir, resume=False: None
    monkeypatch.setitem(sys.modules, "mcls.train.loop", loop)
    assert main(["--config", str(path)]) == 0
    run_dir = tmp_path / "runs" / "n"
    ours = json.loads((run_dir / "meta.json").read_text())
    expected = {"manifest_sha256": sha256_file(manifest), "seed": 4, "model": "cnn1d", "classes": ours["classes"]}
    assert check_run_dir(run_dir, expected) is True  # Engineer 6's check keeps our meta.json
    (tmp_path / "e6").mkdir()
    write_meta(tmp_path / "e6", expected)
    theirs = json.loads((tmp_path / "e6" / "meta.json").read_text())
    assert set(ours) == set(theirs) == set(runs.META_KEYS)
    for key in ("git_commit", "manifest_sha256", "seed", "classes", "model"):
        assert ours[key] == theirs[key], key
