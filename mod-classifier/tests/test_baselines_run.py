"""The baseline run folder, end to end on a tiny fixture (a few dozen windows)."""

import copy
import hashlib
import json
import tomllib

import numpy as np
import pytest

from mcls import config as config_module
from mcls import manifest
from mcls.baselines import FeatureClassifier, load, train
from mcls.baselines.__main__ import main
from mcls.baselines.run import ShardReader, cap_per_class, clear_run_dir, select_rows, to_toml
from mcls.contracts import Classifier, ManifestRow, WINDOW_SAMPLES, check_probabilities

CLASSES = ("drone_link", "wifi", "bluetooth", "background")
CONTRACT_ARRAYS = {"window_id", "recording_id", "label", "logits", "snr_db_est"}


def make_window(rng, label):
    n = WINDOW_SAMPLES
    x = 0.05 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    t = np.arange(n)
    if label == "drone_link":  # narrow tone burst
        x[1000:3000] += np.exp(2j * np.pi * rng.uniform(-0.3, 0.3) * t[1000:3000])
    elif label == "wifi":  # wideband burst
        burst = rng.normal(size=2000) + 1j * rng.normal(size=2000)
        x[500:2500] += burst
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)


def build_fixture(tmp_path, per_split=(6, 3, 3), labels=("drone_link", "wifi", "background"), seed=0):
    """A shard per split and a manifest; one recording per (label, split)."""
    rng = np.random.default_rng(seed)
    rows, n_train, n_val, n_test = [], *per_split
    (tmp_path / "shards").mkdir()
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        windows = []
        for label in labels:
            for k in range(count):
                offset = len(windows) * WINDOW_SAMPLES
                windows.append(make_window(rng, label))
                rows.append(ManifestRow(
                    window_id=f"fx/{split}/{label}/{k}", shard=f"shards/{split}.npy", offset=offset,
                    length=WINDOW_SAMPLES, dataset="fx", recording_id=f"{split}-{label}", label=label,
                    label_strength="strong", native_rate_hz=20e6, center_frequency_hz=2.44e9, band="ism_2400",
                    is_real=True, license="test", split=split,
                    snr_db_est=None if label == "background" else 20.0,
                ))
        np.save(tmp_path / "shards" / f"{split}.npy", np.concatenate(windows))
    path = tmp_path / "manifest.jsonl"
    manifest.write(path, rows)
    return path, rows


def make_config(model="softmax_regression", seed=1):
    config = config_module.load(config_module.Path(__file__).resolve().parents[1] / "configs" / "baseline.toml")
    config["model"]["name"] = model
    config["run"]["seed"] = seed
    config["optim"].update({"epochs": 30, "batch_size": 8})
    return config


def test_softmax_run_folder_matches_the_contract(tmp_path):
    path, rows = build_fixture(tmp_path)
    run = tmp_path / "runs" / "r"
    train(make_config(), run, manifest=path, classes=CLASSES)

    assert {p.name for p in run.iterdir()} == {
        "config.toml", "meta.json", "model.pkl", "predictions_val.npz", "predictions_test.npz"}
    meta = json.loads((run / "meta.json").read_text())
    assert set(meta) == {"git_commit", "manifest_sha256", "seed", "classes", "model", "created"}
    assert meta["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["seed"] == 1 and meta["classes"] == list(CLASSES) and meta["model"] == "softmax_regression"
    with open(run / "config.toml", "rb") as handle:
        written = tomllib.load(handle)
    config_module.validate(written)
    assert written["data"]["manifest"] == str(path)

    for split in ("val", "test"):
        with np.load(run / f"predictions_{split}.npz") as data:
            assert set(data.files) == CONTRACT_ARRAYS
            expected = [r for r in rows if r.split == split]
            n = len(expected)
            assert data["window_id"].tolist() == [r.window_id for r in expected]
            assert data["recording_id"].tolist() == [r.recording_id for r in expected]
            assert data["label"].dtype == np.int64
            assert data["label"].tolist() == [CLASSES.index(r.label) for r in expected]
            assert data["logits"].dtype == np.float32 and data["logits"].shape == (n, len(CLASSES))
            assert data["snr_db_est"].dtype == np.float32
            snr = data["snr_db_est"]
            assert np.isnan(snr[[r.label == "background" for r in expected]]).all()
            assert (snr[[r.label != "background" for r in expected]] == 20).all()
            # Trivially separable fixture: every test window right.
            assert (np.argmax(data["logits"], axis=1) == data["label"]).all()


def test_fitted_model_is_a_contract_classifier(tmp_path):
    path, rows = build_fixture(tmp_path)
    run = tmp_path / "run"
    train(make_config(), run, manifest=path, classes=CLASSES)
    clf = load(run)
    assert isinstance(clf, Classifier) and isinstance(clf, FeatureClassifier)
    assert clf.classes == CLASSES
    test = [r for r in rows if r.split == "test"]
    windows = ShardReader(path.parent).windows(test)
    probabilities = clf.classify(windows)
    check_probabilities(probabilities, len(test), CLASSES)
    with np.load(run / "predictions_test.npz") as data:
        assert np.allclose(clf.logits(windows), data["logits"], atol=1e-5)
    with pytest.raises(ValueError):
        clf.classify(windows.astype(np.complex128))


def test_runs_are_deterministic_given_seed(tmp_path):
    path, _ = build_fixture(tmp_path)
    outputs = []
    for name, seed in (("a", 3), ("b", 3), ("c", 4)):
        train(make_config(seed=seed), tmp_path / name, manifest=path, classes=CLASSES)
        with np.load(tmp_path / name / "predictions_val.npz") as data:
            outputs.append(data["logits"].copy())
    assert np.array_equal(outputs[0], outputs[1])
    assert not np.array_equal(outputs[0], outputs[2])


def dispatcher_meta(run, path, model="majority", seed=1):
    """config.toml and meta.json as Engineer 7's mcls.train dispatcher writes them."""
    from mcls.train.runs import prepare

    config = make_config(model=model, seed=seed)
    config["data"]["manifest"] = str(path)
    prepare(config, run, list(CLASSES))
    return config


def test_matching_dispatcher_meta_is_kept(tmp_path):
    path, _ = build_fixture(tmp_path)
    run = tmp_path / "run"
    config = dispatcher_meta(run, path)
    before = {name: (run / name).read_text() for name in ("config.toml", "meta.json")}
    train(config, run, classes=CLASSES)
    assert {name: (run / name).read_text() for name in ("config.toml", "meta.json")} == before
    with np.load(run / "predictions_test.npz") as data:
        assert len(set(np.argmax(data["logits"], axis=1).tolist())) == 1


@pytest.mark.parametrize("change", ["seed", "model", "manifest"])
def test_meta_of_another_run_is_refused(tmp_path, change):
    path, rows = build_fixture(tmp_path)
    run = tmp_path / "run"
    dispatcher_meta(run, path)
    config = make_config(model="softmax_regression" if change == "model" else "majority",
                         seed=2 if change == "seed" else 1)
    if change == "manifest":
        manifest.write(path, rows[:-1])
    with pytest.raises(FileExistsError, match="another run"):
        train(config, run, manifest=path, classes=CLASSES)
    assert not list(run.glob("predictions_*.npz"))


def test_folder_with_results_is_refused_and_overwrite_clears_it(tmp_path):
    path, _ = build_fixture(tmp_path)
    run = tmp_path / "run"
    train(make_config(model="majority"), run, manifest=path, classes=CLASSES)
    (run / "metrics.json").write_text("{}")
    # Same manifest, seed and model, but results exist: still refused.
    with pytest.raises(FileExistsError, match="already holds results"):
        train(make_config(model="majority"), run, manifest=path, classes=CLASSES)
    train(make_config(model="softmax_regression", seed=5), run, manifest=path, classes=CLASSES, overwrite=True)
    meta = json.loads((run / "meta.json").read_text())
    assert meta["model"] == "softmax_regression" and meta["seed"] == 5
    assert not (run / "metrics.json").exists()


def test_overwrite_refuses_a_folder_that_is_not_a_run(tmp_path):
    path, _ = build_fixture(tmp_path)
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "metrics.json").write_text("{}")  # a known name, but no run marker
    with pytest.raises(FileExistsError, match="does not look like a run folder"):
        train(make_config(model="majority"), folder, manifest=path, classes=CLASSES, overwrite=True)
    assert (folder / "metrics.json").exists()
    # The fixture folder itself holds a manifest and shards: never a run folder.
    with pytest.raises(FileExistsError, match="not run files"):
        clear_run_dir(tmp_path)
    assert path.exists() and (tmp_path / "shards").is_dir()


def test_overwrite_refuses_a_run_folder_holding_an_unknown_file(tmp_path):
    path, _ = build_fixture(tmp_path)
    run = tmp_path / "run"
    train(make_config(model="majority"), run, manifest=path, classes=CLASSES)
    (run / "notes.txt").write_text("keep me")
    (run / "plots").mkdir()
    with pytest.raises(FileExistsError, match="notes.txt, plots"):
        train(make_config(model="majority"), run, manifest=path, classes=CLASSES, overwrite=True)
    assert (run / "notes.txt").read_text() == "keep me" and (run / "meta.json").exists()


def test_overwrite_never_follows_symlinks(tmp_path):
    path, _ = build_fixture(tmp_path)
    run = tmp_path / "run"
    train(make_config(model="majority"), run, manifest=path, classes=CLASSES)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (run / "metrics.json").symlink_to(outside)
    with pytest.raises(FileExistsError, match="metrics.json"):
        clear_run_dir(run)
    linked = tmp_path / "linked_run"
    linked.symlink_to(run, target_is_directory=True)
    with pytest.raises(FileExistsError, match="symlink"):
        clear_run_dir(linked)
    assert outside.exists() and (run / "meta.json").exists()


def test_overwrite_deletes_only_run_files(tmp_path):
    path, _ = build_fixture(tmp_path)
    run = tmp_path / "run"
    train(make_config(model="majority"), run, manifest=path, classes=CLASSES)
    for name in ("calibration.json", "history.jsonl", "checkpoint.pt", "last.pt"):
        (run / name).write_text("x")
    removed = clear_run_dir(run)
    assert "meta.json" in removed and "predictions_val.npz" in removed and "last.pt" in removed
    assert run.is_dir() and not list(run.iterdir())


def test_cli_refuses_an_existing_run_unless_overwrite(tmp_path):
    path, _ = build_fixture(tmp_path, per_split=(3, 2, 2))
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(to_toml(make_config()))
    out = tmp_path / "runs" / "cli"
    args = ["--config", str(config_path), "--manifest", str(path), "--out", str(out), "--model", "majority"]
    assert main(args) == 0
    assert main(args) == 2
    assert main(args + ["--overwrite"]) == 0


def test_generalization_predictions_written_when_present(tmp_path):
    path, rows = build_fixture(tmp_path)
    held = [r for r in rows if r.split == "test" and r.label == "drone_link"]
    for row in held:
        row.split = "generalization"
    manifest.write(path, rows)
    run = tmp_path / "run"
    train(make_config(model="majority"), run, manifest=path, classes=CLASSES)
    with np.load(run / "predictions_generalization.npz") as data:
        assert set(data.files) == CONTRACT_ARRAYS
        assert data["window_id"].tolist() == [r.window_id for r in held]


def test_cross_test_predictions_written_when_present(tmp_path):
    path, rows = build_fixture(tmp_path)
    for row in rows:
        if row.split == "test" and row.label == "wifi":
            row.split = "cross_test"
    manifest.write(path, rows)
    run = tmp_path / "run"
    train(make_config(model="majority"), run, manifest=path, classes=CLASSES)
    with np.load(run / "predictions_cross_test.npz") as data:
        assert data["window_id"].tolist() == [r.window_id for r in rows if r.split == "cross_test"]


def test_row_selection_rules(tmp_path):
    _, rows = build_fixture(tmp_path)
    rows = copy.deepcopy(rows)
    rows[0].is_real = False                              # simulated train row
    next(r for r in rows if r.split == "test").is_real = False  # simulated test row
    rows[1].label = "lora"                                # not a v1 class
    data = {"use_simulated": False}
    picked = select_rows(rows, CLASSES, data, seed=0)
    assert rows[0] not in picked["train"] and rows[1] not in picked["train"]
    assert all(r.is_real for split in picked.values() for r in split)
    picked = select_rows(rows, CLASSES, {"use_simulated": True}, seed=0)
    assert rows[0] in picked["train"] and all(r.is_real for r in picked["test"])
    capped = select_rows(rows, CLASSES, {"max_windows_per_class": 2}, seed=0)["train"]
    assert max(sum(r.label == c for r in capped) for c in CLASSES) == 2
    assert cap_per_class(picked["train"], 2, 5) == cap_per_class(picked["train"], 2, 5)


def test_to_toml_round_trips():
    config = make_config()
    config["data"]["cross_test_datasets"] = ["dronerf", "nist \"x\""]
    config["augment"] = {"enabled": True, "snr_db_range": [-5.0, 20.0], "freq_offset_fraction": 0.01}
    assert tomllib.loads(to_toml(config)) == config


def test_cli_writes_a_run(tmp_path, capsys):
    path, _ = build_fixture(tmp_path, per_split=(3, 2, 2))
    config = make_config()
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(to_toml(config))
    out = tmp_path / "runs" / "cli"
    assert main(["--config", str(config_path), "--manifest", str(path), "--out", str(out), "--model", "majority"]) == 0
    assert json.loads((out / "meta.json").read_text())["model"] == "majority"
    assert json.loads((out / "meta.json").read_text())["classes"] == ["drone_link", "wifi", "bluetooth", "background"]
    assert (out / "predictions_val.npz").exists()


def test_cli_rejects_missing_manifest_and_non_baseline_models(tmp_path):
    config = make_config()
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(to_toml(config))
    assert main(["--config", str(config_path), "--manifest", str(tmp_path / "none.jsonl")]) == 2
    config["model"]["name"] = "cnn1d"
    config_path.write_text(to_toml(config))
    assert main(["--config", str(config_path), "--manifest", str(tmp_path / "none.jsonl")]) == 2


@pytest.mark.parametrize("model,module", [("knn", "sklearn"), ("lightgbm", "lightgbm")])
def test_library_baselines_write_runs(tmp_path, model, module):
    pytest.importorskip(module)
    path, _ = build_fixture(tmp_path)
    run = tmp_path / model
    train(make_config(model=model), run, manifest=path, classes=CLASSES)
    clf = load(run)
    assert clf.classes == CLASSES
    with np.load(run / "predictions_test.npz") as data:
        assert np.isfinite(data["logits"]).all()


def test_runs_on_a_manifest_built_by_the_data_pipeline(tmp_path):
    """Engineer 5's fake shards -> build_manifest CLI -> baseline CLI (plumbing only: random windows)."""
    from mcls.dataset import WindowStore, build_manifest
    from mcls.dataset.fakes import make_fake_shards

    make_fake_shards(tmp_path / "shards", seed=0, recordings_per_class=10, windows_per_recording=2)
    path = tmp_path / "data" / "manifest.jsonl"
    assert build_manifest.main(["--shards", str(tmp_path / "shards"), "--out", str(path), "--seed", "1",
                                "--cap-per-class", "0"]) == 0
    rows = manifest.read(path)
    assert {r.split for r in rows} >= {"train", "val", "test"}
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(to_toml(make_config()))
    out = tmp_path / "runs" / "e5"
    assert main(["--config", str(config_path), "--manifest", str(path), "--out", str(out)]) == 0
    clf = load(out)
    for split in ("val", "test"):
        expected = [r for r in rows if r.split == split]
        with np.load(out / f"predictions_{split}.npz") as data:
            assert data["window_id"].tolist() == [r.window_id for r in expected]
            windows = WindowStore(path.parent).windows(expected)
            assert np.allclose(clf.logits(windows), data["logits"], atol=1e-5)
