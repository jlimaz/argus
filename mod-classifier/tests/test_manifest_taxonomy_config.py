import copy
from pathlib import Path

import pytest

from mcls import config, manifest, taxonomy
from mcls.contracts import WINDOW_SAMPLES, ManifestRow

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def make_rows(recordings=20, windows=5, dataset="tampere"):
    rows = []
    for r in range(recordings):
        for w in range(windows):
            rows.append(ManifestRow(
                window_id=f"{dataset}/rec{r}/{w}", shard=f"shards/rec{r}.npy", offset=w * WINDOW_SAMPLES,
                length=WINDOW_SAMPLES, dataset=dataset, recording_id=f"rec{r}", label="drone_link",
                label_strength="strong", native_rate_hz=120e6, center_frequency_hz=2.44e9, band="ism_2400",
                is_real=True, license="CC BY 4.0", split="train",
            ))
    return rows


def test_manifest_write_read(tmp_path):
    rows = make_rows(3, 2)
    assert manifest.write(tmp_path / "m.jsonl", rows, ("drone_link",)) == 6
    assert manifest.read(tmp_path / "m.jsonl", ("drone_link",)) == rows
    assert len(manifest.digest(tmp_path / "m.jsonl")) == 16


def test_manifest_read_names_the_bad_line(tmp_path):
    path = tmp_path / "m.jsonl"
    manifest.write(path, make_rows(1, 2))
    path.write_text(path.read_text() + '{"window_id": "x"}\n')
    with pytest.raises(ValueError, match="line 3"):
        manifest.read(path)


def test_splits_are_by_recording_and_deterministic():
    rows = make_rows(200, 3)
    manifest.assign_splits(rows, seed=7)
    manifest.check_leakage(rows)
    first = [r.split for r in rows]
    manifest.assign_splits(rows, seed=7)
    assert [r.split for r in rows] == first
    shares = {s: sum(r.split == s for r in rows) / len(rows) for s in ("train", "val", "test")}
    assert 0.6 < shares["train"] < 0.8 and shares["val"] > 0.05 and shares["test"] > 0.05


def test_cross_test_holds_out_a_whole_dataset():
    rows = make_rows(10, 2, "tampere") + make_rows(10, 2, "dronerf")
    manifest.assign_splits(rows, seed=1, cross_test_datasets=["dronerf"])
    assert {r.split for r in rows if r.dataset == "dronerf"} == {"cross_test"}
    assert "cross_test" not in {r.split for r in rows if r.dataset == "tampere"}


def test_leakage_is_caught():
    rows = make_rows(2, 2)
    rows[0].split, rows[1].split = "train", "test"
    with pytest.raises(ValueError, match="more than one split"):
        manifest.check_leakage(rows)


def test_taxonomy_loads_and_labels_files():
    tax = taxonomy.load()
    assert tax.classes("v1") == ("drone_link", "wifi", "bluetooth", "background")
    assert "gnss_jammer" in tax.classes("v2")
    # Exact labels and strengths are Engineer 1's to decide (tests/test_taxonomy_*.py);
    # here only check that a known file resolves to a valid class and strength.
    mini = tax.label_for("tampere", "DJI_mavic_mini_2G.bin")
    assert mini.label in tax.all_classes and mini.strength in ("strong", "weak")
    mambo = tax.label_for("tampere", "Parrot_mambo_control_2G.bin")
    assert mambo.strength == "weak" and "unverified" in mambo.note
    assert tax.label_for("dronerf", "00000L_3.csv").label == "background"
    assert tax.label_for("nist_wifi_bt", "2_4ghz_bluetooth.h5").label == "bluetooth"


def test_taxonomy_refuses_unknown_class():
    with pytest.raises(ValueError, match="unknown class"):
        taxonomy.Taxonomy({
            "classes": {"v1": ["a"]},
            "datasets": {"d": {"license": "x", "is_real": True, "default_label": "b", "default_strength": "strong"}},
        })


@pytest.mark.parametrize("name", ["cnn1d.toml", "baseline.toml"])
def test_shipped_configs_are_valid(name):
    config.load(CONFIGS / name)


def test_config_errors_are_specific():
    good = config.load(CONFIGS / "cnn1d.toml")
    cases = [
        (lambda c: c.pop("model"), "missing \\[model\\]"),
        (lambda c: c["optim"].update(lr=-1), "optim.lr must be positive"),
        (lambda c: c["optim"].update(epochs=True), "got a boolean"),
        (lambda c: c["optim"].update(learning_rate=1e-3), "unknown key optim.learning_rate"),
        (lambda c: c["model"].update(name="transformer"), "model.name must be one of"),
        (lambda c: c["data"].update(classes="v3"), "data.classes"),
        (lambda c: c["augment"].update(snr_db_range=[30, 0]), "snr_db_range"),
    ]
    for change, message in cases:
        broken = copy.deepcopy(good)
        change(broken)
        with pytest.raises(ValueError, match=message):
            config.validate(broken)


def test_named_class_subset_v2a():
    tax = taxonomy.load()
    assert tax.classes("v2a") == ("drone_link", "wifi", "bluetooth", "background", "gnss_jammer", "lora")
    assert "v2a" in tax.class_set_names
    with pytest.raises(ValueError, match="known: v1, v2"):
        tax.classes("v9")
    good = config.load(CONFIGS / "cnn2d.toml")
    good["data"]["classes"] = "v2a"
    config.validate(good)


def test_class_subset_must_use_known_classes():
    with pytest.raises(ValueError, match="must list distinct v1/v2 classes"):
        taxonomy.Taxonomy({"classes": {"v1": ["a"], "v2": ["b"], "mix": ["a", "c"]}, "datasets": {}})
