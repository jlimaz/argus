"""scripts/build_routing.py on small made-up runs: routes come from validation, test only scores them."""

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_routing.py"
spec = importlib.util.spec_from_file_location("build_routing", SCRIPT)
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)

CLASSES = ["a", "b"]


def make_run(root, name, right_in, n_per_band=300):
    """A run that is right on bands in `right_in` and wrong elsewhere, on val and test alike."""
    run = root / name
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"classes": CLASSES}))
    (run / "calibration.json").write_text(json.dumps({"temperature": 1.0, "unknown_threshold": {"value": 0.5}}))
    for split in ("val", "test"):
        ids, labels, logits = [], [], []
        for band in ("x", "y"):
            for i in range(n_per_band):
                label = i % 2
                ids.append(f"{split}/{band}/{i}")
                labels.append(label)
                guess = label if band in right_in else 1 - label
                logits.append([3.0, 0.0] if guess == 0 else [0.0, 3.0])
        np.savez(run / f"predictions_{split}.npz", window_id=np.array(ids), recording_id=np.array(ids),
                 label=np.array(labels), logits=np.array(logits, dtype=np.float32))
    return run


def write_manifest(path):
    with open(path, "w") as handle:
        for split in ("val", "test"):
            for band in ("x", "y"):
                for i in range(300):
                    handle.write(json.dumps({"window_id": f"{split}/{band}/{i}", "band": band, "bandwidth_hz": 2e6,
                                             "dataset": "d", "recording_id": f"{band}{i % 5}"}) + "\n")


def test_each_band_goes_to_the_model_that_is_right_there(tmp_path):
    runs = [make_run(tmp_path, "good_on_x", {"x"}), make_run(tmp_path, "good_on_y", {"y"})]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    result = routing.build(runs, manifest)
    assert result["routes"]["band"]["x"]["model"] == "good_on_x"
    assert result["routes"]["band"]["y"]["model"] == "good_on_y"
    assert result["test"]["route by band"]["accuracy"] == 1.0
    assert result["test"]["good_on_x alone"]["accuracy"] == 0.5
    assert result["test"]["oracle (ceiling, not achievable)"]["accuracy"] == 1.0


def test_thin_groups_fall_back_to_the_overall_best(tmp_path):
    runs = [make_run(tmp_path, "good_on_x", {"x"}), make_run(tmp_path, "good_on_y", {"y"}, n_per_band=300)]
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest)
    routing.MIN_WINDOWS, saved = 10_000, routing.MIN_WINDOWS
    try:
        result = routing.build(runs, manifest)
    finally:
        routing.MIN_WINDOWS = saved
    for entry in result["routes"]["band"].values():
        assert entry["model"] == result["fallback_model"]
        assert "too little" in entry["reason"]


def test_bandwidth_buckets():
    assert routing.bandwidth_bucket(50e3).startswith("narrow")
    assert routing.bandwidth_bucket(2e6).startswith("wide")
    assert routing.bandwidth_bucket(None) == "unknown"
