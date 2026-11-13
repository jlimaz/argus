"""`python -m mcls.train` handing a baseline config to Engineer 6's real entry point (no torch)."""

import json

import numpy as np

from mcls.train import runs
from mcls.train.__main__ import main
from test_baselines_run import build_fixture  # Engineer 6's tiny shard + manifest fixture


def test_majority_baseline_end_to_end(tmp_path):
    manifest_path, _ = build_fixture(tmp_path)
    config = {
        "run": {"name": "maj", "seed": 2, "out_dir": str(tmp_path / "runs"), "device": "cpu"},
        "data": {"manifest": str(manifest_path), "classes": "v1"},
        "model": {"name": "majority"},
    }
    path = tmp_path / "majority.toml"
    path.write_text(runs.dumps_toml(config))
    assert main(["--config", str(path)]) == 0
    run_dir = tmp_path / "runs" / "maj"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["model"] == "majority" and meta["manifest_sha256"] == runs.sha256_file(manifest_path)
    for split in ("val", "test"):
        with np.load(run_dir / f"predictions_{split}.npz", allow_pickle=False) as data:
            assert set(data.files) == set(runs.PREDICTION_KEYS)
            assert data["logits"].shape[1] == len(meta["classes"])
