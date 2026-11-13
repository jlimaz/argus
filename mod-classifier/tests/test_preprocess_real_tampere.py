"""Smoke test on the real Tampere files when they are on this machine (skipped otherwise).

Looks for data/raw/tampere/<file> under MCLS_RAW, this checkout, or any parent
folder's mod-classifier/ (a git worktree sits inside the main checkout).
Processes 0.05 s of each file only. The files are read, never written.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from mcls import preprocess
from mcls.frontend import windows as W
from mcls.frontend.channelize import to_20msps
from mcls.frontend.process import check_rows, windows_from_iq
from mcls.sources import get

FILES = ["DJI_mavic_mini_2G.bin", "DJI_inspire_2_5G_1of2.bin"]
SECONDS = 0.05


def find(name: str) -> Path | None:
    candidates = []
    if os.environ.get("MCLS_RAW"):
        candidates.append(Path(os.environ["MCLS_RAW"]) / "tampere" / name)
    here = Path(__file__).resolve()
    candidates.append(here.parents[1] / "data" / "raw" / "tampere" / name)
    candidates += [p / "mod-classifier" / "data" / "raw" / "tampere" / name for p in here.parents]
    return next((c for c in candidates if c.is_file()), None)


@pytest.mark.parametrize("name", FILES)
def test_real_file(tmp_path, name):
    path = find(name)
    if path is None:
        pytest.skip(f"{name} not on this machine")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / name).symlink_to(path)
    out = tmp_path / "out"
    assert preprocess.main(["--dataset", "tampere", "--raw", str(raw), "--out", str(out),
                            "--limit-seconds", str(SECONDS)]) == 0
    stem = name[:-4]
    stats = json.loads((out / f"{stem}.stats.json").read_text())
    rows = [json.loads(line) for line in open(out / f"{stem}.s000.rows.jsonl")]
    check_rows(rows)
    shard = np.load(out / f"{stem}.s000.npy", mmap_mode="r")
    assert shard.shape == (len(rows) * 4096,) and len(rows) == stats["windows"] > 0
    assert stats["seconds_processed"] == pytest.approx(SECONDS)

    # The chunked front end finds the same bursts as detect_bursts on the one-shot 20 MS/s signal.
    rec = next(get("tampere", root=raw).recordings())
    n = int(round(SECONDS * rec.sample_rate_hz))
    iq20 = to_20msps(rec.read(0, n), rec.sample_rate_hz, stats["slice_offset_hz"])
    assert len(iq20) == stats["samples_20msps"]
    bursts = W.detect(iq20, stats["slice_center_hz"])
    assert len(bursts) == stats["bursts"]
    wins, _ = W.burst_windows(iq20, bursts)
    assert np.array_equal(W.cut(iq20, wins), np.asarray(shard).reshape(-1, 4096))

    # Inference entry point: same windows and burst ids as the shard and rows.
    result = windows_from_iq(rec.read(0, n), rec.sample_rate_hz, rec.center_frequency_hz, id_prefix=f"tampere/{stem}")
    assert np.array_equal(result.windows, np.asarray(shard).reshape(-1, 4096))
    assert result.burst_ids == [r["burst_id"] for r in rows] and all(result.burst_ids)
    assert len(result.bursts) == stats["bursts"]
