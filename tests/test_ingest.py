import json

import numpy as np
import pytest

from pipeline.ingest import hertz, main
from pipeline.inputs import combine_observations


def sweep_file(tmp_path):
    rng = np.random.default_rng(5)
    lines = []
    for p in range(2):
        centres = 88e6 + (np.arange(200) + 0.5) * 1e5
        db = -60 + rng.normal(0, 0.5, 200)
        db[(centres > 91.4e6) & (centres < 91.6e6)] = -30
        lines.append(f"2026-11-11, 20:0{p}:00, 88000000, 108000000, 100000.00, 10, " + ", ".join(f"{v:.2f}" for v in db))
    path = tmp_path / "fm.csv"
    path.write_text("\n".join(lines))
    return path


def test_sweep_to_data_file(tmp_path, capsys):
    out = tmp_path / "signals.jsonl"
    assert main([str(sweep_file(tmp_path)), "-o", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == 0
    assert rows[0]["source"] == "fm.csv"
    combine_observations([("signals.jsonl", out.read_text())])
    assert "wrote 1 signals" in capsys.readouterr().err


def test_prints_to_stdout_without_output(tmp_path, capsys):
    assert main([str(sweep_file(tmp_path))]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["center_frequency_hz"] > 0


def test_raw_iq_needs_rate_and_frequency(tmp_path, capsys):
    path = tmp_path / "x.cu8"
    path.write_bytes(bytes(range(256)) * 16)
    assert main([str(path)]) == 1
    assert "sample rate" in capsys.readouterr().err
    assert main([str(path), "--sample-rate", "2.4M", "--center-freq", "433.92M"]) == 0


def test_missing_file_and_bad_sweep(tmp_path, capsys):
    assert main([str(tmp_path / "nope.csv")]) == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("not, a, sweep\n")
    assert main([str(bad)]) == 1
    assert "Line 1 has 3 fields" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text, value",
    [("2.437G", 2.437e9), ("20M", 20e6), ("915e6", 915e6), ("433920000", 433_920_000), ("12.5kHz", 12_500)],
)
def test_hertz(text, value):
    assert hertz(text) == pytest.approx(value)


def test_hertz_rejects_text():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        hertz("fast")
