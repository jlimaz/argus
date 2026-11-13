"""process_recording and `python -m mcls.preprocess`: chunking, carry-over, rows, CLI on Tampere-format files."""

import json

import numpy as np
import pytest

from mcls import preprocess
from mcls.contracts import WINDOW_SAMPLES, Recording, check_windows
from mcls.frontend.process import FrontendParams, check_rows, process_recording, stem_for


def make_recording(x, rate, label="drone_link", centre=2.44e9, name="fake.bin"):
    x = np.asarray(x, dtype=np.complex64)
    return Recording(
        dataset="fake", recording_id="rec1", source_file=name, sample_rate_hz=rate,
        center_frequency_hz=centre, num_samples=len(x), label=label, label_strength="strong",
        is_real=True, license="CC BY 4.0", read=lambda s, c: x[s : s + c].copy(), emitter_id="e1",
    )


def signal(rate, total, spans, offset_hz=0.0, snr_db=25.0, seed=0):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(total) + 1j * rng.standard_normal(total)) / np.sqrt(2) * 10 ** (-snr_db / 20)
    n = np.arange(total)
    for a, b in spans:
        s = rng.standard_normal(b - a) + 1j * rng.standard_normal(b - a)
        s = np.convolve(s, np.ones(8) / np.sqrt(8), mode="same")
        x[a:b] += s * np.exp(2j * np.pi * offset_hz / rate * n[a:b])
    return x.astype(np.complex64)


def load_outputs(out, stem):
    rows, data = [], []
    for rows_file in sorted(out.glob(f"{stem}.s*.rows.jsonl")):
        shard = np.load(rows_file.with_name(rows_file.name.replace(".rows.jsonl", ".npy")), mmap_mode="r")
        for line in open(rows_file):
            row = json.loads(line)
            rows.append(row)
            data.append(np.array(shard[row["offset"] : row["offset"] + row["length"]]))
    return rows, (np.stack(data) if data else np.zeros((0, WINDOW_SAMPLES), np.complex64))


def test_rows_are_valid_and_windows_normalized(tmp_path):
    rate = 40e6
    x = signal(rate, 2_000_000, [(200_000, 600_000), (1_500_000, 1_560_000)], offset_hz=12e6)
    stats = process_recording(make_recording(x, rate), tmp_path, "rec", FrontendParams(max_windows_per_burst=4))
    rows, w = load_outputs(tmp_path, "rec")
    check_rows(rows)
    check_windows(w)
    assert np.allclose(np.sqrt(np.mean(np.abs(w) ** 2, axis=1)), 1, atol=1e-5)
    assert stats["bursts"] == 2 and stats["windows"] == len(rows) == 8
    assert abs(stats["slice_offset_hz"] - 12e6) <= 5e6
    assert len({r["window_id"] for r in rows}) == len(rows)
    for r in rows:
        assert "split" not in r
        assert r["snr_db_est"] > 10 and 0 < r["duty_cycle"] <= 1 and r["bandwidth_hz"] > 0
        assert r["center_frequency_hz"] == pytest.approx(2.44e9 + stats["slice_offset_hz"])
        assert r["native_rate_hz"] == rate and r["band"] == "ism_2400"


def test_chunked_reads_give_identical_output(tmp_path):
    rate = 120e6
    x = signal(rate, 3_000_000, [(500_000, 2_000_000)], offset_hz=-30e6, seed=5)
    a, b = tmp_path / "a", tmp_path / "b"
    process_recording(make_recording(x, rate), a, "rec", FrontendParams(max_windows_per_burst=1000))
    process_recording(make_recording(x, rate), b, "rec", FrontendParams(max_windows_per_burst=1000, read_chunk=99_991))
    rows_a, w_a = load_outputs(a, "rec")
    rows_b, w_b = load_outputs(b, "rec")
    assert rows_a == rows_b and len(rows_a) > 10
    assert np.array_equal(w_a, w_b)


def test_burst_across_a_block_edge_is_detected_once(tmp_path):
    rate, block = 20e6, 1 << 19
    x = signal(rate, 3 * block, [(int(0.8 * block), int(1.1 * block))], seed=6)
    params = FrontendParams(detect_block=block, max_windows_per_burst=1000, read_chunk=200_000)
    stats = process_recording(make_recording(x, rate), tmp_path, "rec", params)
    assert stats["bursts"] == 1
    rows, _ = load_outputs(tmp_path, "rec")
    expected = (int(1.1 * block) - int(0.8 * block) - WINDOW_SAMPLES) // (WINDOW_SAMPLES // 2) + 1
    assert abs(len(rows) - expected) <= 2


def test_caps_and_limit_seconds(tmp_path):
    rate = 20e6
    x = signal(rate, 2_000_000, [(100_000, 1_900_000)])
    stats = process_recording(make_recording(x, rate), tmp_path / "a", "rec",
                              FrontendParams(max_windows_per_burst=1000, max_windows_per_recording=5))
    assert stats["windows"] == 5
    stats = process_recording(make_recording(x, rate), tmp_path / "b", "rec", FrontendParams(limit_seconds=0.01))
    assert stats["native_samples_read"] == 200_000


def test_background_windows_come_from_quiet_stretches(tmp_path):
    rate = 20e6
    x = signal(rate, 1_000_000, [(400_000, 500_000)])
    stats = process_recording(make_recording(x, rate, label="background"), tmp_path, "rec",
                              FrontendParams(max_windows_per_burst=1000))
    rows, w = load_outputs(tmp_path, "rec")
    check_rows(rows)
    assert stats["windows"] > 100
    assert all(r["snr_db_est"] is None and r["label"] == "background" for r in rows)


def test_stem_for():
    assert stem_for("DJI_inspire_2_5G_1of2.bin") == "DJI_inspire_2_5G_1of2"
    assert stem_for("GPSL1+AMtone/GPSL1@x.bin") == "GPSL1+AMtone__GPSL1@x"


def write_tampere(path, n, seed=0):
    """int16 little-endian interleaved I/Q, Tampere's format, with one strong burst."""
    x = signal(120e6, n, [(n // 4, n // 2)], offset_hz=25e6, seed=seed) * 0.2
    pairs = np.empty(2 * n, dtype="<i2")
    pairs[0::2] = np.clip(np.round(x.real * 32768), -32768, 32767)
    pairs[1::2] = np.clip(np.round(x.imag * 32768), -32768, 32767)
    path.write_bytes(pairs.tobytes())


def test_cli_on_tampere_format_files(tmp_path, capsys):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    write_tampere(raw / "DJI_mavic_mini_2G.bin", 1_200_000)
    write_tampere(raw / "DJI_inspire_2_2G_1of2.bin", 900_000, seed=1)
    code = preprocess.main(["--dataset", "tampere", "--raw", str(raw), "--out", str(out), "--max-windows-per-burst", "6"])
    assert code == 0
    summary = json.loads((out / "preprocess_summary.json").read_text())
    assert summary["recordings"] == 2 and summary["windows"] == 12
    for stem in ("DJI_mavic_mini_2G", "DJI_inspire_2_2G_1of2"):
        stats = json.loads((out / f"{stem}.stats.json").read_text())
        assert stats["resampler"]["down"] == 6 and abs(stats["slice_offset_hz"] - 25e6) <= 5e6
        rows, w = load_outputs(out, stem)
        check_rows(rows)
        assert len(rows) == 6 and rows[0]["dataset"] == "tampere" and rows[0]["native_rate_hz"] == 120e6


def test_cli_with_no_recordings(tmp_path):
    (tmp_path / "raw").mkdir()
    assert preprocess.main(["--dataset", "tampere", "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "o")]) == 2
