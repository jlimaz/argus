import json

import numpy as np
import pytest

from pipeline.inputs import combine_observations
from pipeline.iq import IQError, read_raw, read_sigmf, to_complex

FS = 1_000_000.0
RNG = np.random.default_rng(11)


def capture(total: int, starts: list[int], width: int, offset_hz: float) -> np.ndarray:
    """Noise with tone pulses, scaled to fit an 8-bit file (noise about 0.05, pulses 0.5)."""
    iq = 0.05 / np.sqrt(2) * (RNG.standard_normal(total) + 1j * RNG.standard_normal(total))
    t = np.arange(total) / FS
    for start in starts:
        iq[start : start + width] += 0.5 * np.exp(2j * np.pi * offset_hz * t[start : start + width])
    return iq


def interleave(iq: np.ndarray) -> np.ndarray:
    out = np.empty(2 * len(iq), dtype=np.float64)
    out[0::2], out[1::2] = iq.real, iq.imag
    return out


def write(path, iq: np.ndarray, datatype: str):
    values = interleave(iq)
    if datatype == "cu8":
        data = np.clip(np.round(values * 127.5 + 127.5), 0, 255).astype(np.uint8)
    elif datatype == "ci8":
        data = np.clip(np.round(values * 128), -128, 127).astype(np.int8)
    elif datatype == "ci16_le":
        data = np.round(values * 32768).astype("<i2")
    else:
        data = values.astype("<f4")
    path.write_bytes(data.tobytes())
    return path


@pytest.mark.parametrize("suffix, datatype", [(".cu8", "cu8"), (".cs8", "ci8"), (".ci16", "ci16_le"), (".cf32", "cf32_le")])
def test_raw_files_round_trip(tmp_path, suffix, datatype):
    iq = capture(4096, [], 0, 0.0)
    path = write(tmp_path / f"x{suffix}", iq, datatype)
    values = np.fromfile(path, dtype={"cu8": np.uint8, "ci8": np.int8, "ci16_le": "<i2", "cf32_le": "<f4"}[datatype])
    tolerance = 1e-6 if datatype == "cf32_le" else 1e-2
    assert np.allclose(to_complex(values, datatype), iq, atol=tolerance)


@pytest.mark.parametrize("suffix, datatype", [(".cu8", "cu8"), (".cs8", "ci8")])
def test_pulses_in_a_raw_capture_are_found(tmp_path, suffix, datatype):
    starts = [20_000, 60_000, 100_000]
    path = write(tmp_path / f"drone{suffix}", capture(200_000, starts, 5_000, 120_000.0), datatype)
    rows = read_raw(path, FS, 2.44e9)
    assert len(rows) == 3
    for row, start in zip(rows, starts):
        assert row["center_frequency_hz"] == pytest.approx(2.44e9 + 120_000, abs=5_000)
        assert row["start_s"] == pytest.approx(start / FS, abs=1e-4)
        assert row["modulation"] == "unknown"
        assert row["source"] == path.name
    combine_observations([("rows", json.dumps(rows))], limit=len(rows))


def test_chunks_keep_start_times_from_the_file_start(tmp_path):
    path = write(tmp_path / "x.cs8", capture(300_000, [250_000], 5_000, 50_000.0), "ci8")
    rows = read_raw(path, FS, 100e6, chunk_samples=100_000)
    assert len(rows) == 1
    assert rows[0]["start_s"] == pytest.approx(0.25, abs=1e-4)


def test_raw_capture_needs_rate_and_frequency(tmp_path):
    path = write(tmp_path / "x.cu8", capture(1000, [], 0, 0.0), "cu8")
    with pytest.raises(IQError, match="sample rate"):
        read_raw(path, None, 100e6)
    with pytest.raises(IQError, match="centre frequency"):
        read_raw(path, FS, None)


def test_unknown_suffix_needs_a_format(tmp_path):
    path = write(tmp_path / "x.bin", capture(1000, [], 0, 0.0), "ci8")
    with pytest.raises(IQError, match="does not say the sample format"):
        read_raw(path, FS, 100e6)
    assert read_raw(path, FS, 100e6, datatype="ci8") == []


def sigmf(tmp_path, name="rec.2437M", datatype="ci16_le", **overrides):
    first = capture(100_000, [30_000], 5_000, 100_000.0)
    second = capture(100_000, [40_000], 5_000, -200_000.0)
    write(tmp_path / f"{name}.sigmf-data", np.concatenate([first, second]), datatype)
    meta = {
        "global": {"core:datatype": datatype, "core:sample_rate": FS, "core:version": "1.0.0"},
        "captures": [
            {"core:sample_start": 0, "core:frequency": 2.437e9, "core:datetime": "2026-11-11T20:00:00Z"},
            {"core:sample_start": 100_000, "core:frequency": 5.8e9},
        ],
        "annotations": [],
    }
    for key, value in overrides.items():
        meta["global"][key] = value
    (tmp_path / f"{name}.sigmf-meta").write_text(json.dumps(meta))
    return tmp_path / f"{name}.sigmf-meta"


def test_sigmf_uses_each_capture_frequency(tmp_path):
    rows = read_sigmf(sigmf(tmp_path))
    assert len(rows) == 2
    assert rows[0]["center_frequency_hz"] == pytest.approx(2.437e9 + 100_000, abs=5_000)
    assert rows[0]["start_s"] == pytest.approx(0.03, abs=1e-4)
    assert rows[0]["capture_datetime"] == "2026-11-11T20:00:00Z"
    assert rows[1]["center_frequency_hz"] == pytest.approx(5.8e9 - 200_000, abs=5_000)
    assert rows[1]["start_s"] == pytest.approx(0.14, abs=1e-4)
    assert "capture_datetime" not in rows[1]


def test_sigmf_opens_from_the_data_file_too(tmp_path):
    meta = sigmf(tmp_path)
    assert len(read_sigmf(meta.with_name("rec.2437M.sigmf-data"))) == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"core:datatype": "rf32_le"}, "real-valued"),
        ({"core:datatype": "cf64_be"}, "not supported"),
        ({"core:num_channels": 2}, "single-channel"),
        ({"core:sample_rate": None}, "core:sample_rate"),
    ],
)
def test_sigmf_refusals_say_why(tmp_path, overrides, message):
    with pytest.raises(IQError, match=message):
        read_sigmf(sigmf(tmp_path, **overrides))


def test_sigmf_needs_both_files(tmp_path):
    meta = sigmf(tmp_path)
    meta.with_name("rec.2437M.sigmf-data").unlink()
    with pytest.raises(IQError, match="sigmf-data is missing"):
        read_sigmf(meta)
