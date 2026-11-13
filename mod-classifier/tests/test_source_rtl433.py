"""rtl_433_tests loader on a small tree with the real layout."""

import numpy as np

from mcls.sources import get
from mcls.sources.rtl433 import DEFAULT_RATE_HZ, device_of, parse_name
from pathlib import Path


def write_cu8(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(values, dtype=np.uint8).tofile(path)


def test_names_give_centre_and_rate_or_defaults():
    assert parse_name("g001_315.1M_250k.cu8") == (315.1e6, 250e3, True)
    assert parse_name("01_FR_1_433.92M_250k.cu8") == (433.92e6, 250e3, True)
    assert parse_name("g005_2500k.cu8") == (433.92e6, 2500e3, False)
    assert parse_name("gfile001.cu8") == (433.92e6, DEFAULT_RATE_HZ, False)


def test_device_is_the_folder_under_tests():
    assert device_of(Path("tests/6sc2/01/g001.cu8")) == "6sc2"
    assert device_of(Path("tests/Ascot/gfile001.cu8")) == "Ascot"
    assert device_of(Path("false_positives_test/x/gfile001.cu8")) is None
    assert device_of(Path("tests/loose.cu8")) is None


def test_recordings_group_by_device_and_read_cu8(tmp_path):
    n = 5000
    iq = np.tile([255, 0], n)  # I = +1, Q = -1 after (x - 127.5) / 127.5
    write_cu8(tmp_path / "tests" / "6sc2" / "01" / "g001_315.1M_250k.cu8", iq)
    write_cu8(tmp_path / "tests" / "6sc2" / "02" / "gfile001.cu8", iq)
    write_cu8(tmp_path / "tests" / "Ascot" / "gfile001.cu8", iq)
    write_cu8(tmp_path / "tests" / "Ascot" / "tiny.cu8", iq[:100])  # too short: skipped
    write_cu8(tmp_path / "false_positives_test" / "x" / "gfile001.cu8", iq)  # skipped
    recs = list(get("rtl_433", root=tmp_path).recordings())
    assert [r.recording_id for r in recs] == ["6sc2", "6sc2", "Ascot"]
    assert {r.label for r in recs} == {"other_ism"}
    assert recs[0].center_frequency_hz == 315.1e6 and recs[1].center_frequency_hz == 433.92e6
    x = recs[0].read(3, 2)
    assert x.dtype == np.complex64
    np.testing.assert_allclose(x, [1 - 1j, 1 - 1j], atol=1e-6)
    assert "none stated" in recs[0].license
