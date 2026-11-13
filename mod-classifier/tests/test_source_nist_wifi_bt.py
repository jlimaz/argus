import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mcls.sources import get  # noqa: E402
from mcls.sources.nist_wifi_bt import NistWifiBt, grouping_problems, parse_name  # noqa: E402
from mcls.taxonomy import Taxonomy, load  # noqa: E402


def taxonomy():
    return Taxonomy(
        {
            "classes": {"v1": ["wifi", "bluetooth"]},
            "datasets": {
                "nist_wifi_bt": {
                    "license": "NIST open licence",
                    "is_real": True,
                    "default_label": "wifi",
                    "default_strength": "weak",
                    "overrides": [{"match": "bluetooth", "label": "bluetooth", "strength": "weak"}],
                }
            },
        }
    )


def interleave(iq, dtype="<i2"):
    pairs = np.empty(2 * len(iq), dtype)
    pairs[0::2], pairs[1::2] = iq.real, iq.imag
    return pairs


def write_h5(path, captures, dtype="<i2"):
    """The layout of TN 2237 section 2: one 1-D interleaved int16 dataset per timestamp key."""
    with h5py.File(path, "w") as hf:
        for key, iq in captures.items():
            hf.create_dataset(key, data=interleave(iq, dtype))


IQ = (np.arange(12) * 1000 + 1j * (-np.arange(12) * 500 - 7)).astype(np.complex64)


def test_parse_name():
    assert parse_name("2_4ghz_indoor.h5")["centre_hz"] == 2437e6
    assert parse_name("5ghz_outdoor.h5")["centre_hz"] == 5825e6
    assert parse_name("2_4ghz_bluetooth.h5")["group"] == "2_4ghz_bluetooth"
    assert parse_name("2_4ghz_indoor.csv") is None


def test_reads_interleaved_int16_i_first(tmp_path):
    # Big-endian on disk on purpose: HDF5 records the byte order, h5py must honour it.
    write_h5(tmp_path / "5ghz_indoor.h5", {"12:00:00": IQ, "12:01:00": IQ[::-1]}, dtype=">i2")
    recs = list(NistWifiBt(root=tmp_path, taxonomy=taxonomy()).recordings())
    assert len(recs) == 2
    rec = recs[0]
    assert rec.sample_rate_hz == 30e6 and rec.center_frequency_hz == 5825e6 and rec.num_samples == 12
    got = rec.read(3, 4)
    assert got.dtype == np.complex64
    # Fails if I and Q were swapped or the bytes misread: Q is negative and odd, I is a multiple of 1000.
    np.testing.assert_allclose(got, IQ[3:7] / 32768, rtol=1e-6)
    np.testing.assert_allclose(recs[1].read(0, 1), IQ[-1:] / 32768, rtol=1e-6)
    with pytest.raises(IndexError):
        rec.read(10, 3)


def test_indoor_and_outdoor_of_one_moment_share_an_id(tmp_path):
    write_h5(tmp_path / "2_4ghz_indoor.h5", {"t1": IQ})
    write_h5(tmp_path / "2_4ghz_outdoor.h5", {"t1": IQ, "t2": IQ})
    write_h5(tmp_path / "2_4ghz_bluetooth.h5", {"t1": IQ})
    recs = {r.source_file: r for r in NistWifiBt(root=tmp_path, taxonomy=taxonomy()).recordings()}
    assert recs["2_4ghz_indoor.h5:t1"].recording_id == recs["2_4ghz_outdoor.h5:t1"].recording_id
    assert recs["2_4ghz_outdoor.h5:t2"].recording_id != recs["2_4ghz_outdoor.h5:t1"].recording_id
    assert recs["2_4ghz_bluetooth.h5:t1"].recording_id != recs["2_4ghz_indoor.h5:t1"].recording_id
    assert recs["2_4ghz_bluetooth.h5:t1"].label == "bluetooth"
    assert recs["2_4ghz_indoor.h5:t1"].label == "wifi"


def test_csv_metadata_is_attached(tmp_path):
    write_h5(tmp_path / "5ghz_indoor.h5", {"t1": IQ, "t2": IQ})
    (tmp_path / "5ghz_indoor.csv").write_text("timestamp,gain\nt1,40\nt2,41\n")
    recs = list(NistWifiBt(root=tmp_path, taxonomy=taxonomy()).recordings())
    assert [r.meta["csv"]["gain"] for r in recs] == ["40", "41"]


def test_check_reports_short_files(tmp_path):
    write_h5(tmp_path / "5ghz_indoor.h5", {"t1": IQ})
    (tmp_path / "5ghz_outdoor.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 20)
    problems = NistWifiBt(root=tmp_path, taxonomy=taxonomy()).check()
    assert any("1 captures, expected 180" in p for p in problems)
    assert any("not 60000000" in p for p in problems)
    assert any("5ghz_outdoor.h5: cannot open" in p for p in problems)


def test_wrong_dtype_is_refused(tmp_path):
    with h5py.File(tmp_path / "5ghz_indoor.h5", "w") as hf:
        hf.create_dataset("t1", data=np.zeros(8, np.float32))
    with pytest.raises(ValueError, match="int16"):
        list(NistWifiBt(root=tmp_path, taxonomy=taxonomy()).recordings())


def test_registered():
    assert type(get("nist_wifi_bt", root=".", taxonomy=taxonomy())) is NistWifiBt


def test_check_flags_indoor_outdoor_keys_without_a_twin(tmp_path):
    """Guards the timestamp grouping: a key in only one of a pair would not share its recording_id."""
    write_h5(tmp_path / "2_4ghz_indoor.h5", {"t1": IQ, "t2": IQ})
    write_h5(tmp_path / "2_4ghz_outdoor.h5", {"t1": IQ, "t3": IQ})
    write_h5(tmp_path / "5ghz_indoor.h5", {"t9": IQ})  # no outdoor file: nothing to pair
    write_h5(tmp_path / "2_4ghz_bluetooth.h5", {"t7": IQ})  # not part of a pair
    problems = [p for p in NistWifiBt(root=tmp_path, taxonomy=taxonomy()).check() if "twin" in p]
    assert len(problems) == 2
    assert any(p.startswith("2_4ghz_indoor.h5: 1 capture keys") and "t2" in p for p in problems)
    assert any(p.startswith("2_4ghz_outdoor.h5: 1 capture keys") and "t3" in p for p in problems)
    assert grouping_problems({"5ghz_indoor.h5": ["a"], "5ghz_outdoor.h5": ["a"]}) == []


def test_real_taxonomy(tmp_path):
    for name in ("2_4ghz_indoor.h5", "5ghz_outdoor.h5", "2_4ghz_bluetooth.h5"):
        write_h5(tmp_path / name, {"t1": IQ})
    recs = {r.source_file: r for r in NistWifiBt(root=tmp_path, taxonomy=load()).recordings()}
    assert {f: r.label for f, r in recs.items()} == {
        "2_4ghz_bluetooth.h5:t1": "bluetooth",
        "2_4ghz_indoor.h5:t1": "wifi",
        "5ghz_outdoor.h5:t1": "wifi",
    }
    assert all(r.emitter_id is None for r in recs.values())
