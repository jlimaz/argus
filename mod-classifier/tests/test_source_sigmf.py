import json

import numpy as np
import pytest

from mcls.sources import get
from mcls.sources.sigmf import LoRaIQ, Occupancy, component_dtype
from mcls.taxonomy import Taxonomy, load


def taxonomy():
    entry = {"license": "CC BY 4.0", "is_real": True, "default_label": "lora", "default_strength": "strong"}
    return Taxonomy(
        {
            "classes": {"v1": ["background"], "v2": ["lora"]},
            "datasets": {"loraiq": entry, "occupancy": dict(entry, default_label="background", default_strength="weak")},
        }
    )


def write_sigmf(folder, stem, datatype, raw: bytes, captures=None, annotations=(), rate=1e6):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{stem}.sigmf-data").write_bytes(raw)
    meta = {
        "global": {"core:datatype": datatype, "core:sample_rate": rate, "core:version": "1.0.0"},
        "captures": captures if captures is not None else [{"core:sample_start": 0, "core:frequency": 868e6}],
        "annotations": list(annotations),
    }
    (folder / f"{stem}.sigmf-meta").write_text(json.dumps(meta))


def interleave(iq, dtype):
    pairs = np.empty(2 * len(iq), dtype)
    pairs[0::2], pairs[1::2] = iq.real, iq.imag
    return pairs.tobytes()


@pytest.mark.parametrize(
    "datatype, expected",
    [("cf32_le", "<f4"), ("cf32_be", ">f4"), ("ci16_le", "<i2"), ("ci16_be", ">i2"), ("ci8", "i1"), ("cu8", "u1"), ("ci32_be", ">i4"), ("cf64_le", "<f8")],
)
def test_component_dtype(datatype, expected):
    assert component_dtype(datatype) == np.dtype(expected)


@pytest.mark.parametrize("bad", ["rf32_le", "ci16", "ci8_le", "cf16_le", "xx"])
def test_component_dtype_rejects(bad):
    with pytest.raises(ValueError):
        component_dtype(bad)


IQ = np.array([1000 - 2000j, 3000 + 4000j, -5000 + 6j, 7 - 8j], np.complex64)


@pytest.mark.parametrize(
    "datatype, dtype, scale",
    [("ci16_le", "<i2", 32768), ("ci16_be", ">i2", 32768), ("cf32_le", "<f4", 1), ("cf32_be", ">f4", 1)],
)
def test_byte_order_and_iq_order(tmp_path, datatype, dtype, scale):
    """Fails if a big-endian file were read little-endian, or I and Q swapped."""
    write_sigmf(tmp_path, "a", datatype, interleave(IQ, dtype))
    rec = next(LoRaIQ(root=tmp_path, taxonomy=taxonomy()).recordings())
    np.testing.assert_allclose(rec.read(0, 4), IQ / scale, rtol=1e-6)
    assert rec.read(0, 4).dtype == np.complex64


def test_eight_bit_types(tmp_path):
    write_sigmf(tmp_path, "s", "ci8", np.array([-128, 127, 0, 64], np.int8).tobytes())
    write_sigmf(tmp_path, "u", "cu8", np.array([0, 255, 128, 127], np.uint8).tobytes())
    recs = {r.source_file: r for r in Occupancy(root=tmp_path, taxonomy=taxonomy()).recordings()}
    np.testing.assert_allclose(recs["s.sigmf-meta#0"].read(0, 2), [complex(-1, 127 / 128), 0.5j], rtol=1e-6)
    np.testing.assert_allclose(recs["u.sigmf-meta#0"].read(0, 2), [complex(-1, 1), complex(0.5 / 127.5, -0.5 / 127.5)], rtol=1e-5)


def test_one_recording_per_capture_with_header_bytes(tmp_path):
    a, b = IQ[:3], IQ[3:]
    raw = b"H" * 8 + interleave(a, "<i2") + b"G" * 4 + interleave(b, "<i2")
    captures = [
        {"core:sample_start": 0, "core:frequency": 868.1e6, "core:header_bytes": 8},
        {"core:sample_start": 3, "core:frequency": 868.3e6, "core:header_bytes": 4},
    ]
    write_sigmf(tmp_path, "x", "ci16_le", raw, captures, annotations=[{"core:sample_start": 3, "core:sample_count": 1, "core:label": "frame"}])
    recs = list(LoRaIQ(root=tmp_path, taxonomy=taxonomy()).recordings())
    assert [(r.num_samples, r.center_frequency_hz) for r in recs] == [(3, 868.1e6), (1, 868.3e6)]
    np.testing.assert_allclose(recs[0].read(0, 3), a / 32768, rtol=1e-6)
    np.testing.assert_allclose(recs[1].read(0, 1), b / 32768, rtol=1e-6)
    assert recs[0].recording_id == recs[1].recording_id
    assert recs[1].meta["annotations"][0]["core:label"] == "frame" and recs[0].meta["annotations"] == []
    with pytest.raises(IndexError):
        recs[1].read(0, 2)


def test_loraiq_groups_the_four_receivers_of_one_measurement(tmp_path):
    raw = interleave(IQ, "<f4")
    for rrh in ("RRH1", "RRH2"):
        write_sigmf(tmp_path / "sigmfs" / "2025-01-01_10-00" / rrh, "1", "cf32_le", raw, captures=[{"core:sample_start": 0}])
    write_sigmf(tmp_path / "sigmfs" / "2025-01-02_10-00" / "RRH1", "1", "cf32_le", raw, captures=[{"core:sample_start": 0}])
    recs = list(LoRaIQ(root=tmp_path, taxonomy=taxonomy()).recordings())
    ids = [r.recording_id for r in recs]
    assert ids[0] == ids[1] != ids[2]
    assert recs[0].center_frequency_hz == 862.5e6  # fallback when the capture has no frequency
    assert recs[0].label == "lora"



def test_loraiq_groups_lower_case_receiver_folders(tmp_path):
    # The real zip (central directory read Nov 12) uses sigmfs/25_01_15-10_17/rrh1/10.sigmf-data.
    raw = interleave(IQ, "<f4")
    for rrh in ("rrh1", "rrh2"):
        write_sigmf(tmp_path / "sigmfs" / "25_01_15-10_17" / rrh, "10", "cf32_le", raw, captures=[{"core:sample_start": 0}])
    write_sigmf(tmp_path / "sigmfs" / "25_09_18-14_15" / "rrh1", "10", "cf32_le", raw, captures=[{"core:sample_start": 0}])
    ids = [r.recording_id for r in LoRaIQ(root=tmp_path, taxonomy=taxonomy()).recordings()]
    assert ids[0] == ids[1] != ids[2]

def test_occupancy_needs_a_frequency(tmp_path):
    write_sigmf(tmp_path / "Home" / "iq", "ISM_2400_t01_wifi0_20260822T004844_s0000", "ci16_le", interleave(IQ, "<i2"), captures=[{"core:sample_start": 0}])
    with pytest.raises(ValueError, match="core:frequency"):
        list(Occupancy(root=tmp_path, taxonomy=taxonomy()).recordings())


def test_check_reports_partial_data(tmp_path):
    write_sigmf(tmp_path, "a", "ci16_le", interleave(IQ, "<i2")[:-1])
    write_sigmf(tmp_path, "b", "ci16_le", interleave(IQ, "<i2"), annotations=[{"core:sample_start": 2, "core:sample_count": 10}])
    (tmp_path / "c.sigmf-meta").write_text(json.dumps({"global": {"core:datatype": "ci16_le", "core:sample_rate": 1}}))
    problems = LoRaIQ(root=tmp_path, taxonomy=taxonomy()).check()
    assert any("a.sigmf-data" in p and "whole number" in p for p in problems)
    assert any("b.sigmf-data" in p and "annotation" in p for p in problems)
    assert any("c.sigmf-meta" in p and "missing" in p for p in problems)


def test_registered():
    assert type(get("loraiq", root=".", taxonomy=taxonomy())) is LoRaIQ
    assert type(get("occupancy", root=".", taxonomy=taxonomy())) is Occupancy


def test_real_taxonomy(tmp_path):
    write_sigmf(tmp_path / "loraiq" / "sigmfs" / "2025-01-01_10-00" / "RRH1", "1", "ci16_le", interleave(IQ, "<i2"))
    write_sigmf(tmp_path / "occ" / "Home" / "iq", "ISM_2400_t01_wifi0_20260822T004844_s0000", "ci16_le", interleave(IQ, "<i2"),
                captures=[{"core:sample_start": 0, "core:frequency": 2.44e9}])
    lora = next(LoRaIQ(root=tmp_path / "loraiq", taxonomy=load()).recordings())
    occ = next(Occupancy(root=tmp_path / "occ", taxonomy=load()).recordings())
    assert (lora.label, occ.label) == ("lora", "background")
    assert lora.emitter_id is None and occ.emitter_id is None


def test_excluded_files_are_skipped(tmp_path):
    write_sigmf(tmp_path, "keep", "ci16_le", interleave(IQ, "<i2"))
    write_sigmf(tmp_path, "drop", "ci16_le", interleave(IQ, "<i2"))
    entry = {"license": "CC BY 4.0", "is_real": True, "default_label": "lora", "default_strength": "weak",
             "overrides": [{"match": "drop", "exclude": True, "note": "test"}]}
    t = Taxonomy({"classes": {"v1": ["lora"]}, "datasets": {"loraiq": entry}})
    assert [r.source_file for r in LoRaIQ(root=tmp_path, taxonomy=t).recordings()] == ["keep.sigmf-meta#0"]


def test_loraiq_keeps_every_nth_frame_and_reads_mhz(tmp_path, monkeypatch):
    raw = interleave(IQ, "<f4")
    folder = tmp_path / "sigmfs" / "25_01_15-10_17" / "rrh1"
    for i in range(1, 11):
        write_sigmf(folder, str(i), "cf32_le", raw, captures=[{"core:sample_start": 0, "core:frequency": 862.5}])
    monkeypatch.setenv("MCLS_LORAIQ_STRIDE", "4")
    recs = list(LoRaIQ(root=tmp_path, taxonomy=taxonomy()).recordings())
    assert [r.source_file.split("/")[-1] for r in recs] == ["1.sigmf-meta#0", "5.sigmf-meta#0", "9.sigmf-meta#0"]
    assert recs[0].center_frequency_hz == 862.5e6  # the files state MHz
