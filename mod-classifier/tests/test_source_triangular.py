import gzip

import numpy as np
import pytest

from mcls.sources import get, triangular
from mcls.sources.triangular import GzipReader, TriangularL1, parse_name, sidecar
from mcls.taxonomy import Taxonomy, load


def taxonomy():
    entry = {"license": "CC BY 4.0", "is_real": True, "default_label": "gnss_jammer", "default_strength": "strong"}
    return Taxonomy({"classes": {"v1": ["background"], "v2": ["gnss_jammer"]}, "datasets": {"triangular_l1": entry}})


def name(power, time="101500"):
    return f"LINKS_L1_10Mhz_complex_int8_20250118_{time}_TRI_{power}"


#: I and Q differ in sign and size, so swapping them, or reading unsigned, fails.
IQ = np.array([-128 + 127j, 64 - 32j, -1 + 0j, 5 - 100j], np.complex64)


def interleave(iq):
    pairs = np.empty(2 * len(iq), np.int8)
    pairs[0::2], pairs[1::2] = iq.real, iq.imag
    return pairs.tobytes()


def write_gz(folder, stem, raw):
    path = folder / f"{stem}.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(raw)
    return path


def source(root):
    return TriangularL1(root=root, taxonomy=taxonomy())


def test_parse_name():
    assert parse_name(name(85)) == {"date": "20250118", "time": "101500", "power": 85}
    assert parse_name(name(85) + ".gz") is None
    assert parse_name("LINKS_L5_10Mhz_complex_int8_20250118_1_TRI_85") is None


def test_byte_order_and_iq_order_gzip(tmp_path):
    """Real gzip fixture; fails if I and Q were swapped or int8 were read as uint8."""
    write_gz(tmp_path, name(70), interleave(IQ))
    rec = next(source(tmp_path).recordings())
    assert rec.meta["access"] == "gzip_stream" and rec.num_samples == 4
    np.testing.assert_allclose(rec.read(0, 4), IQ / 128, rtol=1e-6)
    assert rec.read(0, 1).dtype == np.complex64
    assert (rec.sample_rate_hz, rec.center_frequency_hz, rec.band) == (10e6, 1575.42e6, "gnss_l1")
    assert rec.meta["jamming_power"] == 70 and "inferred" in rec.meta["center_frequency_source"]


def test_decompressed_sibling_is_preferred_and_matches(tmp_path):
    write_gz(tmp_path, name(75), interleave(IQ))
    (tmp_path / name(75)).write_bytes(interleave(IQ))
    recs = list(source(tmp_path).recordings())
    assert len(recs) == 1 and recs[0].meta["access"] == "memmap"
    np.testing.assert_allclose(recs[0].read(1, 3), IQ[1:] / 128, rtol=1e-6)


def test_gzip_reader_forward_and_backward(tmp_path):
    rng = np.random.default_rng(0)
    pairs = rng.integers(-128, 128, size=2 * 50_000, dtype=np.int8)
    path = write_gz(tmp_path, name(80), pairs.tobytes())
    reader = GzipReader(path, np.dtype("i1"), 50_000)
    expected = (pairs.astype(np.float32) / 128).view(np.complex64)
    for start, count in [(0, 10), (100, 5), (40_000, 1000), (49_990, 10)]:  # forward: one open
        np.testing.assert_allclose(reader(start, count), expected[start : start + count])
    assert reader.restarts == 1
    np.testing.assert_allclose(reader(3, 4), expected[3:7])  # backward: reopens
    assert reader.restarts == 2
    with pytest.raises(IndexError):
        reader(49_999, 2)
    reader.close()


def test_sample_count_is_cached_in_a_sidecar(tmp_path, monkeypatch):
    gz = write_gz(tmp_path, name(65), interleave(IQ))
    assert next(source(tmp_path).recordings()).num_samples == 4
    assert sidecar(gz).read_text().split() == ["4", str(gz.stat().st_size)]

    def fail(_):
        raise AssertionError("counted again despite the sidecar")

    monkeypatch.setattr(triangular, "count_gzip_bytes", fail)
    assert next(source(tmp_path).recordings()).num_samples == 4
    # A replaced file (different size) is counted again.
    monkeypatch.undo()
    write_gz(tmp_path, name(65), interleave(IQ) * 3)
    assert next(source(tmp_path).recordings()).num_samples == 12


def test_each_file_is_its_own_recording(tmp_path):
    for power in (65, 85):
        write_gz(tmp_path, name(power), interleave(IQ))
    ids = [r.recording_id for r in source(tmp_path).recordings()]
    assert sorted(ids) == [name(65), name(85)]


def test_check_reports_truncated_gzip_and_missing_powers(tmp_path):
    for power in triangular.POWERS:
        write_gz(tmp_path, name(power), interleave(IQ) * 100)
    assert source(tmp_path).check() == []
    whole = (tmp_path / f"{name(80)}.gz").read_bytes()
    (tmp_path / f"{name(80)}.gz").write_bytes(whole[: len(whole) // 2])  # partial download
    (tmp_path / f"{name(85)}.gz").unlink()
    (tmp_path / name(70)).write_bytes(interleave(IQ)[:-1])  # odd-length decompressed sibling
    problems = source(tmp_path).check()
    assert any(name(80) in p and "partial download" in p for p in problems)
    assert any("power 85" in p for p in problems)
    assert any(name(70) in p and "whole number" in p for p in problems)
    assert len(problems) == 3


def test_registered(tmp_path):
    assert type(get("triangular_l1", root=tmp_path, taxonomy=taxonomy())) is TriangularL1


def test_real_taxonomy(tmp_path):
    for power in triangular.POWERS:
        write_gz(tmp_path, name(power), interleave(IQ))
    recs = list(TriangularL1(root=tmp_path, taxonomy=load()).recordings())
    assert len(recs) == 5 and {(r.label, r.license) for r in recs} == {("gnss_jammer", "CC BY 4.0")}
    assert sorted(r.meta["jamming_power"] for r in recs) == list(triangular.POWERS)


def test_excluded_gzip_is_not_decompressed(tmp_path, monkeypatch):
    write_gz(tmp_path, name(65), interleave(IQ))
    write_gz(tmp_path, name(85), interleave(IQ))
    entry = {"license": "CC BY 4.0", "is_real": True, "default_label": "gnss_jammer", "default_strength": "weak",
             "overrides": [{"match": "_TRI_65", "exclude": True, "note": "test"}]}
    t = Taxonomy({"classes": {"v1": ["gnss_jammer"]}, "datasets": {"triangular_l1": entry}})
    counted = []
    real_count = triangular.count_gzip_bytes
    monkeypatch.setattr(triangular, "count_gzip_bytes", lambda p: counted.append(p.name) or real_count(p))
    recs = list(TriangularL1(root=tmp_path, taxonomy=t).recordings())
    assert [r.recording_id for r in recs] == [name(85)] and counted == [f"{name(85)}.gz"]
