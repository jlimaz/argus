import numpy as np
import pytest

from mcls.sources import get
from mcls.sources.gateman import SCENARIOS, Gateman, parse
from mcls.taxonomy import Taxonomy, load


def taxonomy():
    return Taxonomy(
        {
            "classes": {"v1": ["background"], "v2": ["gnss_jammer"]},
            "datasets": {
                "gateman": {
                    "license": "CC BY 4.0",
                    "is_real": True,
                    "default_label": "gnss_jammer",
                    "default_strength": "strong",
                    "overrides": [
                        {"match": "/GPSL1@", "label": "background", "strength": "weak"},
                        {"match": "/GPSL5@", "label": "background", "strength": "weak"},
                        {"match": "/GALE1@", "label": "background", "strength": "weak"},
                    ],
                }
            },
        }
    )


IQ = np.array([1000 - 2000j, 3000 + 4000j, -5000 + 6j, 7 - 8j], np.complex64)


def interleave(iq, dtype):
    pairs = np.empty(2 * len(iq), dtype)
    pairs[0::2], pairs[1::2] = iq.real, iq.imag
    return pairs.tobytes()


def write_scenario(root, scenario, raw=None):
    gnss, jammer = scenario.split("+")
    folder = root / "rawdata" / scenario
    folder.mkdir(parents=True, exist_ok=True)
    raw = raw if raw is not None else interleave(IQ, ">i2")
    (folder / f"{gnss}@20MSps.bin").write_bytes(raw)
    (folder / f"{jammer}@20MSps.bin").write_bytes(raw)
    return folder


def recordings(root):
    return {r.source_file: r for r in Gateman(root=root, taxonomy=taxonomy()).recordings()}


def test_byte_order_and_iq_order(tmp_path):
    """Fails if the big-endian file were read little-endian, or I and Q swapped."""
    write_scenario(tmp_path, "GPSL1+AMtone")
    rec = recordings(tmp_path)["rawdata/GPSL1+AMtone/AMtone@20MSps.bin"]
    np.testing.assert_allclose(rec.read(0, 4), IQ / 32768, rtol=1e-6)
    assert rec.read(1, 2).dtype == np.complex64
    np.testing.assert_allclose(rec.read(1, 2), IQ[1:3] / 32768, rtol=1e-6)
    # The same bytes read as little-endian would differ: the fixture really pins the byte order.
    assert not np.allclose(np.frombuffer(interleave(IQ, ">i2"), "<i2")[:2], [1000, -2000])


def test_metadata_and_inferred_centre(tmp_path):
    write_scenario(tmp_path, "GPSL5+Chirp20MHz")
    write_scenario(tmp_path, "GALE1+Chirp10MHz")
    recs = recordings(tmp_path)
    jam = recs["rawdata/GPSL5+Chirp20MHz/Chirp20MHz@20MSps.bin"]
    assert (jam.sample_rate_hz, jam.center_frequency_hz, jam.num_samples) == (20e6, 1176.45e6, 4)
    assert jam.meta["role"] == "jammer" and jam.meta["jammer"] == "Chirp20MHz"
    assert "inferred" in jam.meta["center_frequency_source"]
    assert jam.label == "gnss_jammer"
    gal = recs["rawdata/GALE1+Chirp10MHz/GALE1@20MSps.bin"]
    assert gal.center_frequency_hz == 1575.42e6 and gal.band == "gnss_l1"
    assert gal.meta["role"] == "gnss" and (gal.label, gal.label_strength) == ("background", "weak")
    with pytest.raises(IndexError):
        jam.read(3, 2)


def test_both_files_of_a_scenario_share_one_recording_id(tmp_path):
    write_scenario(tmp_path, "GPSL1+AMtone")
    write_scenario(tmp_path, "GPSL1+Chirp10MHz")
    recs = recordings(tmp_path)
    a = recs["rawdata/GPSL1+AMtone/AMtone@20MSps.bin"].recording_id
    b = recs["rawdata/GPSL1+AMtone/GPSL1@20MSps.bin"].recording_id
    c = recs["rawdata/GPSL1+Chirp10MHz/Chirp10MHz@20MSps.bin"].recording_id
    assert a == b == "GPSL1+AMtone" != c


def test_ignores_files_that_do_not_fit(tmp_path):
    folder = write_scenario(tmp_path, "GPSL1+AMtone")
    (folder / "GPSL5@20MSps.bin").write_bytes(interleave(IQ, ">i2"))  # wrong band for this folder
    (tmp_path / "readBinData.m").write_text("% reader")
    assert parse(folder / "GPSL5@20MSps.bin") is None
    assert len(recordings(tmp_path)) == 2


def test_check_reports_partial_download_and_missing_files(tmp_path):
    for scenario in SCENARIOS:
        write_scenario(tmp_path, scenario)
    assert Gateman(root=tmp_path, taxonomy=taxonomy()).check() == []
    (tmp_path / "rawdata" / "GPSL5+AMtone" / "AMtone@20MSps.bin").write_bytes(interleave(IQ, ">i2")[:-2])
    (tmp_path / "rawdata" / "GALE1+AMtone" / "GALE1@20MSps.bin").unlink()
    problems = Gateman(root=tmp_path, taxonomy=taxonomy()).check()
    assert any("GPSL5+AMtone/AMtone@20MSps.bin" in p and "partial download" in p for p in problems)
    assert any("GALE1+AMtone: gnss file missing" in p for p in problems)
    assert len(problems) == 2


def test_check_hints_at_unextracted_zip(tmp_path):
    (tmp_path / "rawdata.zip").write_bytes(b"PK")
    assert "extract it first" in Gateman(root=tmp_path, taxonomy=taxonomy()).check()[0]


def test_registered(tmp_path):
    assert type(get("gateman", root=tmp_path, taxonomy=taxonomy())) is Gateman


def test_real_taxonomy_skips_gnss_only_files(tmp_path):
    for scenario in SCENARIOS:
        write_scenario(tmp_path, scenario)
    recs = list(Gateman(root=tmp_path, taxonomy=load()).recordings())
    assert len(recs) == 9 and {r.meta["role"] for r in recs} == {"jammer"}
    assert {(r.label, r.label_strength) for r in recs} == {("gnss_jammer", "strong")}
    assert len({r.recording_id for r in recs}) == 9


def test_excluded_files_are_skipped(tmp_path):
    write_scenario(tmp_path, "GPSL1+AMtone")
    t = Taxonomy(
        {
            "classes": {"v1": ["gnss_jammer"]},
            "datasets": {
                "gateman": {
                    "license": "CC BY 4.0",
                    "is_real": True,
                    "default_exclude": True,
                    "note": "unknown",
                    "overrides": [{"match": "/AMtone@", "label": "gnss_jammer", "strength": "strong", "emitter_id": "vst"}],
                }
            },
        }
    )
    recs = list(Gateman(root=tmp_path, taxonomy=t).recordings())
    assert [(r.source_file, r.emitter_id) for r in recs] == [("rawdata/GPSL1+AMtone/AMtone@20MSps.bin", "vst")]


def test_real_zip_file_names_with_16bit_suffix(tmp_path):
    # Names inside rawdata.zip (central directory read Nov 12): <scenario>/<signal>@20MSps-16bit.bin
    raw = interleave(IQ, ">i2")
    folder = tmp_path / "GPSL1+AMtone"
    folder.mkdir()
    (folder / "AMtone@20MSps-16bit.bin").write_bytes(raw)
    (folder / "GPSL1@20MSps-16bit.bin").write_bytes(raw)
    assert parse(folder / "AMtone@20MSps-16bit.bin") is not None
    labelled = list(Gateman(root=tmp_path, taxonomy=load()).recordings())
    assert [r.source_file for r in labelled] == ["GPSL1+AMtone/AMtone@20MSps-16bit.bin"]
    assert labelled[0].label == "gnss_jammer"
