import numpy as np
import pytest

from mcls.sources import get
from mcls.sources.dronerf import DroneRF, parse_csv, parse_name
from mcls.taxonomy import Taxonomy, load

N_REAL = 4096  # small fixture files instead of 10 million values


def taxonomy():
    return Taxonomy(
        {
            "classes": {"v1": ["wifi", "background"]},
            "datasets": {
                "dronerf": {
                    "license": "CC BY 4.0",
                    "is_real": True,
                    "default_label": "wifi",
                    "default_strength": "weak",
                    "overrides": [
                        {"match": "00000", "label": "background", "strength": "strong"},
                        {"match": "10111", "label": "wifi", "strength": "weak", "emitter_id": "from_taxonomy"},
                        {"match": "10011", "exclude": True, "note": "test exclusion"},
                    ],
                }
            },
        }
    )


def write_csv(path, values):
    path.write_text(",".join(f"{v:.7g}" for v in values))


def tone(freq_hz, n=N_REAL, rate=40e6):
    return 0.5 * np.cos(2 * np.pi * freq_hz / rate * np.arange(n))


def source(root):
    s = DroneRF(root=root, taxonomy=taxonomy())
    s.real_samples_per_file = N_REAL
    return s


def test_parse_name():
    p = parse_name("10111H_3.csv")
    assert (p["bui"], p["band"], p["segment"], p["drone"], p["mode"]) == ("10111", "H", 3, "Parrot_AR_Drone_2", "video_recording")
    assert parse_name("00000L_12.csv")["drone_present"] is False
    assert parse_name("readme.txt") is None and parse_name("1011H_3.csv") is None


def test_parse_csv_keeps_order_across_lines():
    assert parse_csv(b"1,-2.5,3\r\n4,5\n").tolist() == [1, -2.5, 3, 4, 5]


def test_tone_lands_at_the_right_baseband_frequency(tmp_path):
    """A 5 MHz real tone in the L file is 2405 MHz RF: baseband -5 MHz around a 2410 MHz centre.

    A sign error in the real-to-IQ step (or reading I/Q backwards) puts it at +5 MHz.
    """
    write_csv(tmp_path / "10000L_0.csv", tone(5e6))
    rec = next(source(tmp_path).recordings())
    assert rec.sample_rate_hz == 20e6 and rec.center_frequency_hz == 2410e6
    iq = rec.read(0, rec.num_samples)
    assert iq.dtype == np.complex64 and len(iq) == rec.num_samples
    freqs = np.fft.fftfreq(len(iq), 1 / rec.sample_rate_hz)
    peak = freqs[np.argmax(np.abs(np.fft.fft(iq)))]
    assert abs(peak - -5e6) < 50e3
    assert abs(rec.center_frequency_hz + peak - 2405e6) < 50e3


def test_all_segments_of_one_bui_share_a_group_and_labels_follow_the_taxonomy(tmp_path):
    """Segments of one BUI are likely consecutive slices of one session (see the loader), so one id."""
    for name in ("10111L_3.csv", "10111H_3.csv", "10111L_4.csv", "10000L_0.csv", "00000H_1.csv", "00000L_1.csv"):
        write_csv(tmp_path / name, tone(3e6))
    recs = {r.source_file: r for r in source(tmp_path).recordings()}
    assert recs["10111L_3.csv"].recording_id == recs["10111H_3.csv"].recording_id == recs["10111L_4.csv"].recording_id == "10111"
    assert recs["10000L_0.csv"].recording_id == "10000" != recs["00000H_1.csv"].recording_id
    assert recs["10111H_3.csv"].center_frequency_hz == 2450e6
    assert recs["00000H_1.csv"].label == "background" and recs["00000H_1.csv"].emitter_id is None
    assert recs["10111L_3.csv"].label == "wifi" and recs["10111L_3.csv"].emitter_id == "from_taxonomy"
    assert recs["10000L_0.csv"].emitter_id == "Parrot_Bebop"  # parsed fallback: no emitter_id in this taxonomy
    a = recs["10111L_3.csv"]
    np.testing.assert_array_equal(a.read(10, 5), a.read(0, 20)[10:15])
    with pytest.raises(IndexError):
        a.read(a.num_samples - 2, 5)


def test_excluded_files_are_skipped(tmp_path):
    write_csv(tmp_path / "10011L_0.csv", tone(3e6))
    write_csv(tmp_path / "10000L_0.csv", tone(3e6))
    assert [r.source_file for r in source(tmp_path).recordings()] == ["10000L_0.csv"]


def test_real_taxonomy(tmp_path):
    """Against configs/taxonomy.toml: names, labels and exclusions must line up with the loader."""
    for name in ("10000L_0.csv", "10100H_2.csv", "11010L_5.csv", "00000H_1.csv", "01111L_0.csv"):
        write_csv(tmp_path / name, tone(3e6))
    s = DroneRF(root=tmp_path, taxonomy=load())
    s.real_samples_per_file = N_REAL
    recs = {r.source_file: r for r in s.recordings()}
    assert "01111L_0.csv" not in recs  # unknown BUI: excluded by default
    assert {(f, recs[f].label, recs[f].emitter_id) for f in recs} == {
        ("10000L_0.csv", "wifi", "Parrot_Bebop"),
        ("10100H_2.csv", "wifi", "Parrot_AR_Drone_2"),
        ("11010L_5.csv", "wifi", "DJI_Phantom_3_Standard"),
        ("00000H_1.csv", "background", None),
    }
    for f in ("10000L_0.csv", "10100H_2.csv", "11010L_5.csv"):  # parsed names agree with the taxonomy
        assert parse_name(f)["drone"] == recs[f].emitter_id


def test_wrong_length_is_refused(tmp_path):
    write_csv(tmp_path / "10000L_0.csv", tone(5e6, n=N_REAL - 10))
    rec = next(source(tmp_path).recordings())
    with pytest.raises(ValueError, match="real samples"):
        rec.read(0, 4)


class FakeRar:
    """Stands in for unrar.cffi.RarFile: no RAR writer is available to build fixtures."""

    def __init__(self, members):
        self.members = members

    def infolist(self):
        return [type("Info", (), {"filename": name, "file_size": len(data)}) for name, data in self.members.items()]

    def read(self, name):
        return self.members[name]


def test_reads_members_inside_archives(tmp_path):
    payload = ",".join(f"{v:.7g}" for v in tone(5e6)).encode()
    archives = {"DroneRF_part1.rar": FakeRar({"Bebop/10000L_0.csv": payload, "Bebop/10000H_0.csv": payload, "Bebop/notes.txt": b""})}
    for name in archives:
        (tmp_path / name).write_bytes(b"Rar!")
    s = source(tmp_path)
    s.open_archive = lambda path: archives[path.name]
    recs = list(s.recordings())
    assert [r.source_file for r in recs] == ["DroneRF_part1.rar/Bebop/10000H_0.csv", "DroneRF_part1.rar/Bebop/10000L_0.csv"]
    assert recs[0].read(0, 8).dtype == np.complex64
    problems = s.check()
    assert any("expected 23" in p for p in problems)
    assert not any("missing" in p for p in problems)


def test_check_reports_a_missing_half(tmp_path):
    write_csv(tmp_path / "10000L_0.csv", tone(5e6))
    assert any("10000H_0.csv is missing" in p for p in source(tmp_path).check())


def test_registered():
    assert type(get("dronerf", root=".", taxonomy=taxonomy())) is DroneRF


def test_background_session_is_cut_into_time_blocks_with_gaps():
    from mcls.sources.dronerf import recording_id_for

    def rid(segment, bui="00000"):
        return recording_id_for({"bui": bui, "segment": segment})

    assert [rid(s) for s in (0, 6)] == ["00000_blk0"] * 2
    assert rid(7) is None and rid(15) is None and rid(31) is None  # 0.25 s gaps between blocks
    assert rid(8) == "00000_blk1" and rid(24) == "00000_blk3"
    assert [rid(s) for s in (32, 39, 40)] == ["00000_blk4"] * 3  # last block keeps its end
    assert len({rid(s) for s in range(41)} - {None}) == 5
    assert rid(7, bui="10111") == "10111"  # drone sessions keep one id per flight mode
