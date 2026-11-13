"""Drexel CBRS loader on a small file with the real layout (HDF5 like MATLAB v7.3)."""

import csv

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from mcls.sources import get  # noqa: E402
from mcls.sources.drexel import DrexelCbrsRadar, subset_number  # noqa: E402

ROWS, SAMPLES = 4, 300


def write_subset(root, n, status, folder="Group6_exp", csv_rows=True, rate="10000000", var=None):
    base = root / "Group6_data" / folder / "Group6"
    base.mkdir(parents=True, exist_ok=True)
    wave = np.zeros((ROWS, SAMPLES), dtype=[("real", "<f8"), ("imag", "<f8")])
    for i in range(ROWS):
        wave["real"][i] = i + np.arange(SAMPLES) / 1000
        wave["imag"][i] = -i
    with h5py.File(base / f"group6_subset_{n}.mat", "w") as hf:
        v = n if var is None else var
        hf.create_dataset(f"waveformSubset_{v}", data=wave, chunks=(ROWS, 20), compression="gzip")
        hf.create_dataset(f"radarStatusSubset_{v}", data=np.array(status, dtype=np.uint8).reshape(-1, 1))
    if csv_rows:
        info = base / "group6_subset_CSVInfo"
        info.mkdir(exist_ok=True)
        with open(info / f"group6_waveformTableSubset_{n}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, ["BinNo", "SamplingFrequency", "radarStatus", "SNR"])
            writer.writeheader()
            for i, s in enumerate(status):
                writer.writerow({"BinNo": f"B{i}", "SamplingFrequency": rate, "radarStatus": s, "SNR": 12})
    return base / f"group6_subset_{n}.mat"


def test_reads_exp_files_only_with_labels_from_radar_status(tmp_path):
    write_subset(tmp_path, 1, [1, 0, 1, 0])
    write_subset(tmp_path, 1, [1, 1, 1, 1], folder="Group6_orig")  # simulated originals: skipped
    recs = list(get("drexel_cbrs_radar", root=tmp_path).recordings())
    assert len(recs) == ROWS
    assert [r.label for r in recs] == ["pulsed_radar", "background", "pulsed_radar", "background"]
    assert {r.is_real for r in recs} == {False}
    assert len({r.recording_id for r in recs}) == ROWS
    assert recs[0].sample_rate_hz == 10e6 and recs[0].num_samples == SAMPLES
    assert recs[1].meta["BinNo"] == "B1"


def test_read_returns_the_right_row_and_slice(tmp_path):
    write_subset(tmp_path, 2, [1, 0, 1, 0])
    recs = list(DrexelCbrsRadar(root=tmp_path).recordings())
    x = recs[2].read(10, 5)
    assert x.dtype == np.complex64
    np.testing.assert_allclose(x.real, 2 + np.arange(10, 15) / 1000, rtol=1e-6)
    np.testing.assert_allclose(x.imag, -2)
    with pytest.raises(IndexError):
        recs[0].read(SAMPLES - 2, 5)


def test_csv_disagreement_is_an_error(tmp_path):
    write_subset(tmp_path, 3, [1, 0, 1, 0], rate="20000000")
    with pytest.raises(ValueError, match="sampling frequency"):
        list(DrexelCbrsRadar(root=tmp_path).recordings())


def test_subset_number_and_check(tmp_path):
    assert subset_number(tmp_path / "Group6_exp" / "Group6" / "group6_subset_7.mat") == 7
    assert subset_number(tmp_path / "Group6_orig" / "Group6" / "group6_subset_7.mat") is None
    (tmp_path / "Group6_data.tar").write_bytes(b"")
    assert any("extract it first" in p for p in DrexelCbrsRadar(root=tmp_path).check())


def test_variable_suffix_need_not_match_the_file(tmp_path):
    # group6_subset_2.mat on CRC holds waveformSubset_1 / radarStatusSubset_1.
    write_subset(tmp_path, 2, [1, 0, 1, 0], var=1)
    recs = list(DrexelCbrsRadar(root=tmp_path).recordings())
    assert [r.label for r in recs] == ["pulsed_radar", "background", "pulsed_radar", "background"]
    assert recs[3].read(0, 1).imag[0] == -3
