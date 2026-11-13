"""burst_id on rows, and windows_from_iq (inference) reproducing preprocess exactly."""

import numpy as np
import pytest

from test_preprocess_cli import load_outputs, make_recording, signal

from mcls.frontend.channelize import Channelizer
from mcls.frontend.process import FrontendParams, check_rows, process_recording, windows_from_iq


def test_burst_ids_on_rows(tmp_path):
    rate = 40e6
    x = signal(rate, 2_400_000, [(200_000, 600_000), (1_500_000, 1_560_000)], offset_hz=12e6)
    process_recording(make_recording(x, rate), tmp_path, "rec", FrontendParams(max_windows_per_burst=4))
    rows, _ = load_outputs(tmp_path, "rec")
    check_rows(rows)
    ids = [r["burst_id"] for r in rows]
    assert ids == ["fake/rec/b000000"] * 4 + ["fake/rec/b000001"] * 4
    # Rows of one burst share its stats.
    assert len({(r["snr_db_est"], r["duty_cycle"]) for r in rows[:4]}) == 1


def test_carried_burst_keeps_one_id(tmp_path):
    rate, block = 20e6, 1 << 19
    x = signal(rate, 3 * block, [(int(0.8 * block), int(1.1 * block)), (int(2.2 * block), int(2.4 * block))], seed=6)
    params = FrontendParams(detect_block=block, max_windows_per_burst=1000, read_chunk=200_000)
    process_recording(make_recording(x, rate), tmp_path, "rec", params)
    rows, _ = load_outputs(tmp_path, "rec")
    ids = [r["burst_id"] for r in rows]
    assert sorted(set(ids)) == ["fake/rec/b000000", "fake/rec/b000001"]
    first = [i for i in ids if i.endswith("b000000")]
    assert len(first) >= (int(0.3 * block) - 4096) // 2048 - 1  # the whole carried burst, under one id


def test_background_rows_have_no_burst_id(tmp_path):
    rate = 20e6
    x = signal(rate, 1_000_000, [(400_000, 500_000)])
    process_recording(make_recording(x, rate, label="background"), tmp_path, "rec")
    rows, _ = load_outputs(tmp_path, "rec")
    assert rows and all(r["burst_id"] is None for r in rows)


@pytest.mark.parametrize("rate,offset", [(120e6, -30e6), (20e6, 0.0), (15e6, 0.0)])
def test_windows_from_iq_matches_preprocess(tmp_path, rate, offset):
    n = int(0.06 * rate)
    spans = [(int(0.1 * n), int(0.3 * n)), (int(0.75 * n), int(0.8 * n))]
    x = signal(rate, n, spans, offset_hz=offset, seed=9)
    params = FrontendParams(max_windows_per_burst=7, detect_block=1 << 19)
    stats = process_recording(make_recording(x, rate), tmp_path, "rec", params)
    rows, shard_windows = load_outputs(tmp_path, "rec")
    result = windows_from_iq(x, rate, 2.44e9, params=params, id_prefix="fake/rec")
    assert np.array_equal(result.windows, shard_windows)
    assert result.burst_ids == [r["burst_id"] for r in rows]
    assert len(result.bursts) == stats["bursts"]
    assert result.slice_offset_hz == stats["slice_offset_hz"] and result.samples_20msps == stats["samples_20msps"]
    for r, k in zip(rows, result.window_burst):
        assert result.bursts[k]["snr_db"] == r["snr_db_est"] and result.bursts[k]["duty_cycle"] == r["duty_cycle"]


def test_burst_times_are_delay_corrected():
    rate = 120e6
    a, b = 600_000, 1_800_000
    x = signal(rate, 3_000_000, [(a, b)], seed=2)
    result = windows_from_iq(x, rate, 2.44e9)
    assert len(result.bursts) == 1
    burst = result.bursts[0]
    # Edge accuracy: within 60 native samples (0.5 us). Without the correction the
    # start would be off by the resampler lag, about 150 native samples here.
    assert abs(burst["native_start"] - a) < 60 and abs(burst["native_end"] - b) < 60
    assert burst["start_s"] == pytest.approx(burst["native_start"] / rate)
    ch = Channelizer(rate, 0.0)
    assert abs(burst["start"] * ch.down / ch.up - a) > 100  # the naive mapping is measurably wrong
    # Window starts map back inside the burst.
    assert np.all(result.window_start_native >= a - 60)
    assert np.all(result.window_start_native + 4096 * rate / 20e6 <= b + 60)


def test_background_and_empty_input():
    rate = 20e6
    x = signal(rate, 600_000, [(200_000, 260_000)])
    result = windows_from_iq(x, rate, 2.44e9, background=True, id_prefix="p")
    assert len(result.windows) > 0 and all(k is None for k in result.window_burst)
    assert all(i is None for i in result.burst_ids)
    empty = windows_from_iq(np.zeros(0, np.complex64), rate, 2.44e9)
    assert empty.windows.shape == (0, 4096) and empty.bursts == []
