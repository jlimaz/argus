import numpy as np
import pytest

from pipeline.inputs import combine_observations
from pipeline.sweep import Sweep, SweepError, detect_signals, read_sweep

RNG = np.random.default_rng(3)


def rtl_power_lines(passes: int, hops=(88e6, 98e6), bins=100, step=1e5, signals=()) -> str:
    """rtl_power-style CSV: smooth noise at -60 dB plus (low, high, dB, passes) signals."""
    lines = []
    for p in range(passes):
        for low in hops:
            centres = low + (np.arange(bins) + 0.5) * step
            db = -60 + RNG.normal(0, 0.5, bins)
            for lo, hi, level, on in signals:
                if on is None or p in on:
                    db[(centres > lo) & (centres < hi)] = level
            values = ", ".join(f"{v:.2f}" for v in db)
            lines.append(f"2026-11-11, 20:{p:02d}:00, {int(low)}, {int(low + bins * step)}, {step:.2f}, 10, {values}")
    return "\n".join(lines) + "\n"


def test_two_planted_signals_are_found():
    text = rtl_power_lines(3, signals=[(91.4e6, 91.6e6, -30, None), (100.9e6, 101.3e6, -35, {1})])
    rows = detect_signals(read_sweep(text))
    assert len(rows) == 2
    station, burst = rows
    assert station["center_frequency_hz"] == pytest.approx(91.5e6, abs=1e5)
    assert station["bandwidth_hz"] == 200_000
    assert station["occupancy"] == 1.0
    assert station["snr_db"] == pytest.approx(30, abs=1.5)
    assert station["noise_floor_db"] == pytest.approx(-60, abs=1)
    assert burst["center_frequency_hz"] == pytest.approx(101.1e6, abs=1e5)
    assert burst["bandwidth_hz"] == 400_000
    assert burst["occupancy"] == pytest.approx(1 / 3, abs=0.01)
    assert burst["first_seen"] == burst["last_seen"] == "2026-11-11T20:01:00"
    assert all(row["modulation"] == "unknown" for row in rows)


def test_hops_are_stitched_into_passes():
    sweep = read_sweep(rtl_power_lines(4))
    assert sweep.passes == 4
    assert sweep.frequencies_hz.size == 200
    assert sweep.bin_hz == 1e5
    assert sweep.frequencies_hz[0] == pytest.approx(88.05e6)


def test_hackrf_sweep_lines_out_of_order_still_make_passes():
    # hackrf_sweep reports quarter-spans of each tuning, not in frequency order.
    order = [(2400e6, 2405e6), (2410e6, 2415e6), (2405e6, 2410e6), (2415e6, 2420e6)]
    lines = []
    for p in range(4):
        for low, high in order:
            centres = low + (np.arange(50) + 0.5) * 1e5
            db = -70 + 10 * np.log10(RNG.exponential(1.0, 50))
            if p == 2:
                db[(centres > 2411e6) & (centres < 2413e6)] = -30
            values = ", ".join(f"{v:.2f}" for v in db)
            lines.append(f"2026-11-11, 20:10:0{p}.123456, {int(low)}, {int(high)}, 100000.00, 20, {values}")
    sweep = read_sweep("\n".join(lines))
    assert sweep.passes == 4
    rows = detect_signals(sweep)
    assert len(rows) == 1
    assert rows[0]["center_frequency_hz"] == pytest.approx(2412e6, abs=2e5)
    assert rows[0]["occupancy"] == 0.25
    assert rows[0]["first_seen"] == "2026-11-11T20:10:02.123456"


def test_header_comments_and_nan_values_are_skipped():
    lines = rtl_power_lines(2).splitlines()
    # rtl_power writes nan (or -nan) for a bin it could not measure.
    first = lines[0].split(", ")
    first[6], first[7] = "nan", "-nan"
    lines[0] = ", ".join(first)
    text = "# rtl_power capture\ndate, time, Hz low, Hz high, Hz step, samples, dB\n" + "\n".join(lines)
    sweep = read_sweep(text)
    assert sweep.passes == 2
    assert sweep.frequencies_hz.size == 200
    assert np.isnan(sweep.power_db[0, :2]).all()
    assert not np.isnan(sweep.power_db[1]).any()


@pytest.mark.parametrize("passes", [1, 4, 20])
@pytest.mark.parametrize("noisy", [False, True])
def test_noise_alone_is_not_a_signal(passes, noisy):
    rng = np.random.default_rng(passes)
    shape = (passes, 2000)
    power = -70 + (10 * np.log10(rng.exponential(1.0, shape)) if noisy else rng.normal(0, 0.5, shape))
    sweep = Sweep(np.arange(2000) * 1e5, power, 1e5, ["t"] * passes)
    assert len(detect_signals(sweep)) <= (1 if noisy and passes == 1 else 0)


def test_wide_signal_is_one_detection():
    power = -70 + RNG.normal(0, 0.5, (4, 2000))
    power[:, 1200:1360] = -55 + RNG.normal(0, 0.5, (4, 160))
    rows = detect_signals(Sweep(np.arange(2000) * 1e5, power, 1e5, ["t"] * 4))
    assert len(rows) == 1
    assert rows[0]["bandwidth_hz"] == pytest.approx(16e6, abs=3e5)


def test_rows_are_a_valid_data_file():
    import json

    rows = detect_signals(read_sweep(rtl_power_lines(2, signals=[(91.4e6, 91.6e6, -30, None)])))
    observations, total = combine_observations([("sweep.jsonl", json.dumps(rows))])
    assert total == len(rows) == 1
    assert observations[0]["modulation"] == "unknown"


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "no sweep lines"),
        ("2026-11-11, 20:00:00, 88000000, 98000000\n", "Line 1 has 4 fields"),
        ("2026-11-11, 20:00:00, 88M, 98000000, 100000, 10, -60\n", "Hz low is '88M'"),
        ("2026-11-11, 20:00:00, 98000000, 88000000, 100000, 10, -60\n", "do not describe a span"),
        (
            "d, t, 88000000, 89000000, 100000, 10, -60\nd, t, 89000000, 90000000, 50000, 10, -60\n",
            "different bin widths",
        ),
    ],
)
def test_bad_files_say_what_is_wrong(text, message):
    with pytest.raises(SweepError, match=message):
        read_sweep(text)
