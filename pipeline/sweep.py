"""Read rtl_power and hackrf_sweep output and find the signals in it.

Both tools write one CSV line per frequency hop:

    date, time, Hz low, Hz high, Hz step, samples, dB, dB, ...

rtl_power calls the fifth and sixth columns "Hz step" and "samples"; hackrf_sweep
calls them "hz_bin_width" and "num_samples". The shape is the same, so one reader
covers both. The number of dB values varies with the span of the line.

`read_sweep` stitches the lines into a power-versus-frequency table, one row per
pass over the band. `detect_signals` finds each signal in that table and returns
it in the observation format the classifier reads (`pipeline/inputs.py`). A sweep
shows where a signal is and how wide it is, not how it is modulated, so every
detection has `modulation: "unknown"`.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

# A signal is a run of bins at least this far above the local noise floor...
THRESHOLD_DB = 10.0
# ...and its edges are where it falls back to this far above the floor.
EDGE_DB = 3.0
# Runs separated by this many bins or fewer are one signal (an OFDM dip, say).
MERGE_GAP_BINS = 1
# The local floor is this percentile of the bins in a window around each bin.
# A low percentile keeps a wide signal from lifting the floor under itself.
FLOOR_PERCENTILE = 25
FLOOR_WINDOW_BINS = 201
# rtl_power averages many FFTs per line, so its noise is smooth; hackrf_sweep
# writes single FFTs, whose noise swings by several dB from bin to bin. The
# thresholds above are raised to the noise's own median plus this many spreads,
# measured on the file, so single-FFT noise does not read as signals.
NOISE_SPREADS = 2.0


class SweepError(ValueError):
    """A sweep file that cannot be read, with the line at fault."""


@dataclass
class Sweep:
    """Power in dB per pass (rows) and frequency bin (columns). NaN where a pass has no value."""

    frequencies_hz: np.ndarray
    power_db: np.ndarray
    bin_hz: float
    times: list[str]

    @property
    def passes(self) -> int:
        return self.power_db.shape[0]


def _number(text: str, line_number: int, what: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise SweepError(f"Line {line_number}: {what} is {text.strip()!r}, not a number.") from None


def _lines(text: str):
    """(line number, date, time, low, high, step, dB values) for every data line."""
    for number, line in enumerate(text.splitlines(), 1):
        fields = [field.strip() for field in line.split(",")]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # A header such as "date, time, hz_low, ..." is skipped wherever it appears.
        if len(fields) > 2 and fields[2].lower().replace(" ", "_") in ("hz_low", "hz_lo"):
            continue
        if len(fields) < 7:
            raise SweepError(
                f"Line {number} has {len(fields)} fields. A sweep line has a date, a time, "
                "Hz low, Hz high, Hz step, a sample count and at least one dB value."
            )
        low = _number(fields[2], number, "Hz low")
        high = _number(fields[3], number, "Hz high")
        step = _number(fields[4], number, "Hz step")
        if step <= 0 or high <= low:
            raise SweepError(f"Line {number}: Hz low, Hz high and Hz step do not describe a span.")
        values = [_number(value, number, "a dB value") for value in fields[6:]]
        yield number, fields[0], fields[1], low, high, step, values


def read_sweep(text: str) -> Sweep:
    """Stitch rtl_power or hackrf_sweep lines into one table.

    A new pass starts when a line covers a span the current pass already has: the
    radio has come back round to the start of the band.
    """
    passes: list[dict[float, float]] = []
    times: list[str] = []
    seen: set[tuple[float, float]] = set()
    steps: set[float] = set()
    for _number_, date, time, low, high, step, values in _lines(text):
        span = (low, high)
        if not passes or span in seen:
            passes.append({})
            times.append(f"{date}T{time}")
            seen = set()
        seen.add(span)
        steps.add(round(step, 6))
        current = passes[-1]
        for index, value in enumerate(values):
            # Bin i covers [low + i * step, low + (i + 1) * step).
            centre = round(low + (index + 0.5) * step, 3)
            if math.isfinite(value):
                current[centre] = value
    if not passes:
        raise SweepError("The file has no sweep lines.")
    if len(steps) > 1:
        raise SweepError(
            f"The lines use {len(steps)} different bin widths. Record one sweep setting per file."
        )
    frequencies = np.array(sorted({freq for row in passes for freq in row}), dtype=np.float64)
    if frequencies.size == 0:
        raise SweepError("The file has no dB values that are numbers.")
    column = {freq: index for index, freq in enumerate(frequencies)}
    power = np.full((len(passes), len(frequencies)), np.nan)
    for row, values in enumerate(passes):
        for freq, value in values.items():
            power[row, column[freq]] = value
    return Sweep(frequencies, power, steps.pop(), times)


def _to_linear(db: np.ndarray) -> np.ndarray:
    return 10.0 ** (db / 10.0)


def _to_db(linear: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(linear)


def local_floor(spectrum_db: np.ndarray, window: int = FLOOR_WINDOW_BINS) -> np.ndarray:
    """The noise floor under every bin: a low percentile of the bins around it.

    NaN bins (no value, or masked out as signal) are ignored. A bin whose whole
    window is NaN takes its floor from the nearest bins that have one.
    """
    count = len(spectrum_db)
    window = max(1, min(window, count))
    half = window // 2
    padded = np.pad(spectrum_db.astype(np.float64), (half, window - 1 - half), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        floor = np.nanpercentile(windows, FLOOR_PERCENTILE, axis=1)
    known = np.isfinite(floor)
    if known.any() and not known.all():
        index = np.arange(count)
        floor[~known] = np.interp(index[~known], index[known], floor[known])
    return floor


def _floor_under_signals(spectrum_db: np.ndarray, threshold_db: float) -> tuple[np.ndarray, np.ndarray]:
    """(floor, signal mask). Signal bins are masked and the floor measured again,
    so a signal wider than a quarter of the window does not lift the floor under itself."""
    hot = np.zeros(len(spectrum_db), dtype=bool)
    for _ in range(3):
        floor = local_floor(np.where(hot, np.nan, spectrum_db))
        detect_db, _edge = _levels(spectrum_db - floor, threshold_db)
        found = spectrum_db > floor + detect_db
        if np.array_equal(found | hot, hot):
            break
        hot |= found
    return floor, hot


def _levels(residual_db: np.ndarray, threshold_db: float) -> tuple[float, float]:
    """(detection, edge) levels in dB above the local floor.

    Signals cover a small share of the bins, so the median and the lower spread of
    the residual (spectrum minus floor) describe the noise.
    """
    finite = residual_db[np.isfinite(residual_db)]
    if finite.size == 0:
        return threshold_db, EDGE_DB
    middle = float(np.percentile(finite, 50))
    spread = middle - float(np.percentile(finite, 16))
    return max(threshold_db, middle + NOISE_SPREADS * spread), max(EDGE_DB, middle + 0.5 * spread)


def _runs(mask: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """[start, end) spans where mask is true; gaps of merge_gap bins or fewer are bridged."""
    edges = np.flatnonzero(np.diff(np.concatenate([[False], mask, [False]]).astype(np.int8)))
    merged: list[tuple[int, int]] = []
    for start, end in zip(edges[::2], edges[1::2]):
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], int(end))
        else:
            merged.append((int(start), int(end)))
    return merged


def detect_signals(sweep: Sweep, threshold_db: float = THRESHOLD_DB) -> list[dict]:
    """One observation per signal, lowest frequency first.

    Signals are found in the max-hold spectrum (the strongest value of each bin
    over all passes), so a burst seen in one pass is not averaged away. The noise
    floor that is reported comes from the mean spectrum. `occupancy` is the share
    of passes in which the signal was above the floor by `threshold_db`: near 1
    for a signal that is always on, low for one that comes and goes. It is a
    count of sweep passes, not the on-time `duty_cycle` that an IQ capture gives.
    """
    with np.errstate(all="ignore"):
        maximum = np.nanmax(sweep.power_db, axis=0)
        mean = _to_db(np.nanmean(_to_linear(sweep.power_db), axis=0))
    # Max-hold catches what comes and goes; the mean, whose noise is smoother,
    # catches a weak signal that is always on. A bin is signal in either.
    max_floor, max_hot = _floor_under_signals(maximum, threshold_db)
    mean_floor, mean_hot = _floor_under_signals(mean, threshold_db)
    max_detect, max_edge = _levels(maximum - max_floor, threshold_db)
    mean_detect, mean_edge = _levels(mean - mean_floor, threshold_db)
    above = (maximum > max_floor + max_detect) | (mean > mean_floor + mean_detect)
    near = (maximum > max_floor + max_edge) | (mean > mean_floor + mean_edge)
    # A single pass is as noisy as the raw lines, so occupancy gets its own level.
    pass_detect, _edge = _levels((sweep.power_db - mean_floor).ravel(), threshold_db)
    detections = []
    for start, end in _runs(above, MERGE_GAP_BINS):
        # Grow each edge while the signal is still a few dB above the floor.
        while start > 0 and near[start - 1]:
            start -= 1
        while end < len(near) and near[end]:
            end += 1
        if detections and start <= detections[-1][1]:
            detections[-1] = (detections[-1][0], max(end, detections[-1][1]))
        else:
            detections.append((start, end))

    rows = []
    for start, end in detections:
        band = slice(start, end)
        excess = np.clip(_to_linear(maximum[band]) - _to_linear(max_floor[band]), 0, None)
        frequencies = sweep.frequencies_hz[band]
        centre = float(np.average(frequencies, weights=excess)) if excess.sum() > 0 else float(frequencies.mean())
        peak = float(np.nanmax(maximum[band]))
        floor = float(np.nanmedian(mean_floor[band]))
        measured = ~np.isnan(sweep.power_db[:, band]).all(axis=1)
        with np.errstate(all="ignore"):
            on = np.nanmax(sweep.power_db[:, band], axis=1) > floor + pass_detect
        seen = [time for time, hit in zip(sweep.times, on & measured) if hit]
        rows.append(
            {
                "center_frequency_hz": round(centre),
                "bandwidth_hz": round((end - start) * sweep.bin_hz),
                "modulation": "unknown",
                "power_db": round(peak, 2),
                "noise_floor_db": round(floor, 2),
                "snr_db": round(peak - floor, 2),
                "occupancy": round(float(on[measured].mean()), 3) if measured.any() else 0.0,
                "passes": int(measured.sum()),
                "first_seen": seen[0] if seen else None,
                "last_seen": seen[-1] if seen else None,
                "source": "sweep",
            }
        )
    return rows
