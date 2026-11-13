"""Channel choice and channelizer: finds the active slice, mixes it to DC, chunked == one-shot."""

import numpy as np
import pytest

from mcls.frontend.channelize import GRID_BINS, Channelizer, best_slice_bin, choose_offset, to_20msps


def noisy_signal(rate, centre_hz, width_hz, n, snr_db=20, seed=0):
    """Band-limited noise of `width_hz` at `centre_hz` (relative to DC) over white noise."""
    rng = np.random.default_rng(seed)
    spec = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1 / rate)
    band = np.abs(f - centre_hz) < width_hz / 2
    spec[band] = rng.standard_normal(band.sum()) + 1j * rng.standard_normal(band.sum())
    sig = np.fft.ifft(spec)
    sig /= np.sqrt(np.mean(np.abs(sig) ** 2))
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2) * 10 ** (-snr_db / 20)
    return (sig + noise).astype(np.complex64)


def reader(x):
    return lambda start, count: x[start : start + count]


@pytest.mark.parametrize("rate,centre", [(120e6, 31e6), (120e6, -40e6), (200e6, 72e6), (200e6, -5e6)])
def test_choose_offset_finds_the_signal(rate, centre):
    x = noisy_signal(rate, centre, 10e6, 1 << 18)
    offset = choose_offset(reader(x), len(x), rate)
    # The 20 MHz slice must contain the 10 MHz signal whole.
    assert abs(offset - centre) <= 5e6 + rate / GRID_BINS


def test_dc_spike_does_not_pull_the_slice():
    rate = 120e6
    x = noisy_signal(rate, 35e6, 8e6, 1 << 18) + np.complex64(3.0)  # strong DC offset
    assert abs(choose_offset(reader(x), len(x), rate) - 35e6) < 7e6


def test_slice_stays_inside_the_band():
    spectrum = np.zeros(GRID_BINS)
    spectrum[-1] = 1.0  # energy at the very top edge
    b = best_slice_bin(spectrum, 120e6)
    width = round(20e6 / 120e6 * GRID_BINS)
    assert b + width // 2 <= GRID_BINS // 2


def test_narrow_recordings_are_not_shifted():
    x = noisy_signal(20e6, 3e6, 2e6, 1 << 15)
    assert choose_offset(reader(x), len(x), 20e6) == 0.0
    assert choose_offset(reader(x), len(x), 10e6) == 0.0


def test_mixed_tone_lands_at_expected_frequency():
    rate, f_signal = 120e6, 33e6
    n = 1 << 18
    x = np.exp(2j * np.pi * f_signal / rate * np.arange(n)).astype(np.complex64)
    ch = Channelizer(rate, 30e6)
    y = np.concatenate([ch.process(x), ch.flush()])
    body = y[1000:-1000]
    steps = np.angle(body[1:] / body[:-1]) / (2 * np.pi) * 20e6
    measured = np.median(steps)
    assert np.abs(steps - measured).max() < 100.0  # no glitch anywhere (Hz)
    assert abs(measured - (f_signal - ch.offset_hz)) < 1.0  # Hz
    assert abs(np.abs(y[1000:-1000]).mean() - 1) < 1e-3  # passband gain 1


def test_offset_snapped_to_grid():
    ch = Channelizer(120e6, 30.01e6)
    assert ch.offset_hz == pytest.approx(round(30.01e6 / (120e6 / GRID_BINS)) * 120e6 / GRID_BINS)


@pytest.mark.parametrize("rate", [120e6, 200e6, 20e6, 15e6])
def test_chunked_equals_one_shot(rate):
    x = noisy_signal(rate, 0.1 * rate, 5e6, 400_000, seed=3)
    offset = 0.1 * rate if rate > 20e6 else 0.0
    whole = to_20msps(x, rate, offset)
    rng = np.random.default_rng(4)
    cuts = np.sort(rng.integers(0, len(x), 17))
    ch = Channelizer(rate, offset)
    pieces = [ch.process(c) for c in np.split(x, cuts)] + [ch.flush()]
    assert np.array_equal(np.concatenate(pieces), whole)
    assert whole.dtype == np.complex64
