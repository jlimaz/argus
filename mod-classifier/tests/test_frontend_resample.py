"""Rational resampler: tone accuracy, alias rejection, chunked == one-shot, matches direct convolution."""

import numpy as np
import pytest

from mcls.frontend.resample import RationalResampler, design_lowpass, rational, resample

# (native rate, tone offset Hz) for every rate the datasets use or might use.
RATES = [120e6, 200e6, 40e6, 56e6, 15e6, 10e6, 25e6]


def tone(rate, freq, n):
    return np.exp(2j * np.pi * freq / rate * np.arange(n)).astype(np.complex64)


def ideal_output(r: RationalResampler, rate, freq, count):
    """The tone at the exact input times of output samples 0..count-1."""
    m = np.arange(count)
    t_in = (r._first_j + m * r.down - r.delay) / r.up
    return np.exp(2j * np.pi * freq / rate * t_in)


@pytest.mark.parametrize("rate", RATES)
def test_rational_ratio(rate):
    up, down = rational(rate, 20e6)
    assert up * rate == pytest.approx(down * 20e6)


def test_rational_rejects_irrational_ratio():
    with pytest.raises(ValueError):
        rational(20e6 * np.pi, 20e6, max_denominator=50)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("freq", [0.0, 1.3e6, -3.7e6, 6.5e6])
def test_tone_in_tone_out(rate, freq):
    """Passband tones (|f| <= 0.8 x the narrower Nyquist) come out with error below 1e-4 of full scale.

    Measured on this filter (Kaiser, 80 dB): max |error| over every output sample is
    about 2e-6 to 3e-5 depending on the rate (float32 rounding plus passband ripple).
    """
    nyquist = min(rate, 20e6) / 2
    if abs(freq) > 0.8 * nyquist:
        pytest.skip("outside the flat passband for this rate")
    up, down = rational(rate, 20e6)
    r = RationalResampler(up, down)
    y = resample(tone(rate, freq, 200_000), up, down)
    error = np.abs(y - ideal_output(r, rate, freq, len(y))).max()
    assert error < 1e-4
    # Output length: one sample per `down` upsampled samples after the start-up transient.
    assert abs(len(y) - (200_000 * up - r._first_j) / down) <= 1


@pytest.mark.parametrize("rate", [120e6, 200e6, 40e6])
def test_out_of_slice_tone_is_rejected(rate):
    """A tone outside +-10 MHz must not alias into the output: at least 70 dB down."""
    up, down = rational(rate, 20e6)
    y = resample(tone(rate, 14e6, 200_000), up, down)[1000:]
    assert 20 * np.log10(np.abs(y).max()) < -70


def test_matches_direct_convolution():
    """FFT overlap-save equals the textbook form: zero-stuff, convolve, keep every M-th."""
    up, down = 4, 3
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(50_000) + 1j * rng.standard_normal(50_000)).astype(np.complex64)
    r = RationalResampler(up, down)
    y = resample(x, up, down)
    u = np.zeros(len(x) * up, dtype=np.complex128)
    u[::up] = x
    v = np.convolve(u, r.taps)[: len(u)]
    ref = v[r._first_j :: down]
    assert len(ref) == len(y)
    assert np.abs(y - ref).max() < 1e-5 * np.abs(ref).max()


@pytest.mark.parametrize("rate", [120e6, 15e6])
def test_chunked_equals_one_shot_bit_for_bit(rate):
    up, down = rational(rate, 20e6)
    rng = np.random.default_rng(2)
    n = 300_000
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    whole = resample(x, up, down)
    for trial in range(3):
        cuts = np.sort(rng.integers(0, n, 1 + 10 * trial))
        r = RationalResampler(up, down)
        pieces = [r.process(c) for c in np.split(x, cuts)] + [r.flush()]
        assert np.array_equal(np.concatenate(pieces), whole)


def test_filter_shape():
    h = design_lowpass(1, 6)
    assert len(h) % 2 == 1
    assert np.allclose(h, h[::-1])
    assert h.sum() == pytest.approx(1.0)
    assert design_lowpass(4, 3).sum() == pytest.approx(4.0)


def test_process_after_flush_raises():
    r = RationalResampler(1, 2)
    r.flush()
    with pytest.raises(RuntimeError):
        r.process(np.zeros(10, dtype=np.complex64))
