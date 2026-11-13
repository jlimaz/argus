import numpy as np
import pytest

from mcls.baselines import features
from mcls.contracts import WINDOW_SAMPLES


def unit_rms(x):
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)


def noise(rng, n=WINDOW_SAMPLES):
    return (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)


def tone_burst(rng, start=1000, stop=3000, freq=0.1, snr_amp=10.0):
    x = 0.1 * noise(rng)
    t = np.arange(stop - start)
    x[start:stop] += snr_amp * 0.1 * np.exp(2j * np.pi * freq * t)
    return unit_rms(x)


def index(name):
    return features.FEATURE_NAMES.index(name)


def test_shape_dtype_and_finite():
    rng = np.random.default_rng(0)
    windows = np.stack([tone_burst(rng), unit_rms(noise(rng)), np.zeros(WINDOW_SAMPLES, np.complex64)])
    X = features.extract(windows)
    assert X.shape == (3, len(features.FEATURE_NAMES))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_tone_burst_is_found_and_measured():
    rng = np.random.default_rng(1)
    X = features.extract(tone_burst(rng)[None, :])[0]
    assert X[index("burst_missing")] == 0
    assert X[index("snr_db")] > 10
    # 2000 samples at 20 MS/s = 100 microseconds
    assert X[index("burst_duration_us")] == pytest.approx(100, rel=0.1)
    assert X[index("burst_duration_missing")] == 0
    assert X[index("pulse_interval_missing")] == 1
    assert X[index("peak_count")] == 1
    assert X[index("bandwidth_frac")] < 0.05


def test_noise_is_flat_and_tone_is_not():
    rng = np.random.default_rng(2)
    flat = features.spectral_features(unit_rms(noise(rng)))
    tone = features.spectral_features(unit_rms(np.exp(2j * np.pi * 0.2 * np.arange(WINDOW_SAMPLES))))
    names = ("spectral_flatness", "peak_count", "obw50_frac", "obw90_frac", "obw99_frac", "spectral_spread_frac")
    flat, tone = dict(zip(names, flat)), dict(zip(names, tone))
    assert flat["spectral_flatness"] > 0.8 and tone["spectral_flatness"] < 0.05
    assert flat["obw90_frac"] > 0.8 and tone["obw90_frac"] < 0.05
    assert flat["obw50_frac"] <= flat["obw90_frac"] <= flat["obw99_frac"] <= 1


def test_envelope_kurtosis_of_constant_envelope_and_gaussian_noise():
    rng = np.random.default_rng(3)
    kurt_tone, papr_tone, _ = features.envelope_features(np.exp(2j * np.pi * 0.1 * np.arange(WINDOW_SAMPLES)))
    assert kurt_tone == 0.0 and papr_tone == pytest.approx(0, abs=1e-6)
    # |complex Gaussian| is Rayleigh, whose kurtosis is about 3.245.
    kurt_noise = features.envelope_features(noise(rng, 200_000))[0]
    assert kurt_noise == pytest.approx(3.245, abs=0.1)


def test_burst_rows_given_by_the_caller_are_used():
    rng = np.random.default_rng(4)
    window = unit_rms(noise(rng))[None, :]
    row = {"bandwidth_hz": 2e6, "snr_db": 20.0, "duty_cycle": 0.5, "burst_duration_s": None,
           "pulse_interval_s": 1e-4, "hopping": True}
    X = features.extract(window, bursts=[row])[0]
    assert X[index("bandwidth_frac")] == pytest.approx(0.1)
    assert X[index("snr_db")] == 20
    assert X[index("burst_duration_missing")] == 1 and X[index("pulse_interval_us")] == pytest.approx(100)
    assert X[index("hopping")] == 1
    empty = features.extract(window, bursts=[[]])[0]
    assert empty[index("burst_missing")] == 1 and empty[index("snr_db")] == 0


def test_burst_list_length_must_match():
    with pytest.raises(ValueError):
        features.extract(np.zeros((2, WINDOW_SAMPLES), np.complex64), bursts=[None])
