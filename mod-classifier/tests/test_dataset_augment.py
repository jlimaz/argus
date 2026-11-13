import dataclasses
import tomllib
from pathlib import Path

import numpy as np
import pytest

from mcls.contracts import SAMPLE_DTYPE, SAMPLE_RATE_HZ, WINDOW_SAMPLES, ManifestRow
from mcls.dataset import augment as A

N = WINDOW_SAMPLES
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def rng(seed=0):
    return np.random.default_rng(seed)


def unit(x):
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(SAMPLE_DTYPE)


def tone(freq_hz, n=N):
    return unit(np.exp(2j * np.pi * freq_hz * np.arange(n) / SAMPLE_RATE_HZ))


def noise_window(seed=0):
    r = rng(seed)
    return unit(r.standard_normal(N) + 1j * r.standard_normal(N))


def peak_hz(x, pad=16):
    spectrum = np.abs(np.fft.fft(x, pad * len(x)))
    freqs = np.fft.fftfreq(pad * len(x), 1 / SAMPLE_RATE_HZ)
    return freqs[np.argmax(spectrum)]


def snr_db(signal, total):
    noise = total - signal
    return 10 * np.log10(np.mean(np.abs(signal) ** 2) / np.mean(np.abs(noise) ** 2))


def make_row(split="train", **kw):
    base = dict(window_id="w", shard="s.npy", offset=0, length=N, dataset="d", recording_id="r",
                label="wifi", label_strength="strong", native_rate_hz=20e6, center_frequency_hz=2.44e9,
                band="ism_2400", is_real=True, license="x", split=split)
    base.update(kw)
    return ManifestRow(**base)


@pytest.mark.parametrize("seed", range(10))
def test_noise_never_raises_snr_known_current(seed):
    r = rng(seed)
    clean = tone(1e6)
    current = 15.0
    noise = r.standard_normal(N) + 1j * r.standard_normal(N)
    noise *= np.sqrt(np.mean(np.abs(clean) ** 2) / 10 ** (current / 10) / np.mean(np.abs(noise) ** 2))
    noisy = (clean + noise).astype(SAMPLE_DTYPE)
    before = snr_db(clean, noisy)
    out = A.add_noise_to_snr(noisy, r, (0.0, 30.0), current_snr_db=current)
    # Output is rescaled to the input RMS; undo that to measure SNR against the clean part.
    scale = np.vdot(clean, out).real / np.vdot(clean, clean).real
    after = snr_db(scale * clean, out)
    assert after <= before + 0.3
    assert np.sqrt(np.mean(np.abs(out) ** 2)) == pytest.approx(np.sqrt(np.mean(np.abs(noisy) ** 2)), rel=1e-4)


def test_noise_unchanged_when_range_is_above_current():
    w = noise_window()
    out = A.add_noise_to_snr(w, rng(), (20.0, 30.0), current_snr_db=10.0)
    assert np.array_equal(out, w)


@pytest.mark.parametrize("seed", range(5))
def test_noise_reaches_at_most_target_when_clean(seed):
    clean = tone(2e6)
    out = A.add_noise_to_snr(clean, rng(seed), (5.0, 5.0))
    scale = np.vdot(clean, out).real / np.vdot(clean, clean).real
    assert snr_db(scale * clean, out) == pytest.approx(5.0, abs=0.5)


def test_frequency_offset_bounded_and_magnitude_kept():
    w = tone(0.0)
    bandwidth = 4e6
    for seed in range(20):
        out = A.frequency_offset(w, rng(seed), bandwidth_hz=bandwidth, fraction=0.05)
        assert np.allclose(np.abs(out), np.abs(w), atol=1e-5)
        assert abs(peak_hz(out)) <= 0.05 * bandwidth + SAMPLE_RATE_HZ / (16 * N)
    # The offset actually moves the tone for some draw.
    assert any(abs(peak_hz(A.frequency_offset(w, rng(s), 4e6))) > 1e4 for s in range(5))


def test_time_shift_is_a_rotation():
    w = noise_window(1)
    out = A.time_shift(w, rng(2))
    shifts = [k for k in range(N) if np.array_equal(np.roll(w, k), out)]
    assert len(shifts) == 1


def test_time_crop_keeps_a_contiguous_scaled_stretch():
    w = noise_window(3)
    out = A.time_crop(w, rng(4), min_fraction=0.75)
    nonzero = np.flatnonzero(out)
    start, length = nonzero[0], len(nonzero)
    assert np.array_equal(nonzero, np.arange(start, start + length))
    assert length >= 0.75 * N
    segment = out[start:start + length]
    # The kept stretch is a scaled copy of some stretch of the input.
    matches = [s for s in range(N - length + 1)
               if np.allclose(segment / segment[0], w[s:s + length] / w[s], atol=1e-3)]
    assert matches
    assert np.sqrt(np.mean(np.abs(out) ** 2)) == pytest.approx(1.0, rel=1e-4)


def test_random_phase_keeps_magnitude():
    w = noise_window(5)
    out = A.random_phase(w, rng(6))
    assert np.allclose(np.abs(out), np.abs(w), atol=1e-5)
    ratio = out / w
    assert np.allclose(ratio, ratio[0], atol=1e-4)


def test_random_gain_keeps_phase_within_bounds():
    w = noise_window(7)
    for seed in range(10):
        out = A.random_gain(w, rng(seed), gain_db=3.0)
        ratio = out / w
        assert np.allclose(ratio.imag, 0, atol=1e-5)
        gain_db = 20 * np.log10(ratio.real.mean())
        assert -3.0 - 1e-4 <= gain_db <= 3.0 + 1e-4


def test_resample_jitter_scales_tone_frequency_within_one_percent():
    f = 3e6
    w = tone(f)
    seen = []
    for seed in range(10):
        out = A.resample_jitter(w, rng(seed), jitter=0.01)
        assert out.shape == w.shape and out.dtype == np.complex64
        seen.append(peak_hz(out) / f)
    seen = np.array(seen)
    assert np.all(np.abs(seen - 1) <= 0.0105)
    assert np.ptp(seen) > 0.001
    assert np.array_equal(A.resample_jitter(w, rng(), jitter=0.0), w)


def test_multipath_has_two_or_three_taps_direct_path_first():
    impulse = np.zeros(N, dtype=SAMPLE_DTYPE)
    impulse[100] = 1.0
    counts = set()
    for seed in range(20):
        out = A.multipath(impulse, rng(seed))
        taps = np.flatnonzero(np.abs(out) > 1e-6)
        counts.add(len(taps))
        assert taps[0] == 100 and taps[-1] <= 108
        assert np.abs(out[taps[0]]) == np.abs(out[taps]).max()
    assert counts == {2, 3}


def test_interference_mix_hits_the_sir():
    w, other = noise_window(8), noise_window(9)
    mixed, sir = A.interference_mix(w, other, rng(1), (10.0, 10.0))
    assert sir == pytest.approx(10.0)
    added = mixed - w
    assert 10 * np.log10(np.mean(np.abs(w) ** 2) / np.mean(np.abs(added) ** 2)) == pytest.approx(10.0, abs=1e-3)


def test_functions_are_pure_and_deterministic():
    w = noise_window(10)
    keep = w.copy()
    for fn in (A.add_noise_to_snr, A.frequency_offset, A.time_shift, A.time_crop, A.random_phase,
               A.random_gain, A.resample_jitter, A.multipath):
        first, second = fn(w, rng(42)), fn(w, rng(42))
        assert np.array_equal(first, second), fn.__name__
        assert first.dtype == np.complex64 and first.shape == (N,), fn.__name__
        assert np.array_equal(w, keep), fn.__name__


def test_augmenter_train_only_and_deterministic():
    aug = A.Augmenter()
    w = noise_window(11)
    for split in ("val", "test", "cross_test"):
        assert np.array_equal(aug(w, rng(1), make_row(split)), w)
    out = aug(w, rng(1), make_row("train", snr_db_est=25.0, bandwidth_hz=5e6))
    assert not np.array_equal(out, w)
    assert np.array_equal(out, aug(w, rng(1), make_row("train", snr_db_est=25.0, bandwidth_hz=5e6)))
    assert np.array_equal(dataclasses.replace(aug, enabled=False)(w, rng(1), make_row()), w)


def test_augmenter_from_cnn1d_config():
    with open(CONFIGS / "cnn1d.toml", "rb") as handle:
        section = tomllib.load(handle)["augment"]
    aug = A.Augmenter.from_config(section, multipath_fraction=0.0)
    assert aug.snr_db_range == (0.0, 30.0)
    assert aug.freq_offset_fraction == pytest.approx(0.05)
    assert aug.resample_jitter == pytest.approx(0.01)
    assert aug.multipath_fraction == 0.0
    assert A.Augmenter.from_config(None).enabled
