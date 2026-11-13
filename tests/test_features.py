import numpy as np
import pytest

from pipeline.features import detect_bursts

FS = 1_000_000.0
FC = 100e6
RNG = np.random.default_rng(7)


def noise(samples: int, level: float = 1.0) -> np.ndarray:
    return (level / np.sqrt(2)) * (RNG.standard_normal(samples) + 1j * RNG.standard_normal(samples))


def tone(samples: int, offset_hz: float, amplitude: float = 10.0) -> np.ndarray:
    t = np.arange(samples) / FS
    return amplitude * np.exp(2j * np.pi * offset_hz * t)


def pulse_train(total: int, starts: list[int], width: int, offset_hz: float, amplitude: float = 10.0) -> np.ndarray:
    iq = noise(total)
    for start in starts:
        iq[start : start + width] += tone(width, offset_hz, amplitude) * np.exp(2j * np.pi * offset_hz * start / FS)
    return iq.astype(np.complex64)


def test_continuous_tone_is_one_truncated_burst():
    iq = (noise(100_000) + tone(100_000, 50_000.0)).astype(np.complex64)
    bursts = detect_bursts(iq, FS, FC)
    assert len(bursts) == 1
    burst = bursts[0]
    assert burst["truncated"] is True
    assert burst["burst_duration_s"] is None
    assert burst["duty_cycle"] == pytest.approx(1.0, abs=0.02)
    assert burst["center_frequency_hz"] == pytest.approx(FC + 50_000.0, abs=3_000.0)
    assert burst["bandwidth_hz"] < 10_000.0
    assert burst["hopping"] is False
    assert burst["pulse_interval_s"] is None
    assert burst["modulation"] == "unknown"
    assert burst["snr_db"] > 10.0


def test_tone_with_quiet_edges_has_a_duration():
    total = 100_000
    bursts = detect_bursts(pulse_train(total, [20_000], 30_000, -100_000.0), FS, FC)
    assert len(bursts) == 1
    burst = bursts[0]
    assert burst["truncated"] is False
    assert burst["burst_duration_s"] == pytest.approx(0.030, rel=0.05)
    assert burst["center_frequency_hz"] == pytest.approx(FC - 100_000.0, abs=3_000.0)


def test_pulse_train_is_one_burst_with_pulse_interval():
    period, width = 1_000, 100  # 1 kHz PRF, 100 us pulses, duty 0.1
    starts = [10_000 + k * period for k in range(10)]
    bursts = detect_bursts(pulse_train(100_000, starts, width, 0.0), FS, FC)
    assert len(bursts) == 1
    burst = bursts[0]
    assert burst["pulse_interval_s"] == pytest.approx(1e-3, rel=0.05)
    assert burst["duty_cycle"] == pytest.approx(0.1 * 10 / (9 + 0.1), rel=0.15)
    assert burst["burst_duration_s"] == pytest.approx((9 * period + width) / FS, rel=0.05)
    assert burst["hopping"] is False
    assert burst["truncated"] is False


def test_noise_only_has_no_bursts():
    assert detect_bursts(noise(100_000).astype(np.complex64), FS, FC) == []


def test_alternating_tones_are_hopping():
    period, width = 1_000, 200
    iq = noise(100_000)
    for k in range(10):
        start = 10_000 + k * period
        offset = 200_000.0 if k % 2 else -200_000.0
        iq[start : start + width] += tone(width, offset)
    bursts = detect_bursts(iq.astype(np.complex64), FS, FC)
    assert len(bursts) == 1
    assert bursts[0]["hopping"] is True


def test_pulse_groups_far_apart_are_separate_bursts():
    period, width = 1_000, 100
    starts = [5_000 + k * period for k in range(5)] + [60_000 + k * period for k in range(5)]
    bursts = detect_bursts(pulse_train(100_000, starts, width, 0.0), FS, FC)
    assert len(bursts) == 2
    assert bursts[0]["start_s"] < bursts[1]["start_s"]
    assert all(b["pulse_interval_s"] == pytest.approx(1e-3, rel=0.05) for b in bursts)


def test_real_samples_become_baseband_iq():
    from pipeline.features import real_to_iq

    fs_real = 4_000_000.0
    t = np.arange(400_000) / fs_real
    real = 10 * np.cos(2 * np.pi * 300_000.0 * t) + RNG.standard_normal(len(t))
    iq, fs, offset = real_to_iq(real, fs_real)
    assert fs == fs_real / 2
    assert offset == fs_real / 4
    assert iq.dtype == np.complex64
    assert len(iq) == 200_000 - 512
    # A 300 kHz real tone sits at 300 kHz - fs/4 = -700 kHz in the IQ.
    bursts = detect_bursts(iq, fs, 50e6 + offset)
    assert len(bursts) == 1
    assert bursts[0]["center_frequency_hz"] == pytest.approx(50e6 + 300_000.0, abs=3_000.0)
    assert bursts[0]["truncated"] is True
