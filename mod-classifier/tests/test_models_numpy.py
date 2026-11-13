"""Model helpers that need no torch: parameter count, input packing, spectrogram reference."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mcls import models
from mcls.contracts import WINDOW_SAMPLES


def test_importing_models_and_train_does_not_import_torch():
    code = (
        "import sys; import mcls.models, mcls.models.classifier, mcls.train, mcls.train.__main__, "
        "mcls.train.runs, mcls.train.schedules, mcls.train.stopping, mcls.train.device; "
        "assert 'torch' not in sys.modules, 'torch imported at module level'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parent.parent)


def test_cnn1d_has_about_one_million_parameters():
    # PLAN.md 5.2: "About 1 million parameters". Exact count for width 32, 4 classes:
    count = models.cnn1d_param_count(width=32, num_classes=4)
    assert count == 1_117_316
    assert 0.9e6 <= count <= 1.2e6


def test_param_count_grows_with_heads_and_width():
    base = models.cnn1d_param_count(32, 4)
    assert models.cnn1d_param_count(32, 5) - base == 512 + 1
    assert models.cnn1d_param_count(32, 4, num_families=5) - base == 5 * 512 + 5
    assert models.cnn1d_param_count(16, 4) < base < models.cnn1d_param_count(48, 4)


def test_stage_plan_matches_plan():
    assert models.stage_plan_1d(32) == [(32, 5), (64, 5), (128, 5), (256, 3)]


def test_iq_array_packs_i_then_q():
    windows = (np.arange(2 * WINDOW_SAMPLES, dtype=np.float32).reshape(2, -1) * (1 + 2j)).astype(np.complex64)
    iq = models.iq_array(windows)
    assert iq.shape == (2, 2, WINDOW_SAMPLES) and iq.dtype == np.float32
    np.testing.assert_array_equal(iq[:, 0], windows.real)
    np.testing.assert_array_equal(iq[:, 1], windows.imag)


def test_iq_array_rejects_wrong_shape():
    with pytest.raises(ValueError):
        models.iq_array(np.zeros((2, 100), dtype=np.complex64))


def test_spectrogram_puts_a_tone_in_the_right_bin_and_is_standardised():
    n = np.arange(WINDOW_SAMPLES)
    bin_offset = 8  # cycles per 64-sample frame above zero frequency
    tone = np.exp(2j * np.pi * bin_offset * n / models.SPEC_NFFT).astype(np.complex64)
    spec = models.spectrogram_np(models.iq_array(tone[None]))
    assert spec.shape == (1, 1, models.SPEC_FRAMES, models.SPEC_NFFT)
    peak_bins = spec[0, 0].argmax(axis=1)
    assert (peak_bins == models.SPEC_NFFT // 2 + bin_offset).all()  # zero frequency sits at the centre bin
    assert abs(spec.mean()) < 1e-4 and abs(spec.std() - 1) < 1e-3


def test_spectrogram_negative_frequency_lands_below_centre():
    n = np.arange(WINDOW_SAMPLES)
    tone = np.exp(-2j * np.pi * 5 * n / models.SPEC_NFFT).astype(np.complex64)
    spec = models.spectrogram_np(models.iq_array(tone[None]))
    assert (spec[0, 0].argmax(axis=1) == models.SPEC_NFFT // 2 - 5).all()


def test_build_rejects_unknown_names_without_torch():
    with pytest.raises(ValueError):
        models.build("transformer", 4)
    with pytest.raises(ValueError):
        models.build("cnn1d", 1)


def test_classifier_softmax_rows_sum_to_one():
    from mcls.models.classifier import softmax

    probs = softmax(np.array([[1000.0, 0.0, -5.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(probs[1], 1 / 3)
