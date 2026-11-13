"""Model tests that need torch (skipped on the laptop, run on CRC)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcls import models  # noqa: E402
from mcls.contracts import WINDOW_SAMPLES, Classifier, check_probabilities  # noqa: E402


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_cnn1d_parameter_count_matches_formula_and_plan():
    model = models.build("cnn1d", 4)
    assert _params(model) == models.cnn1d_param_count(32, 4) == 1_117_316
    assert 0.9e6 <= _params(model) <= 1.2e6
    for width, k, fam in ((16, 9, 0), (32, 4, 5), (48, 4, 0)):
        model = models.build("cnn1d", k, width=width, num_families=fam)
        assert _params(model) == models.cnn1d_param_count(width, k, fam)


def test_cnn1d_layout_follows_plan():
    from mcls.models.cnn1d import CNN1D

    model = CNN1D(num_classes=4)
    stem = model.backbone.stem[0]
    assert (stem.in_channels, stem.out_channels, stem.kernel_size, stem.stride) == (2, 32, (7,), (2,))
    assert model.backbone.feature_dim == 512
    assert model.head.dropout.p == 0.2 and model.head.classes.in_features == 512
    widths = [stage[0].conv1.out_channels for stage in model.backbone.stages]
    kernels = [stage[0].conv1.kernel_size[0] for stage in model.backbone.stages]
    assert widths == [32, 64, 128, 256] and kernels == [5, 5, 5, 3]


@pytest.mark.parametrize("name", models.NEURAL_MODELS)
def test_output_shapes(name):
    x = torch.randn(3, 2, WINDOW_SAMPLES)
    model = models.build(name, 4, width=8).eval()
    with torch.no_grad():
        assert model(x).shape == (3, 4)
    model = models.build(name, 4, width=8, num_families=len(models.FAMILIES)).eval()
    with torch.no_grad():
        logits, family = model(x)
    assert logits.shape == (3, 4) and family.shape == (3, len(models.FAMILIES))


@pytest.mark.parametrize("name", models.NEURAL_MODELS)
def test_backward_runs(name):
    model = models.build(name, 3, width=4).train()
    loss = model(torch.randn(4, 2, WINDOW_SAMPLES)).logsumexp(dim=1).mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_cnn2d_and_fused_sizes_are_reported():
    # No size target in the plan for the ablations; keep them within an order of magnitude of cnn1d.
    for name in ("cnn2d", "fused"):
        assert 1e5 < _params(models.build(name, 4)) < 5e6


def test_spectrogram_matches_numpy_reference():
    from mcls.models.cnn2d import Spectrogram

    rng = np.random.default_rng(0)
    windows = (rng.standard_normal((3, WINDOW_SAMPLES)) + 1j * rng.standard_normal((3, WINDOW_SAMPLES)))
    n = np.arange(WINDOW_SAMPLES)
    windows[0] += 4 * np.exp(2j * np.pi * 0.1 * n)
    iq = models.iq_array((windows / np.sqrt(2)).astype(np.complex64))
    expected = models.spectrogram_np(iq)
    got = Spectrogram()(torch.from_numpy(iq)).numpy()
    assert got.shape == expected.shape == (3, 1, 64, 64)
    np.testing.assert_allclose(got, expected, atol=2e-3)


def test_head_helpers():
    from mcls.models.heads import avg_max_pool, main_logits

    x = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    assert avg_max_pool(x).tolist() == [[2.5, 8.5, 5.0, 11.0]]
    a = torch.zeros(2, 3)
    assert main_logits(a) is a and main_logits((a, torch.ones(2))) is a


def _write_checkpoint(path, name="cnn1d", classes=("a", "b", "c")):
    from mcls.train import checkpoint

    kwargs = {"num_classes": len(classes), "width": 4, "dropout": 0.2, "num_families": 0}
    torch.manual_seed(0)
    model = models.build(name, **kwargs)
    checkpoint.save_best(path, model_name=name, model_kwargs=kwargs, state_dict=model.state_dict(), classes=classes,
                         family_classes=(), config={"run": {"name": "t", "seed": 1}}, epoch=0, val_macro_f1=0.5,
                         meta={"git_commit": "abc", "manifest_sha256": "0" * 64})
    return model


@pytest.mark.parametrize("name", models.NEURAL_MODELS)
def test_classifier_from_checkpoint_meets_contract(tmp_path, name):
    from mcls.models.classifier import load_classifier

    path = tmp_path / "checkpoint.pt"
    model = _write_checkpoint(path, name)
    clf = load_classifier(path, batch_size=2)
    assert isinstance(clf, Classifier)
    assert clf.classes == ("a", "b", "c")
    rng = np.random.default_rng(1)
    windows = (rng.standard_normal((5, WINDOW_SAMPLES)) + 1j * rng.standard_normal((5, WINDOW_SAMPLES)))
    windows = (windows / np.sqrt(2)).astype(np.complex64)
    probs = clf.classify(windows)
    check_probabilities(probs, 5, clf.classes)
    with torch.no_grad():
        direct = model.eval()(torch.from_numpy(models.iq_array(windows))).numpy()
    np.testing.assert_allclose(clf.logits(windows), direct, rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError):
        clf.classify(windows[:, :100])


def test_calibration_temperature_applies_to_classify_only(tmp_path):
    import json

    from mcls.models.classifier import load_classifier, softmax

    path = tmp_path / "checkpoint.pt"
    _write_checkpoint(path)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"temperature": 4.0, "classes": ["a", "b", "c"]}))
    windows = np.ones((2, WINDOW_SAMPLES), dtype=np.complex64)
    plain = load_classifier(path)
    calibrated = load_classifier(path, calibration=calibration)
    assert plain.temperature == 1.0 and calibrated.temperature == 4.0
    np.testing.assert_array_equal(plain.logits(windows), calibrated.logits(windows))  # logits always raw
    np.testing.assert_allclose(calibrated.classify(windows), softmax(plain.logits(windows) / 4.0))
    assert (calibrated.classify(windows).max(axis=1) <= plain.classify(windows).max(axis=1) + 1e-6).all()


def test_checkpoint_loads_with_weights_only_and_has_no_temperature(tmp_path):
    from mcls.train import checkpoint

    path = tmp_path / "checkpoint.pt"
    _write_checkpoint(path)
    payload = torch.load(path, weights_only=True)
    assert payload["format"] == "mcls-checkpoint-v2" and "temperature" not in payload
    assert payload["classes"] == ["a", "b", "c"]
    assert not hasattr(checkpoint, "set_temperature")
