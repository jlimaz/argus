"""The Burst Router UI's service on synthetic input. Needs the UI venv (onnxruntime, scikit-learn,
lightgbm) and the copied wave 5 runs in runs/crc/, so it is skipped elsewhere."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for module in ("onnxruntime", "sklearn", "lightgbm", "fastapi"):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed (UI venv only)", allow_module_level=True)
if not (ROOT / "runs/crc/routing_w5.json").is_file():
    pytest.skip("runs/crc/routing_w5.json missing (copy the wave 5 runs first)", allow_module_level=True)

sys.path.insert(0, str(ROOT / "ui"))
import service  # noqa: E402


@pytest.fixture(scope="module")
def bank():
    return service.load_bank()


def cf32(iq):
    raw = np.empty(2 * len(iq), np.float32)
    raw[0::2], raw[1::2] = iq.real, iq.imag
    return raw.tobytes()


def test_bank_holds_three_models_with_one_class_order(bank):
    assert [m.label for m in bank.models] == ["2D CNN", "kNN", "LightGBM"]
    assert len(bank.classes) == 8 and "gnss_jammer" in bank.classes


def test_route_follows_the_table_and_falls_back(bank):
    name, why = service.route_for(bank, "ism_2400")
    assert name == bank.routing["routes"]["band"]["ism_2400"]["model"]
    assert "validation" in why
    name, why = service.route_for(bank, "a_band_never_seen")
    assert name == bank.routing["fallback_model"] and "no validation data" in why


def test_burst_is_found_routed_and_classified(bank):
    rng = np.random.default_rng(0)
    n = 1_000_000
    x = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    x[300_000:600_000] += rng.standard_normal(300_000) + 1j * rng.standard_normal(300_000)
    rec = service.read_upload({"t.cf32": cf32(x.astype(np.complex64))}, 20e6, 2.44e9)
    out = service.classify(bank, rec)
    assert out["bursts_found"] == 1
    burst = out["bursts"][0]
    assert burst["band"] == "ism_2400" and burst["windows"] > 0
    assert burst["chosen_model"] == bank.routing["routes"]["band"]["ism_2400"]["model"]
    assert set(burst["models"]) == {m.name for m in bank.models}
    assert burst["class"] in set(bank.classes) | {"unknown"}
    assert service.spectrogram_png(rec, out["bursts"])


def test_always_on_tone_still_gets_windows(bank):
    n = 1_000_000
    tone = np.exp(2j * np.pi * 0.05 * np.arange(n)).astype(np.complex64)
    tone += (0.01 * np.random.default_rng(1).standard_normal(n)).astype(np.complex64)
    rec = service.read_upload({"t.cf32": cf32(tone)}, 20e6, 1575.42e6)
    out = service.classify(bank, rec)
    assert out["bursts"] and all(b["windows"] > 0 for b in out["bursts"])


def test_inputs_are_checked():
    with pytest.raises(service.InputError, match="sample rate"):
        service.read_upload({"t.cf32": cf32(np.ones(5000, np.complex64))}, None, 1e9)
    with pytest.raises(service.InputError, match="file ending"):
        service.read_upload({"t.bin": b"\0" * 40000}, 1e6, 1e9)
    with pytest.raises(service.InputError, match="SigMF"):
        service.read_upload({"a.sigmf-meta": b"{}"}, None, None)
    rec = service.read_upload({"t.sigmf-meta": b'{"global": {"core:datatype": "cf32_le", "core:sample_rate": 250000},'
                                               b' "captures": [{"core:frequency": 862.5}]}',
                               "t.sigmf-data": cf32(np.ones(5000, np.complex64))}, None, None)
    assert rec.center_frequency_hz == 862.5e6  # MHz written where SigMF says Hz
