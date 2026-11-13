"""`python -m pipeline.ingest --classify`: new fields with the flag, byte-identical output without it."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline.ingest import main

FS = 2e6  # a raw recording well below 20 MS/s, like an RTL-SDR
CENTRE = 433.92e6

# sha256 of the output of `main` on the fixtures below, recorded at d2b4208, before
# --classify existed. Without --classify the output must stay exactly this.
GOLDEN = {
    "raw": "bc73887a50a6ff7e8a340a901abcb1e86fdd08b5ac7d3851dc5d2c88eb302b38",
    "sigmf": "b3351bf2c0a944b006bfed450a94912ff352f371ce04479d8c83b9ff2284284b",
}


def capture(n=400_000, seed=7):
    """Noise with two tone bursts, as interleaved values."""
    rng = np.random.default_rng(seed)
    iq = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    t = np.arange(n) / FS
    for a, b, f in ((50_000, 90_000, 200e3), (250_000, 330_000, -300e3)):
        iq[a:b] += 0.5 * np.exp(2j * np.pi * f * t[a:b])
    return iq


def raw_file(tmp_path):
    iq = capture()
    values = np.empty(2 * len(iq), dtype=np.int8)
    values[0::2] = np.clip(np.round(iq.real * 127), -128, 127)
    values[1::2] = np.clip(np.round(iq.imag * 127), -128, 127)
    path = tmp_path / "capture.cs8"
    values.tofile(path)
    return path


def sigmf_file(tmp_path):
    iq = capture(seed=8).astype(np.complex64)
    data = tmp_path / "rec.sigmf-data"
    iq.view(np.float32).astype("<f4").tofile(data)
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": FS, "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0, "core:frequency": 2.437e9},
                     {"core:sample_start": 200_000, "core:frequency": 915e6}],
        "annotations": [],
    }
    (tmp_path / "rec.sigmf-meta").write_text(json.dumps(meta))
    return tmp_path / "rec.sigmf-meta"


def run(argv, tmp_path, name="out.jsonl"):
    out = tmp_path / name
    assert main([*argv, "-o", str(out)]) == 0
    return out.read_bytes()


def test_without_classify_output_is_byte_identical(tmp_path):
    raw = run([str(raw_file(tmp_path)), "--sample-rate", "2M", "--center-freq", "433.92M"], tmp_path, "raw.jsonl")
    sig = run([str(sigmf_file(tmp_path))], tmp_path, "sig.jsonl")
    assert hashlib.sha256(raw).hexdigest() == GOLDEN["raw"]
    assert hashlib.sha256(sig).hexdigest() == GOLDEN["sigmf"]
    for line in (raw + sig).decode().splitlines():
        row = json.loads(line)
        assert not {"signal_class", "class_confidence", "class_probs"} & set(row)



# -- with --classify ---------------------------------------------------------------

CLASSES = ("drone_link", "wifi", "bluetooth", "background")


def baseline_run(folder: Path, threshold: float = 0.5) -> Path:
    """A tiny Engineer 6 baseline (numpy only) in a run folder, with a calibration.json."""
    from pipeline.ingest import _infer

    _infer()  # puts mod-classifier/ on sys.path
    from mcls.baselines import features
    from mcls.baselines.classifier import FeatureClassifier
    from mcls.baselines.models import SoftmaxRegression
    from mcls.frontend.windows import normalize

    rng = np.random.default_rng(1)
    t = np.arange(4096) / 20e6
    noise = (rng.standard_normal((48, 4096)) + 1j * rng.standard_normal((48, 4096))) * 0.05
    noise[:24] += np.exp(2j * np.pi * rng.uniform(-3e6, 3e6, (24, 1)) * t)
    X = features.extract(normalize(noise.astype(np.complex64)))
    y = np.array([0] * 24 + [3] * 24)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std < 1e-9] = 1.0
    clf = FeatureClassifier(None, CLASSES, mean, std)
    clf.model = SoftmaxRegression(len(CLASSES), params={"epochs": 30}).fit(clf.standardize(X), y)
    folder.mkdir()
    clf.save(folder)
    (folder / "calibration.json").write_text(json.dumps(
        {"classes": list(CLASSES), "temperature": 1.0, "unknown_threshold": {"score": "max_softmax", "value": threshold, "applied_after_temperature": True}}
    ))
    return folder


def raw_20m(tmp_path):
    """A 20 MS/s .cs8 with one tone burst (needs no resampler)."""
    n = 400_000
    rng = np.random.default_rng(9)
    iq = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    t = np.arange(n) / 20e6
    iq[100_000:200_000] += 0.5 * np.exp(2j * np.pi * 1e6 * t[100_000:200_000])
    values = np.empty(2 * n, dtype=np.int8)
    values[0::2] = np.clip(np.round(iq.real * 127), -128, 127)
    values[1::2] = np.clip(np.round(iq.imag * 127), -128, 127)
    path = tmp_path / "burst20.cs8"
    values.tofile(path)
    return path


def test_classify_adds_three_fields_and_keeps_modulation(tmp_path):
    model = baseline_run(tmp_path / "run")
    path = raw_20m(tmp_path)
    plain = run([str(path), "--sample-rate", "20M", "--center-freq", "2.437G"], tmp_path, "plain.jsonl")
    out = run([str(path), "--sample-rate", "20M", "--center-freq", "2.437G", "--classify", "--model", str(model)],
              tmp_path, "classified.jsonl")
    plain_rows = [json.loads(line) for line in plain.decode().splitlines()]
    rows = [json.loads(line) for line in out.decode().splitlines()]
    assert len(rows) == len(plain_rows) >= 1
    for before, after in zip(plain_rows, rows):
        assert {k: after[k] for k in before} == before  # nothing existing changes
        assert set(after) - set(before) == {"signal_class", "class_confidence", "class_probs"}
        assert after["modulation"] == "unknown"
        assert after["signal_class"] in (*CLASSES, "unknown")
        assert set(after["class_probs"]) == set(CLASSES)
        assert abs(sum(after["class_probs"].values()) - 1) < 1e-3
        assert after["class_confidence"] == max(after["class_probs"].values())
    from pipeline.inputs import combine_observations

    combine_observations([("classified.jsonl", out.decode())])  # the Library still accepts it


def test_classify_threshold_one_makes_everything_unknown(tmp_path):
    model = baseline_run(tmp_path / "run", threshold=1.0)
    out = run([str(raw_20m(tmp_path)), "--sample-rate", "20M", "--center-freq", "2.437G", "--classify", "--model", str(model)],
              tmp_path)
    assert {json.loads(line)["signal_class"] for line in out.decode().splitlines()} == {"unknown"}


def test_classify_refusals(tmp_path, capsys):
    path = raw_20m(tmp_path)
    with pytest.raises(SystemExit):
        main([str(path), "--sample-rate", "20M", "--center-freq", "2.437G", "--classify"])
    sweep = tmp_path / "s.csv"
    sweep.write_text("x")
    with pytest.raises(SystemExit):
        main([str(sweep), "--classify", "--model", str(tmp_path)])
    model = baseline_run(tmp_path / "run")
    (model / "calibration.json").unlink()
    assert main([str(path), "--sample-rate", "20M", "--center-freq", "2.437G", "--classify", "--model", str(model)]) == 1
    assert "calibration" in capsys.readouterr().err


def test_classify_at_another_rate_and_sigmf(tmp_path):
    """2 MS/s raw and a two-capture SigMF go through Engineer 4's resampler; every burst gets a class."""
    model = baseline_run(tmp_path / "run")
    raw = run([str(raw_file(tmp_path)), "--sample-rate", "2M", "--center-freq", "433.92M", "--classify", "--model", str(model)],
              tmp_path, "raw.jsonl")
    sig = run([str(sigmf_file(tmp_path)), "--classify", "--model", str(model)], tmp_path, "sig.jsonl")
    for text in (raw, sig):
        rows = [json.loads(line) for line in text.decode().splitlines()]
        assert rows and all(r["signal_class"] in (*CLASSES, "unknown") for r in rows)

@pytest.mark.parametrize("change", [
    {"classes": ["wifi", "drone_link", "bluetooth", "background"]},   # another class order
    {"classes": ["drone_link", "wifi", "bluetooth"]},                 # another class list
    {"run": "some_other_run"},                                        # another run folder
    {"manifest_sha256": "f" * 64},                                    # another manifest
])
def test_classify_refuses_a_calibration_from_another_model(tmp_path, capsys, change):
    model = baseline_run(tmp_path / "run")
    (model / "meta.json").write_text(json.dumps({"manifest_sha256": "a" * 64, "classes": list(CLASSES)}))
    cal = json.loads((model / "calibration.json").read_text())
    cal.update({"run": "run", "manifest_sha256": "a" * 64})
    (model / "calibration.json").write_text(json.dumps(cal))
    args = [str(raw_20m(tmp_path)), "--sample-rate", "20M", "--center-freq", "2.437G", "--classify", "--model", str(model)]
    assert main([*args, "-o", str(tmp_path / "ok.jsonl")]) == 0  # the matching file is accepted
    cal.update(change)
    (model / "calibration.json").write_text(json.dumps(cal))
    assert main([*args, "-o", str(tmp_path / "bad.jsonl")]) == 1
    assert "calibration" in capsys.readouterr().err
    assert not (tmp_path / "bad.jsonl").exists()

def test_import_of_ingest_does_not_import_mcls_or_torch():
    import subprocess

    code = "import sys, pipeline.ingest; assert 'mcls' not in sys.modules and 'torch' not in sys.modules; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr


if __name__ == "__main__":  # print the golden hashes (run once on the commit before --classify)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        raw = run([str(raw_file(d)), "--sample-rate", "2M", "--center-freq", "433.92M"], d, "raw.jsonl")
        sig = run([str(sigmf_file(d))], d, "sig.jsonl")
        print("raw", hashlib.sha256(raw).hexdigest(), len(raw.splitlines()))
        print("sigmf", hashlib.sha256(sig).hexdigest(), len(sig.splitlines()))
        sys.stdout.write(sig.decode())
