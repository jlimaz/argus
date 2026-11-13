"""The data pipeline on real front-end output (Engineer 4's process_recording), synthetic signal."""

import numpy as np

from mcls import manifest
from mcls.contracts import Recording
from mcls.dataset import Augmenter, WindowStore, audit_window_hashes, build
from mcls.dataset import build_manifest
from mcls.frontend.process import FrontendParams, process_recording
from mcls.frontend.views import stft_view

CLASSES = ("drone_link", "wifi", "bluetooth", "background")


def recording(recording_id, label, seed):
    rate, total = 40e6, 1_200_000
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(total) + 1j * rng.standard_normal(total)) * 0.01
    burst = rng.standard_normal(300_000) + 1j * rng.standard_normal(300_000)
    x[200_000:500_000] += np.convolve(burst, np.ones(8) / np.sqrt(8), mode="same")
    x = x.astype(np.complex64)
    return Recording(
        dataset="fake", recording_id=recording_id, source_file=f"{recording_id}.bin", sample_rate_hz=rate,
        center_frequency_hz=2.44e9, num_samples=total, label=label, label_strength="strong", is_real=True,
        license="CC BY 4.0", read=lambda s, c: x[s:s + c].copy(), emitter_id=f"model_{seed % 2}",
    )


def test_front_end_rows_build_into_a_manifest(tmp_path):
    shards = tmp_path / "shards"
    for i in range(6):
        rec = recording(f"rec{i}", ("drone_link", "wifi")[i % 2], seed=i)
        process_recording(rec, shards, None, FrontendParams(max_windows_per_burst=4))
    rows = build([shards], seed=2, cap=0)
    assert len(rows) == 6 * 4
    manifest.check_leakage(rows)
    store = WindowStore()
    assert audit_window_hashes(rows, store) == []
    windows, labels = store.get(rows, CLASSES)
    assert windows.shape == (24, 4096) and set(labels.tolist()) == {0, 1}
    train = next(r for r in rows if r.split == "train")
    out = Augmenter()(store.window(train), np.random.default_rng(0), train)
    assert stft_view(out[None]).shape == (1, 1, 64, 64)

    target = tmp_path / "data" / "manifest.jsonl"
    argv = ["--shards", str(shards), "--out", str(target), "--seed", "2", "--classes", "v1"]
    assert build_manifest.main(argv) == 0
    first = target.read_bytes()
    assert build_manifest.main(argv) == 0
    assert target.read_bytes() == first
    reread = manifest.read(target, CLASSES)
    assert np.array_equal(WindowStore(target.parent).windows(reread), store.windows(sorted(rows, key=lambda r: r.window_id)))


def test_always_on_jammer_gets_windows_where_burst_logic_gives_none(tmp_path):
    # A constant-power tone over the whole recording, as GATEMAN's jammer files are.
    rate, total = 20e6, 8_400_000
    t = np.arange(total)
    tone = (np.exp(2j * np.pi * 1e6 / rate * t) + 0.01 * np.random.default_rng(0).standard_normal(total)).astype(np.complex64)

    def rec(label):
        return Recording(
            dataset="fake", recording_id=f"jam_{label}", source_file=f"jam_{label}.bin", sample_rate_hz=rate,
            center_frequency_hz=1.57542e9, num_samples=total, label=label, label_strength="weak", is_real=True,
            license="CC BY 4.0", read=lambda s, c: tone[s:s + c].copy(),
        )

    jammer = process_recording(rec("gnss_jammer"), tmp_path / "a", None, FrontendParams(max_windows_per_burst=8))
    assert jammer["windows"] > 0
    as_burst = process_recording(rec("lora"), tmp_path / "b", None, FrontendParams(max_windows_per_burst=8))
    assert as_burst["windows"] == 0  # the burst path rejects an always-on signal: why CONTINUOUS exists


def test_short_radar_pulses_get_windows(tmp_path):
    # 10 us pulses every 300 us in noise: a few percent of each 205 us window is active.
    rate, total = 10e6, 800_000
    rng = np.random.default_rng(1)
    x = (0.01 * (rng.standard_normal(total) + 1j * rng.standard_normal(total))).astype(np.complex64)
    for start in range(1000, total - 200, 3000):
        x[start:start + 100] += np.exp(2j * np.pi * 0.1 * np.arange(100)).astype(np.complex64)

    def rec(label):
        return Recording(
            dataset="fake", recording_id=f"radar_{label}", source_file=f"radar_{label}.bin", sample_rate_hz=rate,
            center_frequency_hz=3.6e9, num_samples=total, label=label, label_strength="weak", is_real=False,
            license="CC BY 4.0", read=lambda s, c: x[s:s + c].copy(),
        )

    radar = process_recording(rec("pulsed_radar"), tmp_path / "a", None, FrontendParams(max_windows_per_burst=8))
    assert radar["windows"] > 0
