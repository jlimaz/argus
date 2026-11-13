import numpy as np
import pytest

from mcls.contracts import SAMPLE_DTYPE, WINDOW_SAMPLES, ManifestRow
from mcls.dataset import BalancedSampler, WindowStore, class_weights, collect_rows
from mcls.dataset.fakes import make_fake_shards


def row(shard, offset, label="wifi"):
    return ManifestRow(
        window_id=f"w{offset}", shard=shard, offset=offset, length=WINDOW_SAMPLES, dataset="d",
        recording_id="r", label=label, label_strength="strong", native_rate_hz=20e6,
        center_frequency_hz=2.44e9, band="ism_2400", is_real=True, license="x", split="train",
    )


def test_store_reads_1d_and_2d_shards(tmp_path):
    flat = (np.arange(3 * WINDOW_SAMPLES) * (1 + 1j)).astype(SAMPLE_DTYPE)
    np.save(tmp_path / "flat.npy", flat)
    np.save(tmp_path / "grid.npy", flat.reshape(3, WINDOW_SAMPLES))
    store = WindowStore(tmp_path)
    for name in ("flat.npy", "grid.npy"):
        rows = [row(name, 2 * WINDOW_SAMPLES), row(name, 100, "drone_link")]
        windows, labels = store.get(rows, ("drone_link", "wifi"))
        assert windows.dtype == np.complex64 and windows.shape == (2, WINDOW_SAMPLES)
        assert np.array_equal(windows[0], flat[2 * WINDOW_SAMPLES:])
        assert np.array_equal(windows[1], flat[100:100 + WINDOW_SAMPLES])
        assert labels.tolist() == [1, 0] and labels.dtype == np.int64


def test_store_uses_memory_map(tmp_path):
    np.save(tmp_path / "s.npy", np.zeros(WINDOW_SAMPLES, dtype=SAMPLE_DTYPE))
    store = WindowStore(tmp_path)
    store.window(row("s.npy", 0))
    (mapped,) = store._open.values()
    assert isinstance(mapped.base, np.memmap) or isinstance(mapped, np.memmap)


def test_store_rejects_out_of_range_and_wrong_dtype(tmp_path):
    np.save(tmp_path / "s.npy", np.zeros(WINDOW_SAMPLES, dtype=SAMPLE_DTYPE))
    np.save(tmp_path / "f.npy", np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    store = WindowStore(tmp_path)
    with pytest.raises(ValueError, match="outside shard"):
        store.window(row("s.npy", 1))
    with pytest.raises(ValueError, match="dtype"):
        store.window(row("f.npy", 0))


def test_store_matches_fake_shards(tmp_path):
    make_fake_shards(tmp_path, seed=3)
    rows = collect_rows([tmp_path])
    windows = WindowStore().windows(rows)
    rms = np.sqrt(np.mean(np.abs(windows) ** 2, axis=1))
    assert np.allclose(rms, 1.0, atol=1e-4)
    assert len({w.tobytes() for w in windows}) == len(rows)


def test_balanced_sampler_balances_and_is_deterministic():
    labels = np.array([0] * 900 + [1] * 90 + [2] * 10)
    sampler = BalancedSampler(labels, seed=11, num_samples=30000)
    order = sampler.epoch(0)
    assert order.dtype == np.int64 and len(order) == 30000
    share = np.bincount(labels[order], minlength=3) / len(order)
    assert np.allclose(share, 1 / 3, atol=0.02)
    assert np.array_equal(order, BalancedSampler(labels, seed=11, num_samples=30000).epoch(0))
    assert not np.array_equal(order, sampler.epoch(1))
    assert not np.array_equal(order, BalancedSampler(labels, seed=12, num_samples=30000).epoch(0))
    assert len(BalancedSampler(labels, seed=1)) == len(labels)


def test_class_weights_inverse_frequency():
    weights = class_weights(np.array([0, 0, 0, 1]), 3)
    assert weights[2] == 0
    assert weights[1] == pytest.approx(3 * weights[0])
    assert weights[:2].mean() == pytest.approx(1.0)
