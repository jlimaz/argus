"""Runs only where PyTorch is installed (CRC); skipped on the laptop and in CI."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mcls.dataset import Augmenter, WindowStore, build  # noqa: E402
from mcls.dataset.fakes import make_fake_shards  # noqa: E402
from mcls.dataset.torch_dataset import BalancedTorchSampler, WindowDataset  # noqa: E402

CLASSES = ("drone_link", "wifi")


def test_items_shapes_and_train_only_augmentation(tmp_path):
    make_fake_shards(tmp_path)
    rows = build([tmp_path], seed=1, cap=0)
    store = WindowStore()
    plain = WindowDataset(rows, store, CLASSES)
    aug = WindowDataset(rows, store, CLASSES, augmenter=Augmenter(), seed=3)
    for i in range(len(rows)):
        item = plain[i]
        assert item["iq"].shape == (2, 4096) and item["iq"].dtype == torch.float32
        assert item["label"].dtype == torch.int64
        assert float(item["weight"]) in (0.5, 1.0)
        same = torch.equal(item["iq"], aug[i]["iq"])
        assert same == (rows[i].split != "train")
    i = next(k for k, r in enumerate(rows) if r.split == "train")
    assert torch.equal(aug[i]["iq"], aug[i]["iq"])
    aug.set_epoch(1)
    first = aug[i]["iq"]
    aug.set_epoch(0)
    assert not torch.equal(first, aug[i]["iq"])


def test_stft_view_and_sampler(tmp_path):
    make_fake_shards(tmp_path)
    rows = build([tmp_path], seed=1, cap=0)
    ds = WindowDataset(rows, WindowStore(), CLASSES, stft=lambda w: np.zeros((len(w), 1, 64, 64), np.float32))
    assert ds[0]["stft"].shape == (1, 64, 64)
    sampler = BalancedTorchSampler(ds.labels, seed=2)
    assert list(sampler) == list(BalancedTorchSampler(ds.labels, seed=2))
