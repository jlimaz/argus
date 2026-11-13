"""PyTorch `Dataset` over manifest rows. Imports torch: CRC only, never on the laptop.

Each item is a dict:
    "iq"     float32 tensor [2, WINDOW_SAMPLES]  (I, Q)
    "label"  int64 scalar tensor (index into `classes`)
    "weight" float32 scalar tensor (1.0 strong, 0.5 weak; PLAN 5.3)
    "index"  int (row index, to map predictions back to window_id)
    "stft"   float32 tensor [1, 64, 64], only with stft=True (Engineer 4's
             mcls.frontend.views.stft_view) or an equivalent batch function

Augmentation runs only on rows whose split is "train". Its generator is seeded
from (seed, epoch, index), so call `set_epoch(e)` once per epoch; the same seed
and epoch give the same augmented window for an index, in any worker.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from mcls.contracts import ManifestRow
from mcls.dataset.augment import Augmenter
from mcls.dataset.rows import WEAK_WEIGHT, label_indices, row_weights
from mcls.dataset.sampler import BalancedSampler
from mcls.dataset.store import WindowStore


def iq_view(window: np.ndarray) -> np.ndarray:
    """complex64 [n] -> float32 [2, n] (I, Q)."""
    return np.stack([window.real, window.imag]).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        store: WindowStore,
        classes: Sequence[str],
        augmenter: Augmenter | None = None,
        seed: int = 0,
        stft: bool | Callable[[np.ndarray], np.ndarray] = False,
        weak_weight: float = WEAK_WEIGHT,
    ):
        if stft is True:
            # Engineer 4's view: (n, 4096) complex64 -> (n, 1, 64, 64) float32.
            from mcls.frontend.views import stft_view as stft
        elif stft is False:
            stft = None
        self.rows = list(rows)
        self.store = store
        self.classes = tuple(classes)
        self.labels = label_indices(self.rows, self.classes)
        self.weights = row_weights(self.rows, weak_weight)
        self.augmenter = augmenter
        self.seed = int(seed)
        self.stft = stft
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        window = self.store.window(row)
        if self.augmenter is not None and row.split == "train":
            rng = np.random.default_rng([self.seed, self.epoch, int(index)])
            window = self.augmenter(window, rng, row)
        item = {
            "iq": torch.from_numpy(iq_view(window)),
            "label": torch.tensor(self.labels[index], dtype=torch.int64),
            "weight": torch.tensor(self.weights[index], dtype=torch.float32),
            "index": int(index),
        }
        if self.stft is not None:
            view = np.asarray(self.stft(window[None, :]), dtype=np.float32)
            item["stft"] = torch.from_numpy(np.ascontiguousarray(view.reshape((1, *view.shape[-2:]))))
        return item


class BalancedTorchSampler(Sampler):
    """Wraps `BalancedSampler`; call `set_epoch(e)` before each epoch."""

    def __init__(self, labels: np.ndarray, seed: int, num_samples: int | None = None):
        self.inner = BalancedSampler(labels, seed, num_samples)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.inner)

    def __iter__(self):
        return iter(self.inner.epoch(self.epoch).tolist())
