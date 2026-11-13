"""Checkpoints. Imports torch.

checkpoint.pt (the model to evaluate and deploy) holds only tensors and plain
Python values, so it loads with torch.load(weights_only=True):

    {"format": "mcls-checkpoint-v2", "model": "cnn1d", "model_kwargs": {...},
     "state_dict": EMA weights, "classes": [...], "family_classes": [...],
     "config": {...}, "epoch": int, "val_macro_f1": float,
     "git_commit": str, "manifest_sha256": str}

No temperature is stored here. Engineer 8's calibration.json is the single
place a temperature lives; pass its path to
mcls.models.classifier.load_classifier(path, calibration=...). (v1, which
carried a temperature placeholder, was never used for a run.)

last.pt is the full training state for --resume (optimizer, scheduler, scaler,
EMA, early stopping, random states); it holds numpy random state, so it loads
with weights_only=False and must only be read from runs this project wrote.
"""

from __future__ import annotations

from pathlib import Path

import torch

FORMAT = "mcls-checkpoint-v2"
BEST = "checkpoint.pt"
LAST = "last.pt"


def save_best(path: Path, *, model_name: str, model_kwargs: dict, state_dict: dict, classes, family_classes,
              config: dict, epoch: int, val_macro_f1: float, meta: dict) -> None:
    payload = {
        "format": FORMAT,
        "model": model_name,
        "model_kwargs": dict(model_kwargs),
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "classes": list(classes),
        "family_classes": list(family_classes),
        "config": config,
        "epoch": int(epoch),
        "val_macro_f1": float(val_macro_f1),
        "git_commit": str(meta.get("git_commit", "unknown")),
        "manifest_sha256": str(meta.get("manifest_sha256", "")),
    }
    _atomic_save(payload, path)


def load_best(path: Path, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path}: not an {FORMAT} checkpoint")
    return payload


def build_from(payload: dict):
    """The model a checkpoint describes, with its weights loaded, in eval mode."""
    from mcls.models import build

    model = build(payload["model"], **payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def save_last(path: Path, state: dict) -> None:
    _atomic_save(state, path)


def load_last(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def _atomic_save(obj, path: Path) -> None:
    """Write to a temporary file then rename, so a job killed mid-save leaves the old file intact."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)
