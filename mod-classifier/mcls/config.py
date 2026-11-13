"""Run configs: TOML files in configs/, checked before a job starts.

A typo in a config should fail in a second on the laptop, not after hours in
the CRC queue, so every key is checked here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

MODELS = ("softmax_regression", "lightgbm", "knn", "majority", "cnn1d", "cnn2d", "fused")
OPTIMIZERS = ("adamw", "sgd")
SCHEDULES = ("onecycle", "cosine", "constant")
LOSSES = ("cross_entropy", "focal")
DEVICES = ("auto", "cuda", "mps", "cpu")

#: section -> key -> (type, required)
SCHEMA: dict[str, dict[str, tuple[type | tuple[type, ...], bool]]] = {
    "run": {"name": (str, True), "seed": (int, True), "out_dir": (str, True), "device": (str, False)},
    "data": {
        "manifest": (str, True),
        "classes": (str, True),
        "max_windows_per_class": (int, False),
        "cross_test_datasets": (list, False),
        "use_simulated": (bool, False),
    },
    "model": {
        "name": (str, True),
        "dropout": ((int, float), False),
        "width": (int, False),
        "aux_head": (bool, False),
    },
    "optim": {
        "name": (str, False),
        "lr": ((int, float), False),
        "weight_decay": ((int, float), False),
        "momentum": ((int, float), False),
        "schedule": (str, False),
        "epochs": (int, False),
        "batch_size": (int, False),
        "label_smoothing": ((int, float), False),
        "grad_clip": ((int, float), False),
        "ema_decay": ((int, float), False),
        "early_stop_patience": (int, False),
        "loss": (str, False),
        "focal_gamma": ((int, float), False),
        "nesterov": (bool, False),
        "balanced_sampler": (bool, False),
        "aux_weight": ((int, float), False),
    },
    "augment": {
        "enabled": (bool, False),
        "snr_db_range": (list, False),
        "freq_offset_fraction": ((int, float), False),
        "time_shift": (bool, False),
        "random_phase": (bool, False),
        "resample_jitter": ((int, float), False),
    },
}
REQUIRED_SECTIONS = ("run", "data", "model")


def load(path: Path) -> dict:
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    validate(config, path.name)
    return config


def validate(config: dict, name: str = "config") -> None:
    """Raise ValueError naming the first problem found."""
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"{name}: missing [{section}]")
    for section, values in config.items():
        if section not in SCHEMA:
            raise ValueError(f"{name}: unknown section [{section}]")
        allowed = SCHEMA[section]
        for key in values:
            if key not in allowed:
                raise ValueError(f"{name}: unknown key {section}.{key}")
        for key, (kind, required) in allowed.items():
            if key not in values:
                if required:
                    raise ValueError(f"{name}: missing {section}.{key}")
                continue
            value = values[key]
            kinds = kind if isinstance(kind, tuple) else (kind,)
            # bool is a subclass of int in Python, so `epochs = true` would otherwise pass.
            if isinstance(value, bool) and bool not in kinds:
                raise ValueError(f"{name}: {section}.{key} must be {kind}, got a boolean")
            if not isinstance(value, kind):
                raise ValueError(f"{name}: {section}.{key} must be {kind}, got {type(value).__name__}")
    _check_choices(config, name)


def _check_choices(config: dict, name: str) -> None:
    model = config["model"]["name"]
    if model not in MODELS:
        raise ValueError(f"{name}: model.name must be one of {MODELS}")
    from mcls import taxonomy  # here, not at the top: taxonomy is only needed for this check

    known = taxonomy.load().class_set_names
    if config["data"]["classes"] not in known:
        raise ValueError(f"{name}: data.classes must be one of {', '.join(known)}")
    device = config["run"].get("device", "auto")
    if device not in DEVICES:
        raise ValueError(f"{name}: run.device must be one of {DEVICES}")
    optim = config.get("optim", {})
    if "name" in optim and optim["name"] not in OPTIMIZERS:
        raise ValueError(f"{name}: optim.name must be one of {OPTIMIZERS}")
    if "loss" in optim and optim["loss"] not in LOSSES:
        raise ValueError(f"{name}: optim.loss must be one of {LOSSES}")
    if "schedule" in optim and optim["schedule"] not in SCHEDULES:
        raise ValueError(f"{name}: optim.schedule must be one of {SCHEDULES}")
    for key in ("lr", "epochs", "batch_size"):
        if key in optim and optim[key] <= 0:
            raise ValueError(f"{name}: optim.{key} must be positive")
    snr = config.get("augment", {}).get("snr_db_range")
    if snr is not None and (len(snr) != 2 or snr[0] > snr[1]):
        raise ValueError(f"{name}: augment.snr_db_range must be [low, high]")
