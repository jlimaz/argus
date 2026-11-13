"""Every training setting the loop reads, with its default (PLAN.md 5.3, 5.6). No torch.

One place for defaults, so they are tested on the laptop and the loop never
spells a default twice.
"""

from __future__ import annotations

DEFAULTS = {
    "optimizer": "adamw",
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "momentum": 0.9,
    "nesterov": True,
    "schedule": "onecycle",
    "epochs": 40,
    "batch_size": 256,
    "label_smoothing": 0.1,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "early_stop_patience": 8,
    "loss": "cross_entropy",
    "focal_gamma": 2.0,
    "balanced_sampler": True,
    "aux_weight": 0.5,
    "aux_head": False,
    "width": 32,
    "dropout": 0.2,
}

#: option name -> (config section, key)
SOURCES = {name: ("optim", name) for name in DEFAULTS}
SOURCES.update({"optimizer": ("optim", "name"), "aux_head": ("model", "aux_head"),
                "width": ("model", "width"), "dropout": ("model", "dropout")})


def options(config: dict) -> dict:
    """The effective settings of a run: config values where given, DEFAULTS otherwise."""
    out = {}
    for name, default in DEFAULTS.items():
        section, key = SOURCES[name]
        value = config.get(section, {}).get(key, default)
        out[name] = type(default)(value) if isinstance(default, (int, float)) and not isinstance(default, bool) \
            else value
    return out
