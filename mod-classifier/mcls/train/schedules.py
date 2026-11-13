"""Learning-rate schedules (PLAN.md 5.3), as pure functions of the step. No torch.

The training loop wraps `lr_at` in a torch LambdaLR and steps it once per
optimizer step, so the same numbers are tested here without torch.

- onecycle: linear warm-up from peak/25 to peak over the first 30% of steps,
  then cosine decay to peak/(25 * 1e4). These are the defaults of
  torch.optim.lr_scheduler.OneCycleLR (pct_start 0.3, div_factor 25,
  final_div_factor 1e4), with a linear rather than cosine warm-up.
- cosine: cosine decay from peak to 0 over all steps.
- constant: peak throughout.
"""

from __future__ import annotations

import math

SCHEDULES = ("onecycle", "cosine", "constant")
WARMUP_FRACTION = 0.3
DIV_FACTOR = 25.0
FINAL_DIV_FACTOR = 1e4


def lr_at(step: int, total_steps: int, peak_lr: float, kind: str) -> float:
    """Learning rate at optimizer step `step` (0-based) of `total_steps`."""
    if kind not in SCHEDULES:
        raise ValueError(f"unknown schedule {kind!r}; expected one of {SCHEDULES}")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if kind == "constant":
        return peak_lr
    step = min(max(step, 0), total_steps - 1)
    if kind == "cosine":
        progress = step / max(total_steps - 1, 1)
        return peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
    start = peak_lr / DIV_FACTOR
    end = start / FINAL_DIV_FACTOR
    warmup = max(int(WARMUP_FRACTION * total_steps), 1)
    if step < warmup:
        return start + (peak_lr - start) * step / warmup
    progress = (step - warmup) / max(total_steps - 1 - warmup, 1)
    return end + (peak_lr - end) * 0.5 * (1 + math.cos(math.pi * progress))


def lambda_for(total_steps: int, peak_lr: float, kind: str):
    """The multiplier function torch's LambdaLR expects (base lr = peak_lr)."""
    return lambda step: lr_at(step, total_steps, peak_lr, kind) / peak_lr
