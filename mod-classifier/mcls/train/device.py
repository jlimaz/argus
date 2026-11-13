"""Device choice: auto, cuda, mps or cpu (PLAN.md 5.3)."""

from __future__ import annotations

DEVICES = ("auto", "cuda", "mps", "cpu")


def pick_device(requested: str, cuda_available: bool, mps_available: bool) -> str:
    """The device to use. "auto" prefers cuda, then mps, then cpu; an unavailable explicit choice is an error."""
    if requested not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {requested!r}")
    if requested == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("device cuda requested but torch.cuda.is_available() is False")
    if requested == "mps" and not mps_available:
        raise RuntimeError("device mps requested but torch.backends.mps.is_available() is False")
    return requested


def resolve_device(requested: str = "auto"):
    """A torch.device for `requested`. Imports torch."""
    import torch

    mps = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps is not None and mps.is_available())
    return torch.device(pick_device(requested, torch.cuda.is_available(), mps_ok))
