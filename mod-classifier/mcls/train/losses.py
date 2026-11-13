"""Losses (PLAN.md 5.3, 5.6). Imports torch.

- Cross-entropy with label smoothing (0.1 in the plan) and a weight per row
  (strong 1.0, weak 0.5). The batch loss is sum(w_i * l_i) / sum(w_i).
- Focal loss (ablation): the same smoothed target, each row scaled by
  (1 - p_true) ** gamma. gamma 0 equals the cross-entropy above.
- Family loss for the auxiliary head: cross-entropy over rows whose family
  label is known (>= 0); 0 when none is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LOSSES = ("cross_entropy", "focal")  # the names mcls.config.LOSSES accepts


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return values.mean()
    weight = weight.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1e-12)


def classification_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None,
                        label_smoothing: float = 0.1, kind: str = "cross_entropy", focal_gamma: float = 2.0) -> torch.Tensor:
    if kind not in LOSSES:
        raise ValueError(f"unknown loss {kind!r}; expected one of {LOSSES}")
    logits = logits.float()
    if kind == "cross_entropy":
        per_row = F.cross_entropy(logits, target, reduction="none", label_smoothing=label_smoothing)
        return _weighted_mean(per_row, weight)
    log_p = F.log_softmax(logits, dim=1)
    k = logits.shape[1]
    smooth = torch.full_like(log_p, label_smoothing / k)
    smooth.scatter_add_(1, target[:, None], torch.full_like(log_p[:, :1], 1.0 - label_smoothing))
    ce = -(smooth * log_p).sum(dim=1)
    p_true = log_p.gather(1, target[:, None]).squeeze(1).exp()
    per_row = (1.0 - p_true).clamp_min(0.0) ** focal_gamma * ce
    return _weighted_mean(per_row, weight)


def family_loss(family_logits: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on rows with a known family (>= 0); a zero that keeps the graph when none is known."""
    known = family >= 0
    if not bool(known.any()):
        return family_logits.sum() * 0.0
    return F.cross_entropy(family_logits[known].float(), family[known])
