"""Shared contrastive objectives used by alignment and joint Grounding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_positive_contrastive_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    item_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor | None:
    """Symmetric InfoNCE where repeated catalogue IDs are all positives."""

    if item_ids.numel() < 2 or item_ids.unique().numel() < 2:
        return None
    logits = F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).t()
    logits = logits / temperature
    positives = item_ids[:, None].eq(item_ids[None, :])
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    left_loss = -(
        torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)
    ).mean()
    right_loss = -(
        torch.logsumexp(positive_logits.t(), dim=1) - torch.logsumexp(logits.t(), dim=1)
    ).mean()
    return (left_loss + right_loss) / 2


__all__ = ["multi_positive_contrastive_loss"]
