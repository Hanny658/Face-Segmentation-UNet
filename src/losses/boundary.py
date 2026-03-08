from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F


# Defines and calculates a boundary loss based on the idea of "soft" boundaries derived from segmentation logits.
def logits_to_boundary_probability(logits: torch.Tensor, pred_scale: float = 4.0) -> torch.Tensor:
    """Build a soft boundary probability map from segmentation logits."""
    probs = F.softmax(logits, dim=1)

    dv = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
    dh = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)

    dv_up = F.pad(dv, (0, 0, 1, 0), mode="constant", value=0.0)
    dv_down = F.pad(dv, (0, 0, 0, 1), mode="constant", value=0.0)
    dh_left = F.pad(dh, (1, 0, 0, 0), mode="constant", value=0.0)
    dh_right = F.pad(dh, (0, 1, 0, 0), mode="constant", value=0.0)
    edge = torch.maximum(torch.maximum(dv_up, dv_down), torch.maximum(dh_left, dh_right))

    if pred_scale > 0:
        edge = 1.0 - torch.exp(-pred_scale * edge)
    return edge.clamp(0.0, 1.0)


def boundary_bce_from_logits(
    logits: torch.Tensor,
    boundary_target: torch.Tensor,
    pos_weight: Optional[float] = 4.0,
    pred_scale: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted BCE between soft boundary prediction and binary boundary target."""
    pred = logits_to_boundary_probability(logits, pred_scale=pred_scale).clamp(eps, 1.0 - eps)
    target = boundary_target.float()

    pos_loss = -target * torch.log(pred)
    neg_loss = -(1.0 - target) * torch.log(1.0 - pred)
    if pos_weight is not None:
        loss = float(pos_weight) * pos_loss + neg_loss
    else:
        loss = pos_loss + neg_loss
    return loss.mean()
