from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .dice import MultiClassDiceLoss


class SegmentationLoss(nn.Module):
    """Total loss = CrossEntropy + dice_weight * Dice."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ignore_index: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        dice_present_only: bool = True,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            class_weights = class_weights.float()
        if ignore_index is None:
            self.ce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = MultiClassDiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            present_only=dice_present_only,
        )
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        total_loss = ce_loss + self.dice_weight * dice_loss
        components = {
            "ce": ce_loss.detach(),
            "dice": dice_loss.detach(),
            "total": total_loss.detach(),
        }
        return total_loss, components
