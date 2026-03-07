from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiClassDiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        eps: float = 1e-6,
        present_only: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps
        self.present_only = present_only

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        target_clamped = target.clamp(min=0, max=self.num_classes - 1)
        target_one_hot = F.one_hot(target_clamped, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).unsqueeze(1)
            probs = probs * valid_mask
            target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dims)
        denominator = probs.sum(dims) + target_one_hot.sum(dims)
        dice_per_class = (2.0 * intersection + self.eps) / (denominator + self.eps)
        if self.present_only:
            present = target_one_hot.sum(dims) > 0
            if present.any():
                return 1.0 - dice_per_class[present].mean()
        return 1.0 - dice_per_class.mean()
