from __future__ import annotations

from typing import Dict

import torch


def _fast_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    with torch.no_grad():
        pred = pred.view(-1)
        target = target.view(-1)
        valid = (target >= 0) & (target < num_classes)
        pred = pred[valid]
        target = target[valid]
        indices = target * num_classes + pred
        hist = torch.bincount(indices, minlength=num_classes**2)
        return hist.view(num_classes, num_classes)


def metrics_from_confusion(
    confusion_matrix: torch.Tensor, beta: float = 1.0, eps: float = 1e-7
) -> Dict[str, float]:
    conf = confusion_matrix.double()
    tp = torch.diag(conf)
    total = conf.sum()
    pixel_acc = (tp.sum() / total).item() if total > 0 else 0.0

    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    beta_sq = beta**2
    f_beta = ((1 + beta_sq) * (precision * recall)) / ((beta_sq * precision) + recall + eps)

    # Align with reference behavior: average only classes present in ground truth.
    gt_present = conf.sum(dim=1) > 0
    f1_macro = f_beta[gt_present].mean().item() if gt_present.any() else 0.0

    return {
        "pixel_accuracy": float(pixel_acc),
        "f1_macro": float(f1_macro),
        "f_measure": float(f1_macro),
    }


class SegmentationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        cm = _fast_confusion_matrix(pred.cpu(), target.cpu(), self.num_classes)
        self.confusion += cm.double()

    def compute(self) -> Dict[str, float]:
        return metrics_from_confusion(self.confusion)
