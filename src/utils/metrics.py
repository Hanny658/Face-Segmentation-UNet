from __future__ import annotations

from typing import Any, Dict

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
) -> Dict[str, Any]:
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

    # Per-class score from dataset-level confusion (kept for detailed reporting).
    gt_present = conf.sum(dim=1) > 0
    f_score_confusion = f_beta[gt_present].mean().item() if gt_present.any() else 0.0

    return {
        "pixel_accuracy": float(pixel_acc),
        "f_score_confusion": float(f_score_confusion),
        "f_score_per_class": [float(x) for x in f_beta.tolist()],
        "gt_present": [bool(x) for x in gt_present.tolist()],
    }


def compute_multiclass_fscore(mask_gt: torch.Tensor, mask_pred: torch.Tensor, beta: float = 1.0) -> float:
    """Strictly match reference function semantics on a single sample."""
    eps = 1e-7
    beta_sq = beta**2
    f_scores = []
    for class_id in torch.unique(mask_gt):
        gt_mask = mask_gt == class_id
        pred_mask = mask_pred == class_id

        tp = (gt_mask & pred_mask).sum().double()
        fp = ((~gt_mask) & pred_mask).sum().double()
        fn = (gt_mask & (~pred_mask)).sum().double()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f_score = ((1 + beta_sq) * (precision * recall)) / ((beta_sq * precision) + recall + eps)
        f_scores.append(f_score)

    if not f_scores:
        return 0.0
    return float(torch.stack(f_scores).mean().item())


class SegmentationMeter:
    def __init__(self, num_classes: int, beta: float = 1.0) -> None:
        self.num_classes = num_classes
        self.beta = beta
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
        self.sample_fscore_sum = 0.0
        self.sample_count = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        pred_cpu = pred.cpu()
        target_cpu = target.cpu()
        cm = _fast_confusion_matrix(pred_cpu, target_cpu, self.num_classes)
        self.confusion += cm.double()
        for mask_gt, mask_pred in zip(target_cpu, pred_cpu):
            self.sample_fscore_sum += compute_multiclass_fscore(mask_gt, mask_pred, beta=self.beta)
            self.sample_count += 1

    def compute(self) -> Dict[str, Any]:
        metrics = metrics_from_confusion(self.confusion, beta=self.beta)
        f_score = self.sample_fscore_sum / self.sample_count if self.sample_count > 0 else 0.0

        # Primary metric now follows strict per-image reference function.
        metrics["f_score"] = float(f_score)
        metrics["f_measure"] = float(f_score)

        # Backward-compatible aliases.
        metrics["f1_macro"] = float(f_score)
        metrics["f1_per_class"] = metrics["f_score_per_class"]
        return metrics
