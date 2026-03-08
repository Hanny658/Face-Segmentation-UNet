from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import SegmentationMeter
from src.utils.tta import predict_with_tta


@torch.no_grad() # not for evaluation runtime
def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    use_amp: bool = True,
    tta_enabled: bool = False,
    tta_flip: bool = False,
    tta_scales: Sequence[float] | None = None,
    flip_pairs: Sequence[Tuple[int, int]] | None = None,
    desc: str = "eval",
) -> Dict[str, Any]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes)
    losses = []

    amp_enabled = use_amp and device.type == "cuda"
    for batch in tqdm(data_loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = predict_with_tta(
                model=model,
                images=images,
                use_tta=bool(tta_enabled),
                tta_flip=bool(tta_flip),
                tta_scales=tta_scales,
                flip_pairs=flip_pairs,
            )
            loss, _ = criterion(logits, masks)

        losses.append(loss.item())
        meter.update(logits, masks)

    metrics: Dict[str, Any] = meter.compute()
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics
