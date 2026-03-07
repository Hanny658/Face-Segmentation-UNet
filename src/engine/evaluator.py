from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from src.utils.metrics import SegmentationMeter
from src.utils.model_outputs import split_model_outputs


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    use_amp: bool = True,
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
            outputs = model(images)
            logits, _, _ = split_model_outputs(outputs)
            loss, _ = criterion(logits, masks)

        losses.append(loss.item())
        meter.update(logits, masks)

    metrics: Dict[str, Any] = meter.compute()
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics
