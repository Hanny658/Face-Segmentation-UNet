from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.utils.tta import predict_with_tta


def _to_size_list(orig_sizes) -> Sequence[Sequence[int]]:
    if torch.is_tensor(orig_sizes):
        return orig_sizes.cpu().tolist()
    if isinstance(orig_sizes, (list, tuple)) and len(orig_sizes) == 2:
        if torch.is_tensor(orig_sizes[0]) and torch.is_tensor(orig_sizes[1]):
            widths = orig_sizes[0].cpu().tolist()
            heights = orig_sizes[1].cpu().tolist()
            return list(zip(widths, heights))
    return orig_sizes


@torch.no_grad() # not for inference runtime
def run_inference(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    output_ext: str = ".png",
    tta_enabled: bool = True,
    tta_flip: bool = False,
    tta_scales: Sequence[float] | None = None,
    flip_pairs: Sequence[Tuple[int, int]] | None = None,
    palette: Optional[Sequence[int]] = None,
    use_amp: bool = True,
) -> None:
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    output_ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    palette_list = [int(x) & 255 for x in palette] if palette is not None else None

    for batch in tqdm(data_loader, desc="infer", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        names: Iterable[str] = batch["name"]
        orig_sizes = _to_size_list(batch["orig_size"])

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = predict_with_tta(
                model=model,
                images=images,
                use_tta=bool(tta_enabled),
                tta_flip=bool(tta_flip),
                tta_scales=tta_scales,
                flip_pairs=flip_pairs,
            )
            pred = logits.argmax(dim=1)

        pred_np = pred.cpu().numpy().astype(np.uint8)
        for mask_arr, name, size in zip(pred_np, names, orig_sizes):
            width, height = int(size[0]), int(size[1])
            mask_img = Image.fromarray(mask_arr, mode="P")
            if palette_list is not None:
                mask_img.putpalette(palette_list)
            if mask_img.size != (width, height):
                mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
            mask_img.save(output_dir / f"{name}{output_ext}")

# FUN FUN FUN
