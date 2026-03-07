from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.utils.model_outputs import split_model_outputs


def _to_size_list(orig_sizes) -> Sequence[Sequence[int]]:
    if torch.is_tensor(orig_sizes):
        return orig_sizes.cpu().tolist()
    if isinstance(orig_sizes, (list, tuple)) and len(orig_sizes) == 2:
        if torch.is_tensor(orig_sizes[0]) and torch.is_tensor(orig_sizes[1]):
            widths = orig_sizes[0].cpu().tolist()
            heights = orig_sizes[1].cpu().tolist()
            return list(zip(widths, heights))
    return orig_sizes


def _swap_logit_channels(
    logits: torch.Tensor, flip_pairs: Tuple[Tuple[int, int], ...]
) -> torch.Tensor:
    if not flip_pairs:
        return logits
    out = logits.clone()
    for left_id, right_id in flip_pairs:
        out[:, left_id] = logits[:, right_id]
        out[:, right_id] = logits[:, left_id]
    return out


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    output_ext: str = ".png",
    tta_flip: bool = False,
    flip_pairs: Sequence[Tuple[int, int]] | None = None,
    use_amp: bool = True,
) -> None:
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    output_ext = output_ext if output_ext.startswith(".") else f".{output_ext}"
    flip_pairs = tuple((int(a), int(b)) for a, b in (flip_pairs or []))

    for batch in tqdm(data_loader, desc="infer", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        names: Iterable[str] = batch["name"]
        orig_sizes = _to_size_list(batch["orig_size"])

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, _, _ = split_model_outputs(model(images))
            if tta_flip:
                flipped_logits, _, _ = split_model_outputs(model(torch.flip(images, dims=[3])))
                flipped_logits = torch.flip(flipped_logits, dims=[3])
                flipped_logits = _swap_logit_channels(flipped_logits, flip_pairs)
                logits = 0.5 * (logits + flipped_logits)
            pred = logits.argmax(dim=1)

        pred_np = pred.cpu().numpy().astype(np.uint8)
        for mask_arr, name, size in zip(pred_np, names, orig_sizes):
            width, height = int(size[0]), int(size[1])
            mask_img = Image.fromarray(mask_arr, mode="L")
            if mask_img.size != (width, height):
                mask_img = mask_img.resize((width, height), resample=Image.NEAREST)
            mask_img.save(output_dir / f"{name}{output_ext}")
