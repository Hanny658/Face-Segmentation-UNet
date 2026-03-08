from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from src.utils.model_outputs import split_model_outputs

# Test-Time Augmentations, not ensembleing models!!!

def _normalize_scales(scales: Optional[Sequence[float]]) -> Tuple[float, ...]:
    if not scales:
        return (1.0,)
    out = []
    for s in scales:
        v = float(s)
        if v <= 0:
            continue
        out.append(v)
    if not out:
        return (1.0,)
    return tuple(out)


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


def _forward_main_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    outputs = model(images)
    logits, _, _ = split_model_outputs(outputs)
    return logits


def _resize_images_for_scale(images: torch.Tensor, scale: float) -> torch.Tensor:
    h, w = images.shape[-2:]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if nh == h and nw == w:
        return images
    return F.interpolate(images, size=(nh, nw), mode="bilinear", align_corners=False)


def predict_with_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    use_tta: bool,
    tta_flip: bool = False,
    tta_scales: Optional[Sequence[float]] = None,
    flip_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> torch.Tensor:
    if not use_tta:
        return _forward_main_logits(model, images)

    scales = _normalize_scales(tta_scales)
    flip_pairs_t = tuple((int(a), int(b)) for a, b in (flip_pairs or []))
    target_size = images.shape[-2:]

    logits_acc = None
    count = 0
    for scale in scales:
        images_s = _resize_images_for_scale(images, scale)

        logits_s = _forward_main_logits(model, images_s)
        if logits_s.shape[-2:] != target_size:
            logits_s = F.interpolate(logits_s, size=target_size, mode="bilinear", align_corners=False)
        logits_acc = logits_s if logits_acc is None else (logits_acc + logits_s)
        count += 1

        if tta_flip:
            images_sf = torch.flip(images_s, dims=[3])
            logits_sf = _forward_main_logits(model, images_sf)
            logits_sf = torch.flip(logits_sf, dims=[3])
            logits_sf = _swap_logit_channels(logits_sf, flip_pairs_t)
            if logits_sf.shape[-2:] != target_size:
                logits_sf = F.interpolate(logits_sf, size=target_size, mode="bilinear", align_corners=False)
            logits_acc = logits_acc + logits_sf
            count += 1

    return logits_acc / max(1, count)
