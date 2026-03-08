from __future__ import annotations

import random
from typing import Dict, Tuple
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter, RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.utils.flip_pairs import get_flip_pairs_from_cfg


def _swap_label_pairs(mask_tensor: torch.Tensor, pairs: Tuple[Tuple[int, int], ...]) -> torch.Tensor:
    if not pairs:
        return mask_tensor
    out = mask_tensor.clone()
    for left_id, right_id in pairs:
        left_mask = mask_tensor == left_id
        right_mask = mask_tensor == right_id
        out[left_mask] = right_id
        out[right_mask] = left_id
    return out


# Augmentationnnnnnn-----s transform here
class SegTrainTransform:
    """Synchronized train-time transforms for image and segmentation mask."""

    def __init__(self, cfg: Dict) -> None:
        size = int(cfg["data"]["input_size"])
        aug_cfg = cfg["augmentation"]
        self.flip_pairs = tuple(
            get_flip_pairs_from_cfg(cfg, num_classes=int(cfg["data"]["num_classes"]))
        )

        self.size = size
        self.scale = tuple(aug_cfg["resize_scale"])
        self.ratio = tuple(aug_cfg["resize_ratio"])
        self.hflip_prob = float(aug_cfg["hflip_prob"])
        self.rotation_deg = float(aug_cfg["rotation_deg"])

        color_cfg = aug_cfg["color_jitter"]
        self.color_jitter = ColorJitter(
            brightness=float(color_cfg["brightness"]),
            contrast=float(color_cfg["contrast"]),
            saturation=float(color_cfg["saturation"]),
            hue=float(color_cfg["hue"]),
        )
        self.color_jitter_prob = float(color_cfg["prob"])

        blur_cfg = aug_cfg["gaussian_blur"]
        self.blur_prob = float(blur_cfg["prob"])
        self.blur_sigma = tuple(float(v) for v in blur_cfg["sigma"])

        # Fixed normalization keeps behavior stable and reproducible.
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def __call__(self, image: Image.Image, mask: Image.Image):
        i, j, h, w = RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        image = TF.resized_crop(
            image, i, j, h, w, size=[self.size, self.size], interpolation=InterpolationMode.BILINEAR
        )
        mask = TF.resized_crop(
            mask, i, j, h, w, size=[self.size, self.size], interpolation=InterpolationMode.NEAREST
        )

        did_hflip = False
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            did_hflip = True

        if self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

        if random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)

        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        if did_hflip and self.flip_pairs:
            mask_tensor = _swap_label_pairs(mask_tensor, self.flip_pairs)
        return image_tensor, mask_tensor


class SegEvalTransform:
    """Deterministic transforms for validation/evaluation."""

    def __init__(self, cfg: Dict) -> None:
        self.size = int(cfg["data"]["input_size"])
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor


class InferenceTransform:
    def __init__(self, cfg: Dict) -> None:
        self.size = int(cfg["data"]["input_size"])
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)

    def __call__(self, image: Image.Image):
        image = TF.resize(image, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        return image_tensor
