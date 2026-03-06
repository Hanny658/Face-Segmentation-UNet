from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

Sample = Tuple[Path, Path]


def _list_files(directory: Path, allowed_suffixes: Sequence[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed_suffixes]
    )


def list_image_files(images_dir: Path) -> List[Path]:
    return _list_files(images_dir, IMAGE_EXTENSIONS)


def match_image_mask_pairs(images_dir: Path, masks_dir: Path, strict: bool = True) -> List[Sample]:
    image_files = list_image_files(images_dir)
    mask_files = _list_files(masks_dir, MASK_EXTENSIONS)
    mask_by_stem = {p.stem: p for p in mask_files}

    pairs: List[Sample] = []
    missing_masks: List[str] = []
    for image_path in image_files:
        mask_path = mask_by_stem.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue
        pairs.append((image_path, mask_path))

    if strict and missing_masks:
        first_five = ", ".join(missing_masks[:5])
        raise FileNotFoundError(f"Missing masks for {len(missing_masks)} images. Examples: {first_five}")
    if not pairs:
        raise RuntimeError(f"No image-mask pairs found under {images_dir} and {masks_dir}.")
    return pairs


def split_samples(
    samples: Sequence[Sample],
    val_split: float,
    seed: int,
    use_internal_val: bool,
) -> Tuple[List[Sample], List[Sample]]:
    samples = list(samples)
    if not use_internal_val or val_split <= 0:
        return samples, []

    num_samples = len(samples)
    num_val = max(1, int(num_samples * val_split))
    if num_val >= num_samples:
        num_val = num_samples - 1

    indices = list(range(num_samples))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[:num_val])

    train_samples = [samples[i] for i in range(num_samples) if i not in val_indices]
    val_samples = [samples[i] for i in range(num_samples) if i in val_indices]
    return train_samples, val_samples


def has_val_masks(data_root: Path, cfg: Dict) -> bool:
    val_masks = data_root / cfg["data"]["val_masks"]
    if not val_masks.exists():
        return False
    return any(p.suffix.lower() in MASK_EXTENSIONS for p in val_masks.iterdir() if p.is_file())


class SegmentationDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: Optional[Callable] = None) -> None:
        self.samples: List[Sample] = list(samples)
        self.transform = transform
        if not self.samples:
            raise ValueError("SegmentationDataset received an empty sample list.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        # Keep indexed labels intact for palette masks; avoid forced grayscale remapping.
        mask = Image.open(mask_path)
        if mask.mode not in {"L", "P", "I"}:
            mask = mask.convert("L")

        if self.transform is None:
            raise ValueError("SegmentationDataset requires a transform that returns tensor image/mask.")
        image, mask = self.transform(image, mask)

        return {
            "image": image,
            "mask": mask,
            "name": image_path.stem,
        }


class InferenceDataset(Dataset):
    def __init__(self, images_dir: Path, transform: Callable) -> None:
        self.images = list_image_files(images_dir)
        self.transform = transform
        if not self.images:
            raise RuntimeError(f"No images found in {images_dir}.")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        image_tensor = self.transform(image)
        return {
            "image": image_tensor,
            "name": image_path.stem,
            "orig_size": torch.tensor([original_size[0], original_size[1]], dtype=torch.int64),
        }
