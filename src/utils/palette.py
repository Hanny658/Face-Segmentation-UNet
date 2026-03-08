from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence
from PIL import Image
from src.datasets.celebamask_dataset import MASK_EXTENSIONS


def normalize_palette(palette: Sequence[int]) -> List[int]:
    pal = [int(x) & 255 for x in palette]
    if len(pal) < 768:
        pal = pal + [0] * (768 - len(pal))
    elif len(pal) > 768:
        pal = pal[:768]
    return pal


def make_pascal_palette(num_colors: int = 256) -> List[int]:
    """Generate a standard PASCAL VOC-style palette (length = 768)."""
    palette = [0] * (num_colors * 3)
    for j in range(num_colors):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= (((lab >> 0) & 1) << (7 - i))
            g |= (((lab >> 1) & 1) << (7 - i))
            b |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        palette[3 * j + 0] = r
        palette[3 * j + 1] = g
        palette[3 * j + 2] = b
    return normalize_palette(palette)


def load_palette_from_masks_dir(masks_dir: Path) -> Optional[List[int]]:
    if not masks_dir.exists():
        return None
    mask_files = sorted(
        p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS
    )
    for path in mask_files:
        try:
            with Image.open(path) as m:
                pal = m.getpalette()
                if pal:
                    return normalize_palette(pal)
        except OSError:
            continue
    return None
