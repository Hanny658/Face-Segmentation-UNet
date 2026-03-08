from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# Oringinal is JPG, idk what will mask be required for
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze segmentation mask labels and class proportions for weighted CE."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Optional config path.")
    parser.add_argument("--masks-dir", type=str, default="data/train/masks")
    parser.add_argument("--num-classes", type=int, default=None, help="Override number of classes.")
    parser.add_argument("--output-json", type=str, default="experiments/mask_stats.json")
    parser.add_argument("--output-csv", type=str, default="experiments/mask_stats.csv")
    parser.add_argument("--eps", type=float, default=1e-7)
    return parser.parse_args()


def maybe_load_num_classes(config_path: Path) -> Optional[int]:
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        return int(cfg["data"]["num_classes"])
    except Exception:
        return None


def list_masks(mask_dir: Path) -> List[Path]:
    if not mask_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {mask_dir}")
    files = [p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS]
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No mask files found in: {mask_dir}")
    return files


def read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 3:
        # Fallback for color masks: use first channel.
        return arr[..., 0].astype(np.int64, copy=False)
    raise ValueError(f"Unsupported mask shape {arr.shape} at {path}")


def analyze_masks(mask_paths: List[Path]) -> Tuple[Dict[int, int], Dict[int, int], int]:
    pixel_counts: Dict[int, int] = {}
    image_counts: Dict[int, int] = {}
    total_pixels = 0

    for p in tqdm(mask_paths, desc="scan-masks"):
        mask = read_mask(p)
        total_pixels += int(mask.size)
        labels, counts = np.unique(mask, return_counts=True)

        for label, count in zip(labels.tolist(), counts.tolist()):
            label_int = int(label)
            count_int = int(count)
            pixel_counts[label_int] = pixel_counts.get(label_int, 0) + count_int
            image_counts[label_int] = image_counts.get(label_int, 0) + 1

    return pixel_counts, image_counts, total_pixels


def compute_weights(pixel_counts: np.ndarray, eps: float) -> Dict[str, List[float]]:
    total = float(pixel_counts.sum())
    freq = pixel_counts / (total + eps)
    nonzero = pixel_counts > 0

    inverse = np.zeros_like(freq, dtype=np.float64)
    inverse[nonzero] = 1.0 / (freq[nonzero] + eps)

    median_freq = np.zeros_like(freq, dtype=np.float64)
    if np.any(nonzero):
        median = float(np.median(freq[nonzero]))
        median_freq[nonzero] = median / (freq[nonzero] + eps)

    def normalize_mean_one(weights: np.ndarray) -> np.ndarray:
        out = weights.copy()
        valid = out > 0
        if np.any(valid):
            out[valid] = out[valid] / np.mean(out[valid])
        return out

    inverse_norm = normalize_mean_one(inverse)
    median_norm = normalize_mean_one(median_freq)

    return {
        "inverse_freq_raw": inverse.tolist(),
        "inverse_freq_norm_mean1": inverse_norm.tolist(),
        "median_freq_raw": median_freq.tolist(),
        "median_freq_norm_mean1": median_norm.tolist(),
    }


def main() -> None:
    args = parse_args()
    mask_dir = Path(args.masks_dir)
    mask_paths = list_masks(mask_dir)

    config_num_classes = maybe_load_num_classes(Path(args.config))
    if args.num_classes is not None:
        num_classes = int(args.num_classes)
    elif config_num_classes is not None:
        num_classes = int(config_num_classes)
    else:
        num_classes = None

    pixel_counts_dict, image_counts_dict, total_pixels = analyze_masks(mask_paths)
    unique_labels = sorted(pixel_counts_dict.keys())
    max_label = max(unique_labels)

    if num_classes is None:
        num_classes = max_label + 1

    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    image_counts = np.zeros(num_classes, dtype=np.int64)
    out_of_range: Dict[int, int] = {}

    for label, count in pixel_counts_dict.items():
        if 0 <= label < num_classes:
            pixel_counts[label] = count
        else:
            out_of_range[label] = count

    for label, count in image_counts_dict.items():
        if 0 <= label < num_classes:
            image_counts[label] = count

    num_masks = len(mask_paths)
    pixel_ratio = pixel_counts / max(float(total_pixels), 1.0)
    image_ratio = image_counts / max(float(num_masks), 1.0)
    weights = compute_weights(pixel_counts, eps=float(args.eps))

    per_class = []
    for class_id in range(num_classes):
        per_class.append(
            {
                "class_id": class_id,
                "pixel_count": int(pixel_counts[class_id]),
                "pixel_ratio": float(pixel_ratio[class_id]),
                "image_count": int(image_counts[class_id]),
                "image_ratio": float(image_ratio[class_id]),
            }
        )

    stats = {
        "masks_dir": str(mask_dir.resolve()),
        "num_masks": num_masks,
        "total_pixels": int(total_pixels),
        "unique_labels_found": unique_labels,
        "max_label_found": int(max_label),
        "num_classes_used": int(num_classes),
        "config_num_classes": config_num_classes,
        "out_of_range_labels": {str(k): int(v) for k, v in out_of_range.items()},
        "per_class": per_class,
        "weights": weights,
        "recommended_weighted_ce": {
            "name": "median_freq_norm_mean1",
            "weights": weights["median_freq_norm_mean1"],
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "pixel_count", "pixel_ratio", "image_count", "image_ratio"],
        )
        writer.writeheader()
        writer.writerows(per_class)

    print(f"Scanned masks: {num_masks}")
    print(f"Total pixels: {total_pixels}")
    print(f"Unique labels found: {unique_labels}")
    print(f"num_classes used: {num_classes} (config: {config_num_classes})")
    if out_of_range:
        print(f"Out-of-range labels (w.r.t num_classes): {sorted(out_of_range.keys())}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved CSV: {output_csv}")
    print("Recommended CE weights key: weights.median_freq_norm_mean1")


if __name__ == "__main__":
    main()
