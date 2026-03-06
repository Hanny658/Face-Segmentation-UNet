import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets.celebamask_dataset import SegmentationDataset, has_val_masks, match_image_mask_pairs, split_samples
from src.datasets.transforms import SegEvalTransform
from src.engine.evaluator import evaluate
from src.losses.segmentation_loss import SegmentationLoss
from src.models.lightweight_unet import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--source",
        type=str,
        choices=["internal", "val"],
        default="internal",
        help="internal: split from train set; val: use data/val/images + data/val/masks.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = SegEvalTransform(cfg)
    data_root = Path(cfg["data"]["root"])

    if args.source == "internal":
        train_images = data_root / cfg["data"]["train_images"]
        train_masks = data_root / cfg["data"]["train_masks"]
        samples = match_image_mask_pairs(train_images, train_masks, strict=True)
        _, eval_samples = split_samples(
            samples=samples,
            val_split=float(cfg["data"]["val_split"]),
            seed=int(cfg["seed"]),
            use_internal_val=True,
        )
        if not eval_samples:
            raise ValueError("Internal validation split is empty. Increase data.val_split.")
    else:
        if not has_val_masks(data_root, cfg):
            raise FileNotFoundError(
                f"Missing labeled val masks at: {data_root / cfg['data']['val_masks']}. "
                "Use --source internal or add val masks."
            )
        eval_samples = match_image_mask_pairs(
            images_dir=data_root / cfg["data"]["val_images"],
            masks_dir=data_root / cfg["data"]["val_masks"],
            strict=True,
        )

    dataset = SegmentationDataset(eval_samples, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)
    criterion = SegmentationLoss(
        num_classes=int(cfg["data"]["num_classes"]),
        dice_weight=float(cfg["loss"]["dice_weight"]),
        ignore_index=cfg["loss"]["ignore_index"],
    )

    metrics = evaluate(
        model=model,
        data_loader=loader,
        criterion=criterion,
        device=device,
        num_classes=int(cfg["data"]["num_classes"]),
        use_amp=bool(cfg["train"]["use_amp"]),
        desc=f"validate:{args.source}",
    )
    print(f"Samples: {len(dataset)}")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.6f}")
    print(f"F1 (macro): {metrics['f1_macro']:.6f}")


if __name__ == "__main__":
    main()
