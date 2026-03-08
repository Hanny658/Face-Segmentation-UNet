import argparse
import json
from pathlib import Path
from typing import Any, Dict
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from src.datasets.celebamask_dataset import InferenceDataset
from src.datasets.celebamask_dataset import SegmentationDataset, match_image_mask_pairs, split_samples
from src.datasets.transforms import InferenceTransform, SegEvalTransform, SegTrainTransform
from src.engine.inference import run_inference
from src.engine.trainer import fit
from src.losses.segmentation_loss import SegmentationLoss
from src.models.lightweight_unet import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.class_weights import maybe_load_ce_class_weights
from src.utils.flip_pairs import get_flip_pairs_from_cfg
from src.utils.param_count import count_trainable_parameters
from src.utils.palette import load_palette_from_masks_dir, make_pascal_palette
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight UNet for face parsing.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--save-dir", type=str, default=None, help="Override train.save_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers.")
    parser.add_argument("--full-train", action="store_true", help="Disable internal val split.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.save_dir is not None:
        cfg["train"]["save_dir"] = args.save_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.full_train:
        cfg["train"]["use_internal_val"] = False
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(cfg["data"]["root"])
    train_images = data_root / cfg["data"]["train_images"]
    train_masks = data_root / cfg["data"]["train_masks"]

    samples = match_image_mask_pairs(train_images, train_masks, strict=True)
    train_samples, val_samples = split_samples(
        samples=samples,
        val_split=float(cfg["data"]["val_split"]),
        seed=int(cfg["seed"]),
        use_internal_val=bool(cfg["train"]["use_internal_val"]),
    )

    train_transform = SegTrainTransform(cfg)
    eval_transform = SegEvalTransform(cfg)
    train_dataset = SegmentationDataset(train_samples, transform=train_transform)
    val_dataset = SegmentationDataset(val_samples, transform=eval_transform) if val_samples else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=bool(cfg["train"]["drop_last"]),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=bool(cfg["data"]["pin_memory"]),
            drop_last=False,
        )

    model = build_model(cfg).to(device)
    trainable_params = count_trainable_parameters(model)
    max_params = int(cfg["model"]["max_trainable_params"])
    if trainable_params >= max_params:
        raise ValueError(
            f"Model has {trainable_params:,} trainable params, which violates limit {max_params:,}."
        )

    num_classes = int(cfg["data"]["num_classes"])
    ce_class_weights = maybe_load_ce_class_weights(cfg, num_classes=num_classes)
    criterion = SegmentationLoss(
        num_classes=num_classes,
        dice_weight=float(cfg["loss"]["dice_weight"]),
        ignore_index=cfg["loss"]["ignore_index"],
        class_weights=ce_class_weights,
        dice_present_only=bool(cfg["loss"].get("dice_present_only", True)),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"]["epochs"]),
        eta_min=float(cfg["train"]["min_lr"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["train"]["use_amp"]) and device.type == "cuda")

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {0 if val_dataset is None else len(val_dataset)}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Saving artifacts to: {save_dir}")
    if ce_class_weights is not None:
        w_min = float(ce_class_weights.min().item())
        w_max = float(ce_class_weights.max().item())
        print(f"Weighted CE enabled: min_weight={w_min:.6f}, max_weight={w_max:.6f}")
    boundary_cfg = cfg.get("loss", {}).get("boundary", {})
    if bool(boundary_cfg.get("enabled", False)):
        print(
            "Boundary loss enabled: "
            f"weight={float(boundary_cfg.get('weight', 0.05))}, "
            f"pos_weight={boundary_cfg.get('pos_weight', 4.0)}, "
            f"warmup_epochs={int(boundary_cfg.get('warmup_epochs', 0))}, "
            f"pred_scale={float(boundary_cfg.get('pred_scale', 4.0))}"
        )

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        cfg=cfg,
        save_dir=save_dir,
    )

    if bool(cfg["train"].get("run_val_data", False)):
        best_ckpt = save_dir / "best.pt"
        if not best_ckpt.exists():
            raise FileNotFoundError(
                f"run_val_data is enabled but best checkpoint was not found: {best_ckpt}"
            )

        print("[post-train] run_val_data=true: running inference on data/val/images ...")
        load_checkpoint(str(best_ckpt), model=model, map_location=device)

        val_image_dir = data_root / cfg["data"]["val_images"]
        val_mask_dir = data_root / cfg["data"]["val_masks"]
        val_mask_dir.mkdir(parents=True, exist_ok=True)
        palette = load_palette_from_masks_dir(train_masks)
        if palette is None:
            palette = make_pascal_palette()

        infer_dataset = InferenceDataset(images_dir=val_image_dir, transform=InferenceTransform(cfg))
        infer_loader = DataLoader(
            infer_dataset,
            batch_size=int(cfg["inference"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"]["num_workers"]),
            pin_memory=bool(cfg["data"]["pin_memory"]),
            drop_last=False,
        )
        flip_pairs = get_flip_pairs_from_cfg(cfg, num_classes=int(cfg["data"]["num_classes"]))
        run_inference(
            model=model,
            data_loader=infer_loader,
            device=device,
            output_dir=val_mask_dir,
            output_ext=str(cfg["inference"]["output_ext"]),
            tta_enabled=bool(cfg.get("inference", {}).get("tta_enabled", True)),
            tta_flip=bool(cfg["inference"]["tta_flip"]),
            tta_scales=cfg.get("inference", {}).get("tta_scales", [1.0]),
            flip_pairs=flip_pairs,
            palette=palette,
            use_amp=bool(cfg["train"]["use_amp"]),
        )
        print(f"[post-train] Saved val predictions to: {val_mask_dir.resolve()}")


if __name__ == "__main__":
    main()
