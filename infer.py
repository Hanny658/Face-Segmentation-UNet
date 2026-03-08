import argparse
from pathlib import Path
from typing import Any, Dict
import torch
import yaml
from torch.utils.data import DataLoader
from src.datasets.celebamask_dataset import InferenceDataset
from src.datasets.transforms import InferenceTransform
from src.engine.inference import run_inference
from src.models.lightweight_unet import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.flip_pairs import get_flip_pairs_from_cfg
from src.utils.palette import load_palette_from_masks_dir, make_pascal_palette


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference on val images for face parsing.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--tta-flip", action="store_true", help="Enable horizontal flip TTA.")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.batch_size is not None:
        cfg["inference"]["batch_size"] = args.batch_size
    if args.output_dir is not None:
        cfg["inference"]["output_dir"] = args.output_dir
    if args.tta_flip:
        cfg["inference"]["tta_flip"] = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location=device)

    data_root = Path(cfg["data"]["root"])
    image_dir = data_root / cfg["data"]["val_images"]
    dataset = InferenceDataset(images_dir=image_dir, transform=InferenceTransform(cfg))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["inference"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=bool(cfg["data"]["pin_memory"]),
        drop_last=False,
    )

    output_dir = Path(cfg["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    flip_pairs = get_flip_pairs_from_cfg(cfg, num_classes=int(cfg["data"]["num_classes"]))
    tta_enabled = bool(cfg.get("inference", {}).get("tta_enabled", True))
    tta_flip = bool(cfg["inference"]["tta_flip"])
    tta_scales = cfg.get("inference", {}).get("tta_scales", [1.0])
    palette = load_palette_from_masks_dir(data_root / cfg["data"]["train_masks"])
    if palette is None:
        palette = make_pascal_palette()
    run_inference(
        model=model,
        data_loader=loader,
        device=device,
        output_dir=output_dir,
        output_ext=str(cfg["inference"]["output_ext"]),
        tta_enabled=tta_enabled,
        tta_flip=tta_flip,
        tta_scales=tta_scales,
        flip_pairs=flip_pairs,
        palette=palette,
        use_amp=bool(cfg["train"]["use_amp"]),
    )
    print(f"TTA: enabled={tta_enabled}, flip={tta_flip}, scales={tta_scales}")
    print(f"Saved predictions to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
