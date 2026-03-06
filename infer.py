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
    run_inference(
        model=model,
        data_loader=loader,
        device=device,
        output_dir=output_dir,
        output_ext=str(cfg["inference"]["output_ext"]),
        tta_flip=bool(cfg["inference"]["tta_flip"]),
        use_amp=bool(cfg["train"]["use_amp"]),
    )
    print(f"Saved predictions to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
