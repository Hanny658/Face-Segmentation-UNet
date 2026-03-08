import argparse
from typing import Any, Dict
import yaml
from src.models.lightweight_unet import build_model
from src.utils.param_count import count_parameters, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count model parameters.")
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# get and print Model Params
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model = build_model(cfg)

    total = count_parameters(model)
    trainable = count_trainable_parameters(model)
    limit = int(cfg["model"]["max_trainable_params"])

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Limit: {limit:,}")
    print(f"Within limit: {trainable < limit}")


if __name__ == "__main__":
    main()
