from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def _resolve_dotted_key(data: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Key '{dotted_key}' not found in stats JSON (stopped at '{part}').")
        cur = cur[part]
    return cur


def maybe_load_ce_class_weights(cfg: Dict[str, Any], num_classes: int) -> Optional[torch.Tensor]:
    loss_cfg = cfg.get("loss", {})
    ce_cfg = loss_cfg.get("ce_weighting", {})
    if not bool(ce_cfg.get("enabled", False)):
        return None

    stats_json = Path(str(ce_cfg.get("stats_json", "experiments/mask_stats.json")))
    if not stats_json.exists():
        raise FileNotFoundError(
            f"Weighted CE is enabled but stats JSON was not found: {stats_json}. "
            "Run analyze_masks.py first."
        )

    with open(stats_json, "r", encoding="utf-8") as f:
        stats = json.load(f)

    key = str(ce_cfg.get("key", "recommended_weighted_ce.weights"))
    raw = _resolve_dotted_key(stats, key)
    weights = np.asarray(raw, dtype=np.float32)
    if weights.ndim != 1:
        raise ValueError(f"Class weights at '{key}' must be a 1D list, got shape {weights.shape}.")
    if len(weights) != int(num_classes):
        raise ValueError(
            f"Class weight length mismatch: got {len(weights)} from '{key}', expected {num_classes}."
        )
    if np.any(weights < 0):
        raise ValueError("Class weights must be non-negative.")

    min_weight = float(ce_cfg.get("min_weight", 0.0))
    if min_weight > 0:
        weights = np.maximum(weights, min_weight)

    max_weight = ce_cfg.get("max_weight", None)
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))

    if bool(ce_cfg.get("normalize_mean_one", False)):
        mean = float(weights.mean())
        if mean > 0:
            weights = weights / mean

    return torch.tensor(weights, dtype=torch.float32)
