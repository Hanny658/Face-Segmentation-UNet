from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from src.utils.class_names import get_class_names


def _normalize_pair(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def infer_lr_flip_pairs(class_names: Sequence[str]) -> List[Tuple[int, int]]:
    """Infer left/right class-id pairs from names like l_eye/r_eye."""
    name_to_id = {str(name): idx for idx, name in enumerate(class_names)}
    pairs = set()
    for name, idx in name_to_id.items():
        if name.startswith("l_"):
            right_name = f"r_{name[2:]}"
            if right_name in name_to_id:
                pairs.add(_normalize_pair(idx, name_to_id[right_name]))
    return sorted(pairs)


def get_flip_pairs_from_cfg(cfg: Dict[str, Any], num_classes: int) -> List[Tuple[int, int]]:
    data_cfg = cfg.get("data", {})
    raw_pairs = data_cfg.get("flip_pairs")
    if raw_pairs is not None:
        pairs: List[Tuple[int, int]] = []
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each data.flip_pairs item must be [left_id, right_id], got: {pair}")
            a, b = int(pair[0]), int(pair[1])
            if a < 0 or b < 0 or a >= num_classes or b >= num_classes or a == b:
                raise ValueError(f"Invalid flip pair {pair} for num_classes={num_classes}")
            pairs.append(_normalize_pair(a, b))
        return sorted(set(pairs))

    class_names = get_class_names(cfg, num_classes=num_classes)
    return infer_lr_flip_pairs(class_names)
