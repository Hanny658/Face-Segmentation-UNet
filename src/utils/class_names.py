from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_CELEBAMASK_19 = [
    "background",
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]


def get_class_names(cfg: Dict[str, Any], num_classes: int) -> List[str]:
    names = cfg.get("data", {}).get("class_names", None)
    if names is not None:
        names = [str(x) for x in names]
        if len(names) != num_classes:
            raise ValueError(
                f"data.class_names has {len(names)} entries, but num_classes={num_classes}."
            )
        return names

    if num_classes == 19:
        return DEFAULT_CELEBAMASK_19.copy()
    return [f"class_{i}" for i in range(num_classes)]
