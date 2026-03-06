from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def plot_validation_f1(history: List[Dict[str, Any]], output_path: Path) -> bool:
    """Plot validation F1 (F-measure) across epochs.

    Returns True if a plot was created, False if no usable validation F1 values were found.
    """
    epochs: List[int] = []
    f1_scores: List[float] = []

    for row in history:
        epoch = row.get("epoch")
        val = row.get("val", {})
        score = val.get("f1_macro") if isinstance(val, dict) else None
        if epoch is None or score is None:
            continue
        score = float(score)
        if not isfinite(score):
            continue
        epochs.append(int(epoch))
        f1_scores.append(score)

    if not f1_scores:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, f1_scores, marker="o", linewidth=2, color="#1f77b4")
    plt.title("Validation F-measure (F1 Macro) vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True
