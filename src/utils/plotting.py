from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def plot_training_curve(history: List[Dict[str, Any]], output_path: Path) -> bool:
    """Plot train loss and optional validation F-score on one figure.

    Returns True if at least one usable series was found.
    """
    train_epochs: List[int] = []
    train_losses: List[float] = []
    val_epochs: List[int] = []
    val_f_scores: List[float] = []

    for row in history:
        epoch = row.get("epoch")
        if epoch is None:
            continue

        train = row.get("train", {})
        loss = train.get("loss") if isinstance(train, dict) else None
        if loss is not None:
            loss = float(loss)
            if isfinite(loss):
                train_epochs.append(int(epoch))
                train_losses.append(loss)

        val = row.get("val", {})
        score = val.get("f_score") if isinstance(val, dict) else None
        if score is None and isinstance(val, dict):
            score = val.get("f1_macro")
        if score is not None:
            score = float(score)
            if isfinite(score):
                val_epochs.append(int(epoch))
                val_f_scores.append(score)

    has_train = len(train_losses) > 0
    has_val = len(val_f_scores) > 0
    if not (has_train or has_val):
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_xlabel("Epoch")
    ax1.grid(True, linestyle="--", alpha=0.4)

    if has_train:
        ax1.plot(train_epochs, train_losses, marker="o", linewidth=2, color="#d62728", label="Train Loss")
        ax1.set_ylabel("Train Loss", color="#d62728")
        ax1.tick_params(axis="y", labelcolor="#d62728")

    if has_val:
        ax2 = ax1.twinx()
        ax2.plot(val_epochs, val_f_scores, marker="s", linewidth=2, color="#1f77b4", label="Val F-score")
        ax2.set_ylabel("Val F-score", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.set_ylim(0.0, 1.0)

    if has_train and has_val:
        ax1.set_title("Train Loss and Validation F-score vs Epoch")
    elif has_train:
        ax1.set_title("Train Loss vs Epoch")
    else:
        ax1.set_title("Validation F-score vs Epoch")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_validation_f_score(history: List[Dict[str, Any]], output_path: Path) -> bool:
    """Backward-compatible alias."""
    return plot_training_curve(history, output_path)


def plot_validation_f1(history: List[Dict[str, Any]], output_path: Path) -> bool:
    """Backward-compatible alias."""
    return plot_training_curve(history, output_path)
