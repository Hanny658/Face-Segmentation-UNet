from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.engine.evaluator import evaluate
from src.utils.checkpoint import save_checkpoint


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
    epoch: int = 0,
    epochs: int = 0,
) -> Dict[str, float]:
    model.train()
    amp_enabled = use_amp and device.type == "cuda"

    total_losses = []
    ce_losses = []
    dice_losses = []

    progress = tqdm(data_loader, desc=f"train {epoch + 1}/{epochs}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss, comp = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_losses.append(float(comp["total"].item()))
        ce_losses.append(float(comp["ce"].item()))
        dice_losses.append(float(comp["dice"].item()))
        progress.set_postfix(loss=f"{np.mean(total_losses):.4f}")

    return {
        "loss": float(np.mean(total_losses)),
        "ce": float(np.mean(ce_losses)),
        "dice": float(np.mean(dice_losses)),
    }


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: Dict,
    save_dir: Path,
) -> None:
    num_classes = int(cfg["data"]["num_classes"])
    epochs = int(cfg["train"]["epochs"])
    use_amp = bool(cfg["train"]["use_amp"])

    best_score = -float("inf")
    history = []

    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            epochs=epochs,
        )
        scheduler.step()

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                use_amp=use_amp,
                desc=f"val {epoch + 1}/{epochs}",
            )
            current_score = val_metrics["f1_macro"]
        else:
            val_metrics = {}
            # If no validation set is used, pick latest by minimizing train loss.
            current_score = -train_metrics["loss"]

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": cfg,
        }
        save_checkpoint(checkpoint, save_dir / "last.pt")

        is_best = current_score > best_score
        if is_best:
            best_score = current_score
            save_checkpoint(checkpoint, save_dir / "best.pt")

        row = {
            "epoch": epoch + 1,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
            "best_score": best_score,
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_f1={val_metrics.get('f1_macro', float('nan')):.4f} "
            f"best={'yes' if is_best else 'no'}"
        )

    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
