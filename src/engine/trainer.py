from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.engine.evaluator import evaluate
from src.utils.checkpoint import save_checkpoint
from src.utils.model_outputs import split_model_outputs
from src.utils.plotting import plot_validation_f1


def _mask_to_boundary_target(mask: torch.Tensor) -> torch.Tensor:
    """Convert class mask (N,H,W) to binary boundary target (N,1,H,W)."""
    edge = torch.zeros_like(mask, dtype=torch.bool)
    edge[:, 1:, :] |= mask[:, 1:, :] != mask[:, :-1, :]
    edge[:, :-1, :] |= mask[:, :-1, :] != mask[:, 1:, :]
    edge[:, :, 1:] |= mask[:, :, 1:] != mask[:, :, :-1]
    edge[:, :, :-1] |= mask[:, :, :-1] != mask[:, :, 1:]
    return edge.unsqueeze(1).float()


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    cfg: Dict,
    use_amp: bool = True,
    epoch: int = 0,
    epochs: int = 0,
) -> Dict[str, float]:
    model.train()
    amp_enabled = use_amp and device.type == "cuda"

    aux_cfg = cfg.get("loss", {}).get("pid_aux", {})
    aux_enabled = bool(aux_cfg.get("enabled", False))
    aux_p_weight = float(aux_cfg.get("p_weight", 0.4))
    aux_d_weight = float(aux_cfg.get("d_weight", 0.1))
    pos_w = aux_cfg.get("d_pos_weight", None)
    if pos_w is not None:
        pos_weight = torch.tensor(float(pos_w), device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        bce = nn.BCEWithLogitsLoss()

    total_losses = []
    ce_losses = []
    dice_losses = []
    aux_p_losses = []
    aux_d_losses = []

    progress = tqdm(data_loader, desc=f"train {epoch + 1}/{epochs}", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            main_logits, aux_p_logits, aux_d_logits = split_model_outputs(outputs)
            loss, comp = criterion(main_logits, masks)

            aux_p_loss = torch.zeros((), device=device)
            aux_d_loss = torch.zeros((), device=device)
            if aux_enabled and aux_p_logits is not None:
                aux_p_loss, _ = criterion(aux_p_logits, masks)
                loss = loss + aux_p_weight * aux_p_loss
            if aux_enabled and aux_d_logits is not None:
                boundary_target = _mask_to_boundary_target(masks)
                aux_d_loss = bce(aux_d_logits, boundary_target)
                loss = loss + aux_d_weight * aux_d_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_losses.append(float(loss.detach().item()))
        ce_losses.append(float(comp["ce"].item()))
        dice_losses.append(float(comp["dice"].item()))
        aux_p_losses.append(float(aux_p_loss.detach().item()))
        aux_d_losses.append(float(aux_d_loss.detach().item()))
        progress.set_postfix(loss=f"{np.mean(total_losses):.4f}")

    return {
        "loss": float(np.mean(total_losses)),
        "ce": float(np.mean(ce_losses)),
        "dice": float(np.mean(dice_losses)),
        "aux_p": float(np.mean(aux_p_losses)),
        "aux_d": float(np.mean(aux_d_losses)),
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
            cfg=cfg,
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

    if val_loader is not None:
        plot_path = save_dir / "val_f1_curve.png"
        created = plot_validation_f1(history, plot_path)
        if created:
            print(f"Saved validation F1 curve to: {plot_path}")
