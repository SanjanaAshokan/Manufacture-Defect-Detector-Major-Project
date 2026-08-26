"""
Training Script — Fine-tune EfficientNet-B0 / ResNet-50 for Defect Detection
=============================================================================
Implements the full training loop with:
  - AdamW optimizer with weight decay
  - OneCycleLR scheduler
  - Weighted cross-entropy loss for class imbalance
  - Mixed precision training (AMP)
  - Early stopping
  - Checkpoint saving (best + last)
  - TensorBoard logging

Usage:
    python src/train.py
    python src/train.py --architecture resnet50 --epochs 50
    python src/train.py --config configs/config.yaml
"""

import os
import sys
import time
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.models import get_model
from src.utils import (
    set_seed, get_device, save_checkpoint, EarlyStopping,
    AverageMeter, format_time, setup_logger
)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True,
    scheduler = None,
) -> dict:
    """Train for one epoch. Returns dict with loss and accuracy."""
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Step OneCycleLR scheduler after each batch
        if scheduler is not None:
            scheduler.step()

        # Track metrics
        loss_meter.update(loss.item(), images.size(0))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100 * correct / total:.1f}%")

    accuracy = correct / total if total > 0 else 0
    return {"loss": loss_meter.avg, "accuracy": accuracy}


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Validate the model. Returns dict with loss and accuracy."""
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Val  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss_meter.update(loss.item(), images.size(0))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100 * correct / total:.1f}%")

    accuracy = correct / total if total > 0 else 0
    return {"loss": loss_meter.avg, "accuracy": accuracy}


def train(config: dict):
    """Main training function."""

    # ---- Setup ----
    set_seed(config["training"]["seed"])
    device = get_device()

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["training"]["log_dir"])
    logger = setup_logger(str(log_dir))

    architecture = config["model"]["architecture"]
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    use_amp = config["training"]["use_amp"]
    warmup_epochs = config["model"].get("warmup_epochs", 3)

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Architecture:    {architecture}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Learning rate:   {lr}")
    print(f"  Weight decay:    {weight_decay}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Device:          {device}")
    print(f"  Warmup epochs:   {warmup_epochs}")
    print("=" * 60)

    # ---- Data ----
    use_weighted_sampler = config["training"].get("use_weighted_sampler", True)
    dataloaders = get_dataloaders(
        processed_dir=config["dataset"]["processed_dir"],
        batch_size=batch_size,
        image_size=config["dataset"]["image_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=config["training"]["pin_memory"],
        use_weighted_sampler=use_weighted_sampler,
    )

    # ---- Model ----
    model, gradcam_target_layer = get_model(
        architecture=architecture,
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        hidden_dim=config["model"]["hidden_dim"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"].get("freeze_backbone", True),
    )
    model = model.to(device)

    # ---- Loss ----
    # If using weighted sampler (default in get_dataloaders), we don't need class weights in loss
    # using both double-corrects class imbalance and degrades performance
    if use_weighted_sampler:
        criterion = nn.CrossEntropyLoss()
        print("\n[Loss] Standard CrossEntropy (data is balanced via sampler)")
    else:
        train_dataset = dataloaders["train"].dataset
        class_weights = train_dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"\n[Loss] Weighted CrossEntropy — weights: {class_weights.cpu().numpy()}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ---- Scheduler ----
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(dataloaders["train"]),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # ---- AMP Scaler ----
    scaler = GradScaler(enabled=use_amp)

    # ---- Early Stopping ----
    early_stopping = None
    if config["training"]["early_stopping"]["enabled"]:
        early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping"]["patience"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
            mode="max",
        )

    # ---- Training Loop ----
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    start_time = time.time()
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after warmup
        if epoch == warmup_epochs + 1 and config["model"].get("freeze_backbone", True):
            print(f"\n[Epoch {epoch}] Unfreezing backbone for full fine-tuning")
            model.unfreeze_backbone()
            # Recreate optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr * 0.1,  # Lower LR for fine-tuning
                weight_decay=weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr * 0.1,
                epochs=epochs - warmup_epochs,
                steps_per_epoch=len(dataloaders["train"]),
                pct_start=0.1,
                anneal_strategy="cos",
            )

        print(f"\nEpoch {epoch}/{epochs}")
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_metrics = train_one_epoch(
            model, dataloaders["train"], criterion, optimizer, device, scaler, use_amp, scheduler
        )

        # OneCycleLR is step-based and is now correctly stepped per batch inside train_one_epoch.

        # Validate
        val_metrics = validate(model, dataloaders["val"], criterion, device, use_amp)

        epoch_time = time.time() - epoch_start

        # Log
        print(
            f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val   Loss: {val_metrics['loss']:.4f} | Val   Acc: {val_metrics['accuracy']:.4f}"
        )
        print(f"  LR: {current_lr:.6f} | Time: {format_time(epoch_time)}")

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)

        # Save best model
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch,
                metrics={"val_acc": val_metrics["accuracy"], "val_loss": val_metrics["loss"]},
                path=str(checkpoint_dir / f"best_{architecture}.pth"),
                architecture=architecture,
            )
            print(f"  * New best model! Val Acc: {best_val_acc:.4f}")

        # Early stopping
        if early_stopping:
            if early_stopping(val_metrics["accuracy"]):
                print(f"\n[Early Stopping] No improvement for {early_stopping.patience} epochs. Stopping.")
                break

    # Save last checkpoint
    save_checkpoint(
        model, optimizer, epoch,
        metrics={"val_acc": val_metrics["accuracy"], "val_loss": val_metrics["loss"]},
        path=str(checkpoint_dir / f"last_{architecture}.pth"),
        architecture=architecture,
        extra={"history": history},
    )

    # ---- Summary ----
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time:       {format_time(total_time)}")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Best val accuracy: {best_val_acc:.4f} ({best_val_acc * 100:.1f}%)")
    print(f"  Best checkpoint:  {checkpoint_dir / f'best_{architecture}.pth'}")
    print(f"  Last checkpoint:  {checkpoint_dir / f'last_{architecture}.pth'}")
    print("=" * 60)

    return history


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train transistor defect detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--architecture", type=str, default=None, help="Override model architecture")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.architecture:
        config["model"]["architecture"] = args.architecture
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.no_amp:
        config["training"]["use_amp"] = False

    # Train
    history = train(config)
    print("\n[Done] Training finished successfully.")


if __name__ == "__main__":
    main()
