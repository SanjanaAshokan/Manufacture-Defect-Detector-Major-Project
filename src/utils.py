"""
Utility Functions — Checkpoint Management, Logging, Device Detection
====================================================================
Common helpers used across training, evaluation, and inference.
"""

import os
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    
    Args:
        patience: How many epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss, 'max' for accuracy.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Utils] Random seed set to {seed}")


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"[Utils] Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Utils] Using Apple MPS backend")
    else:
        device = torch.device("cpu")
        print("[Utils] Using CPU (training will be slower)")
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: str,
    architecture: str = "",
    extra: Optional[Dict[str, Any]] = None,
):
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save.
        optimizer: The optimizer state.
        epoch: Current epoch number.
        metrics: Dictionary of metric values.
        path: Path to save the checkpoint.
        architecture: Model architecture name.
        extra: Any extra information to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "architecture": architecture,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        checkpoint.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device = torch.device("cpu"),
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        model: Model to load weights into.
        path: Path to the checkpoint file.
        device: Device to load onto.
        optimizer: Optional optimizer to restore state.
    
    Returns:
        Dictionary with checkpoint metadata.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    architecture = checkpoint.get("architecture", "unknown")

    print(f"[Checkpoint] Loaded from {path}")
    print(f"  Architecture: {architecture}")
    print(f"  Epoch: {epoch}")
    if metrics:
        for key, val in metrics.items():
            print(f"  {key}: {val:.4f}")

    return checkpoint


def setup_logger(log_dir: str, name: str = "defect_detector") -> logging.Logger:
    """
    Set up a logger that writes to both console and file.
    
    Args:
        log_dir: Directory to save log files.
        name: Logger name.
    
    Returns:
        Configured logger.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if logger.handlers:
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


class AverageMeter:
    """Computes and stores the running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}hr"
