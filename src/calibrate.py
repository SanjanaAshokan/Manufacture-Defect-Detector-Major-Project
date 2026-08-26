"""
Confidence Calibration — Temperature Scaling
=============================================
A trained classifier can be accurate on average while still being
overconfident (e.g. it says "97% defective" on images it's only right
about 80% of the time). Accuracy/precision/recall don't catch this —
only calibration metrics like Expected Calibration Error (ECE) do.

This script fits a single scalar "temperature" on the validation set
(Guo et al., 2017) that rescales the model's logits so the reported
confidence actually matches its real-world hit rate. It does not
change any prediction (argmax is unaffected), only the confidence
percentage shown to the user — so it's safe to apply to an
already-trained checkpoint without retraining.

Usage:
    python src/calibrate.py --checkpoint outputs/checkpoints/best_efficientnet_b0.pth
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.models import get_model
from src.utils import load_checkpoint, get_device, set_seed
from src.evaluate import compute_calibration_error


@torch.no_grad()
def collect_logits(model: nn.Module, dataloader, device: torch.device):
    """Run the model and collect raw logits + labels (pre-softmax)."""
    model.eval()
    all_logits, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """
    Fit a single scalar T minimizing NLL of softmax(logits / T) via LBFGS.
    T > 1 softens (de-confidences) predictions; T < 1 sharpens them.
    """
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=0.05).item())


def main():
    parser = argparse.ArgumentParser(description="Fit temperature scaling for calibrated confidence")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--architecture", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    architecture = args.architecture or config["model"]["architecture"]
    set_seed(config["training"]["seed"])
    device = get_device()

    model, _ = get_model(
        architecture=architecture,
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        hidden_dim=config["model"]["hidden_dim"],
        pretrained=False,
        freeze_backbone=False,
    )
    checkpoint_path = args.checkpoint or str(
        Path(config["training"]["checkpoint_dir"]) / f"best_{architecture}.pth"
    )
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    # Fit on the VALIDATION split, never on test — fitting on test would leak
    # test-set information into a number the dashboard later reports.
    dataloaders = get_dataloaders(
        processed_dir=config["dataset"]["processed_dir"],
        batch_size=config["training"]["batch_size"],
        image_size=config["dataset"]["image_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=False,
        use_weighted_sampler=False,
    )

    print("\n[Calibrate] Collecting validation logits...")
    logits, labels = collect_logits(model, dataloaders["val"], device)

    probs_before = torch.softmax(logits, dim=1).numpy()
    ece_before = compute_calibration_error(labels.numpy(), probs_before)["ece"]

    print("[Calibrate] Fitting temperature via LBFGS...")
    temperature = fit_temperature(logits, labels)

    probs_after = torch.softmax(logits / temperature, dim=1).numpy()
    ece_after = compute_calibration_error(labels.numpy(), probs_after)["ece"]

    print(f"\n  Temperature:        {temperature:.4f}")
    print(f"  ECE before scaling: {ece_before:.4f}")
    print(f"  ECE after scaling:  {ece_after:.4f}")
    if ece_after < ece_before:
        print("  → Calibration improved. Confidence values will better match real accuracy.")
    else:
        print("  → No improvement found; the model's confidence was already well-calibrated,")
        print("    or the validation set is too small for a reliable fit.")

    # Save alongside the checkpoint so the web app / evaluate.py can pick it up.
    calib_path = Path(config["evaluation"]["plots_dir"]) / f"calibration_{architecture}.json"
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    with open(calib_path, "w") as f:
        json.dump({
            "architecture": architecture,
            "checkpoint": checkpoint_path,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "temperature": temperature,
            "ece_before": ece_before,
            "ece_after": ece_after,
        }, f, indent=2)
    print(f"\n[Calibrate] Saved to {calib_path}")


if __name__ == "__main__":
    main()
