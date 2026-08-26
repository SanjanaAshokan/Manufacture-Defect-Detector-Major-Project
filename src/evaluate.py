"""
Evaluation Script — Metrics, Confusion Matrix, PR/ROC Curves, Threshold Analysis
=================================================================================
Evaluates the trained model on the test set and generates comprehensive
performance reports and visualizations.

Usage:
    python src/evaluate.py
    python src/evaluate.py --checkpoint outputs/checkpoints/best_efficientnet_b0.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score,
)
from tqdm import tqdm
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.models import get_model
from src.gradcam import batch_gradcam
from src.utils import load_checkpoint, get_device, set_seed


CLASS_NAMES = ["Good", "Defective"]


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> dict:
    """
    Run the model on the test set and collect predictions.
    
    Returns:
        Dictionary with all_labels, all_preds, all_probs.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
    }


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    """Compute classification metrics."""
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary", pos_label=1),
        "recall": recall_score(labels, preds, average="binary", pos_label=1),
        "f1_score": f1_score(labels, preds, average="binary", pos_label=1),
    }

    # AUC scores (need probabilities)
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs[:, 1])
        metrics["avg_precision"] = average_precision_score(labels, probs[:, 1])
    except ValueError:
        metrics["roc_auc"] = 0.0
        metrics["avg_precision"] = 0.0

    return metrics


def compute_defect_type_breakdown(labels: np.ndarray, preds: np.ndarray, sample_paths: list) -> dict:
    """
    Break recall down by the specific defect type (bent_lead, cut_lead,
    damaged_case, misplaced), recovered from the filename prefix that
    download_dataset.py adds. This is what actually tells you *which* kind
    of defect the model is missing, instead of a single blended recall
    number that can hide a defect type it gets wrong almost every time.
    """
    known_defect_types = ["bent_lead", "cut_lead", "damaged_case", "misplaced"]
    per_type = defaultdict(lambda: {"total": 0, "correct": 0})

    for label, pred, path in zip(labels, preds, sample_paths):
        if label != 1:
            continue  # only defective samples carry a defect-type prefix
        filename = Path(path).name
        defect_type = next((t for t in known_defect_types if filename.startswith(t)), "unknown")
        per_type[defect_type]["total"] += 1
        if pred == label:
            per_type[defect_type]["correct"] += 1

    breakdown = {}
    for defect_type, counts in per_type.items():
        total = counts["total"]
        recall = counts["correct"] / total if total else 0.0
        breakdown[defect_type] = {
            "support": total,
            "correct": counts["correct"],
            "recall": round(recall, 4),
        }
    return breakdown


def compute_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    """
    Expected Calibration Error (ECE) for the "defective" probability.
    A model can have high accuracy but still be badly overconfident (e.g.
    saying 99% when it's right 70% of the time) — ECE is what actually
    tells you whether the displayed confidence numbers can be trusted,
    which plain accuracy/precision cannot.
    """
    confidences = probs[:, 1]
    predictions = (confidences >= 0.5).astype(int)
    correct = (predictions == labels).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    bins_report = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)
        bins_report.append({
            "range": f"{lo:.1f}-{hi:.1f}",
            "count": int(bin_count),
            "avg_confidence": round(float(bin_conf), 4),
            "avg_accuracy": round(float(bin_acc), 4),
        })

    return {"ece": round(float(ece), 4), "bins": bins_report}


def plot_confusion_matrix(labels, preds, save_path: str):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"size": 18},
        linewidths=0.5, linecolor="gray"
    )
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved to {save_path}")


def plot_precision_recall_curve(labels, probs, save_path: str):
    """Plot and save the Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probs[:, 1])
    ap = average_precision_score(labels, probs[:, 1])

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#7c3aed", linewidth=2, label=f"AP = {ap:.3f}")
    plt.fill_between(recall, precision, alpha=0.2, color="#7c3aed")
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall Curve", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Precision-Recall curve saved to {save_path}")


def plot_roc_curve(labels, probs, save_path: str):
    """Plot and save the ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#00d4ff", linewidth=2, label=f"AUC = {roc_auc_val:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    plt.fill_between(fpr, tpr, alpha=0.2, color="#00d4ff")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] ROC curve saved to {save_path}")


def plot_training_history(history: dict, save_path: str):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", linewidth=2, label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r-", linewidth=2, label="Val Accuracy")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training history saved to {save_path}")


def threshold_analysis(labels, probs, thresholds: list) -> list:
    """
    Analyze precision/recall at different confidence thresholds.
    
    For production use: higher thresholds = fewer false positives but more false negatives.
    """
    results = []
    for thresh in thresholds:
        preds_at_thresh = (probs[:, 1] >= thresh).astype(int)
        p = precision_score(labels, preds_at_thresh, average="binary", zero_division=0)
        r = recall_score(labels, preds_at_thresh, average="binary", zero_division=0)
        f1 = f1_score(labels, preds_at_thresh, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds_at_thresh)
        results.append({
            "threshold": thresh,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate transistor defect detector")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (overrides config)")
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--generate-gradcam", action="store_true", help="Generate Grad-CAM samples")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    architecture = args.architecture or config["model"]["architecture"]
    plots_dir = Path(config["evaluation"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["training"]["seed"])
    device = get_device()

    # Load model
    model, gradcam_target_layer = get_model(
        architecture=architecture,
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        hidden_dim=config["model"]["hidden_dim"],
        pretrained=False,
        freeze_backbone=False,
    )

    # Load checkpoint
    checkpoint_path = args.checkpoint or str(
        Path(config["training"]["checkpoint_dir"]) / f"best_{architecture}.pth"
    )
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)

    # Load test data
    dataloaders = get_dataloaders(
        processed_dir=config["dataset"]["processed_dir"],
        batch_size=config["training"]["batch_size"],
        image_size=config["dataset"]["image_size"],
        num_workers=config["training"]["num_workers"],
        pin_memory=False,
        use_weighted_sampler=False,
    )

    # ---- Evaluate ----
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = evaluate_model(model, dataloaders["test"], device)
    metrics = compute_metrics(results["labels"], results["preds"], results["probs"])

    # Print metrics
    print(f"\n  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.1f}%)")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1-score:          {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {metrics['avg_precision']:.4f}")

    # Classification report
    report_text = classification_report(results['labels'], results['preds'], target_names=CLASS_NAMES)
    report_dict = classification_report(
        results['labels'], results['preds'], target_names=CLASS_NAMES, output_dict=True
    )
    print(f"\n{report_text}")

    # ---- Per-defect-type breakdown (which defect types are actually missed) ----
    test_sample_paths = [p for p, _ in dataloaders["test"].dataset.samples]
    defect_breakdown = compute_defect_type_breakdown(results["labels"], results["preds"], test_sample_paths)
    if defect_breakdown:
        print("\n" + "=" * 60)
        print("PER-DEFECT-TYPE RECALL (test set)")
        print("=" * 60)
        for dtype, stats in sorted(defect_breakdown.items()):
            print(f"  {dtype:>15}: {stats['correct']}/{stats['support']} caught  ({stats['recall'] * 100:.1f}%)")

    # ---- Confidence calibration (is the displayed % trustworthy?) ----
    calibration = compute_calibration_error(results["labels"], results["probs"])
    print(f"\n  Expected Calibration Error (ECE): {calibration['ece']:.4f}")
    print("  (0 = perfectly calibrated; >0.1 usually means confidence numbers")
    print("   are overstated and shouldn't be shown to users as-is — consider")
    print("   running src/calibrate.py to fit a temperature-scaling correction.)")

    # ---- Generate Plots ----
    print("\n[Plots] Generating evaluation plots...")
    plot_confusion_matrix(results["labels"], results["preds"], str(plots_dir / "confusion_matrix.png"))
    plot_precision_recall_curve(results["labels"], results["probs"], str(plots_dir / "pr_curve.png"))
    plot_roc_curve(results["labels"], results["probs"], str(plots_dir / "roc_curve.png"))

    # Training history (if available in checkpoint)
    if "history" in checkpoint:
        plot_training_history(checkpoint["history"], str(plots_dir / "training_history.png"))

    # ---- Threshold Analysis ----
    thresholds = config["evaluation"]["thresholds"]
    thresh_results = threshold_analysis(results["labels"], results["probs"], thresholds)

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS (for production tuning)")
    print("=" * 60)
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
    print("  " + "-" * 54)
    for r in thresh_results:
        print(
            f"  {r['threshold']:>10.2f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1_score']:>10.4f} {r['accuracy']:>10.4f}"
        )

    # ---- Save Results ----
    results_data = {
        "architecture": architecture,
        "metrics": metrics,
        "per_class_report": report_dict,
        "defect_type_breakdown": defect_breakdown,
        "calibration": calibration,
        "threshold_analysis": thresh_results,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        # Traceability: these are the numbers that get shown on the public
        # dashboard/landing page, so record exactly when/what they came from
        # instead of letting a website display a stale run forever.
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "num_test_samples": int(len(results["labels"])),
        "test_class_counts": {
            "good": int((results["labels"] == 0).sum()),
            "defective": int((results["labels"] == 1).sum()),
        },
    }

    results_path = plots_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[Results] Saved to {results_path}")

    # ---- Grad-CAM Samples ----
    if args.generate_gradcam:
        print("\n[Grad-CAM] Generating sample visualizations...")
        batch_gradcam(
            model=model,
            dataloader=dataloaders["test"],
            target_layer=gradcam_target_layer,
            save_dir=config["evaluation"]["gradcam_dir"],
            device=str(device),
            num_samples=config["evaluation"]["num_gradcam_samples"],
        )

    print("\n[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
