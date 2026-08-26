"""
Single-Image Prediction with Grad-CAM Visualization
====================================================
Load a trained model, run inference on a single image, and
generate a Grad-CAM explanation overlay.

Usage:
    python src/predict.py --image path/to/transistor.png
    python src/predict.py --image path/to/image.png --architecture resnet50
"""

import sys
import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.gradcam import predict_with_gradcam
from src.utils import load_checkpoint, get_device


def main():
    parser = argparse.ArgumentParser(description="Predict defect on a single transistor image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Path to save the output image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    architecture = args.architecture or config["model"]["architecture"]
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
    load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] Image not found: {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")

    # Predict with Grad-CAM
    result = predict_with_gradcam(
        model=model,
        image=image,
        target_layer=gradcam_target_layer,
        device=str(device),
        image_size=config["dataset"]["image_size"],
    )

    # Print results
    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"  Image:      {image_path.name}")
    print(f"  Prediction: {result['predicted_label'].upper()}")
    print(f"  Confidence: {result['confidence']:.4f} ({result['confidence'] * 100:.1f}%)")
    print(f"  Threshold:  {args.threshold}")

    is_defective = result["probabilities"]["defective"] >= args.threshold
    if is_defective:
        print(f"  Status:     ⚠ DEFECTIVE (above threshold)")
    else:
        print(f"  Status:     ✓ GOOD (below threshold)")

    for cls, prob in result["probabilities"].items():
        print(f"  P({cls}):  {prob:.4f}")
    print("=" * 50)

    # Save visualization
    output_path = args.output or str(image_path.parent / f"{image_path.stem}_gradcam.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(result["original_image"])
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(result["heatmap"])
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(result["overlay"])
    axes[2].set_title("Overlay", fontsize=13)
    axes[2].axis("off")

    status = "DEFECTIVE ⚠" if is_defective else "GOOD ✓"
    fig.suptitle(
        f"Prediction: {result['predicted_label'].upper()} | "
        f"Confidence: {result['confidence']:.1%} | Status: {status}",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Visualization → {output_path}")


if __name__ == "__main__":
    main()
