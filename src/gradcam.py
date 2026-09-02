"""
Grad-CAM Implementation for Defect Visualization
=================================================
Generates Grad-CAM heatmaps to explain model predictions by
highlighting the regions that contributed most to the classification.
Uses the pytorch-grad-cam library for robust hook-based computation.
"""

import io
import base64
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Get the transform pipeline for inference (same as validation)."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])


def preprocess_image(
    image: Image.Image,
    image_size: int = 224,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess an image for inference and Grad-CAM visualization.
    
    Args:
        image: PIL Image.
        image_size: Target size.
    
    Returns:
        Tuple of (input_tensor [1, 3, H, W], rgb_image [H, W, 3] in [0, 1] range).
    """
    # Prepare the normalized tensor for the model
    transform = get_inference_transform(image_size)
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

    # Prepare the RGB image for Grad-CAM overlay (needs to be in [0, 1])
    resize_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
    ])
    resized_image = resize_transform(image)
    rgb_image = np.array(resized_image).astype(np.float32) / 255.0

    return input_tensor, rgb_image


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer,
    target_class: Optional[int] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap for the given input.
    
    Args:
        model: Trained model in eval mode.
        input_tensor: Preprocessed input tensor [1, 3, H, W].
        target_layer: The convolutional layer to compute Grad-CAM on.
        target_class: Class index to explain (None = predicted class).
        device: Device to run on.
    
    Returns:
        Grayscale heatmap as numpy array [H, W] in [0, 1].
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    # Set up targets
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    # Create Grad-CAM (use GradCAMPlusPlus for EfficientNet due to known gradient artifacts with base GradCAM)
    if "efficientnet" in model.__class__.__name__.lower():
        from pytorch_grad_cam import GradCAMPlusPlus
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    else:
        cam = GradCAM(model=model, target_layers=[target_layer])
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam[0, :]  # Return first (and only) image's heatmap


def create_gradcam_overlay(
    rgb_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Create a Grad-CAM overlay on the original image.
    
    Args:
        rgb_image: Original RGB image [H, W, 3] in [0, 1].
        heatmap: Grayscale heatmap [H, W] in [0, 1].
        alpha: Overlay transparency.
        colormap: OpenCV colormap for the heatmap.
    
    Returns:
        Overlay image [H, W, 3] in [0, 255] as uint8.
    """
    visualization = show_cam_on_image(rgb_image, heatmap, use_rgb=True, image_weight=1 - alpha)
    return visualization


def create_colored_heatmap(heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Create a standalone colored heatmap (without the original image).
    
    Args:
        heatmap: Grayscale heatmap [H, W] in [0, 1].
        colormap: OpenCV colormap.
    
    Returns:
        Colored heatmap [H, W, 3] as uint8 in RGB.
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored_rgb


def predict_with_gradcam(
    model: torch.nn.Module,
    image: Image.Image,
    target_layer,
    device: str = "cpu",
    image_size: int = 224,
    class_names: list = None,
    temperature: float = 1.0,
) -> dict:
    """
    Run inference and generate Grad-CAM visualization.
    
    Args:
        model: Trained model.
        image: PIL Image.
        target_layer: Grad-CAM target layer.
        device: Device.
        image_size: Target image size.
        class_names: List of class names.
        temperature: Optional temperature-scaling factor (see src/calibrate.py).
            Dividing logits by a fitted temperature > 1 corrects overconfident
            softmax outputs so the displayed percentage matches real accuracy.
            Leave at 1.0 (no-op) if no calibration has been fit yet.
    
    Returns:
        Dictionary with prediction results and visualizations.
    """
    if class_names is None:
        class_names = ["good", "defective"]

    model.eval()

    # Preprocess
    input_tensor, rgb_image = preprocess_image(image, image_size)
    input_tensor = input_tensor.to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs / max(temperature, 1e-3), dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = predicted.item()
    confidence_score = confidence.item()
    all_probs = probabilities[0].cpu().numpy()

    # Generate Grad-CAM for predicted class
    heatmap = generate_gradcam(model, input_tensor, target_layer, predicted_class, device)

    # Create visualizations
    overlay = create_gradcam_overlay(rgb_image, heatmap)
    colored_heatmap = create_colored_heatmap(heatmap)

    # Convert to PIL for easy display / encoding
    overlay_pil = Image.fromarray(overlay)
    heatmap_pil = Image.fromarray(colored_heatmap)
    original_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))

    return {
        "predicted_class": predicted_class,
        "predicted_label": class_names[predicted_class],
        "confidence": confidence_score,
        "probabilities": {name: float(prob) for name, prob in zip(class_names, all_probs)},
        "original_image": original_pil,
        "heatmap": heatmap_pil,
        "overlay": overlay_pil,
        "heatmap_raw": heatmap,
    }


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def batch_gradcam(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_layer,
    save_dir: str,
    device: str = "cpu",
    num_samples: int = 20,
    class_names: list = None,
):
    """
    Generate and save Grad-CAM visualizations for a batch of images.
    
    Args:
        model: Trained model.
        dataloader: DataLoader to sample from.
        target_layer: Grad-CAM target layer.
        save_dir: Directory to save visualizations.
        device: Device.
        num_samples: Number of samples to process.
        class_names: List of class names.
    """
    import matplotlib.pyplot as plt

    if class_names is None:
        class_names = ["good", "defective"]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    count = 0

    for images, labels in dataloader:
        if count >= num_samples:
            break

        images = images.to(device)

        for i in range(images.shape[0]):
            if count >= num_samples:
                break

            input_tensor = images[i].unsqueeze(0)
            true_label = labels[i].item()

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

            # Grad-CAM
            heatmap = generate_gradcam(model, input_tensor, target_layer, pred.item(), device)

            # Denormalize image for display
            img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img = img * IMAGENET_STD + IMAGENET_MEAN
            img = np.clip(img, 0, 1)

            # Create overlay
            overlay = create_gradcam_overlay(img, heatmap)

            # Plot side-by-side
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img)
            axes[0].set_title("Original")
            axes[0].axis("off")

            colored_hm = create_colored_heatmap(heatmap)
            axes[1].imshow(colored_hm)
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            pred_label = class_names[pred.item()]
            true_label_str = class_names[true_label]
            correct = "✓" if pred.item() == true_label else "✗"

            fig.suptitle(
                f"{correct}  True: {true_label_str} | Predicted: {pred_label} "
                f"(Confidence: {conf.item():.1%})",
                fontsize=14, fontweight="bold"
            )

            plt.tight_layout()
            plt.savefig(save_dir / f"gradcam_{count:03d}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            count += 1

    print(f"[Grad-CAM] Saved {count} visualizations to {save_dir}")
