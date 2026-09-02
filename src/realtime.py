"""
Real-time Defect Detection with Grad-CAM Visualization
====================================================
Captures video from the webcam, runs inference frame-by-frame,
and displays the Grad-CAM overlay in real-time.

Usage:
    python src/realtime.py
"""

import sys
import argparse
from pathlib import Path
import time
import cv2
import torch
import yaml
from PIL import Image
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.gradcam import predict_with_gradcam
from src.utils import load_checkpoint, get_device

def main():
    parser = argparse.ArgumentParser(description="Real-time defect detection from webcam")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    architecture = args.architecture or config["model"]["architecture"]
    device = get_device()

    print(f"[Info] Loading model {architecture} on {device}...")
    model, gradcam_target_layer = get_model(
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
    
    if Path(checkpoint_path).exists():
        load_checkpoint(model, checkpoint_path, device)
        print(f"[Info] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[Warning] Checkpoint not found at {checkpoint_path}. Using untrained weights.")

    model = model.to(device)
    model.eval()

    # Initialize video capture
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[Error] Could not open camera {args.camera}.")
        print("Make sure it's connected and not used by another application.")
        sys.exit(1)

    print("[Info] Starting real-time capture. Press 'q' to exit.")

    fps_start_time = time.time()
    fps_frames = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] Failed to read frame from camera.")
            break

        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Predict
        try:
            result = predict_with_gradcam(
                model=model,
                image=pil_image,
                target_layer=gradcam_target_layer,
                device=str(device),
                image_size=config["dataset"]["image_size"],
            )
            
            # The overlay is a PIL Image
            overlay_pil = result["overlay"]
            # Convert back to BGR for OpenCV
            overlay_bgr = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)

            # Resize the overlay for better viewing if it's too small (default is 224x224 usually from gradcam)
            # The output of Grad-CAM overlay matches the target image_size by default (224)
            # We can scale it up for the desktop window so it's easier to see.
            display_size = (640, 640)
            overlay_bgr_resized = cv2.resize(overlay_bgr, display_size, interpolation=cv2.INTER_LINEAR)

            is_defective = result["probabilities"]["defective"] >= args.threshold
            status_text = "DEFECTIVE" if is_defective else "GOOD"
            color = (0, 0, 255) if is_defective else (0, 255, 0)

            # Display prediction and confidence
            text = f"Status: {status_text} ({result['confidence']:.1%})"
            cv2.putText(overlay_bgr_resized, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Display FPS
            fps_frames += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_frames / (time.time() - fps_start_time)
                fps_frames = 0
                fps_start_time = time.time()
            cv2.putText(overlay_bgr_resized, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Real-time Defect Detector", overlay_bgr_resized)
            
        except Exception as e:
            print(f"[Error] Inference failed: {e}")
            # Just show the original frame if inference fails
            cv2.imshow("Real-time Defect Detector", frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Info] Exiting.")

if __name__ == "__main__":
    main()
