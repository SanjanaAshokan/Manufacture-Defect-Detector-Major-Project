"""
Flask Web Application — Transistor Defect Detector
===================================================
Custom website with:
  - Landing page with hero section and feature showcase
  - Detection page with drag-and-drop upload and AJAX inference
  - Dashboard page with model metrics and threshold analysis
  - About page with architecture explanation
  - REST API endpoint for inference

Deployed on Hugging Face Spaces via Docker SDK (port 7860).
"""

import os
import sys
import json
import traceback
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone

import torch
import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.gradcam import predict_with_gradcam, image_to_base64
from src.utils import load_checkpoint, get_device

# ============================================================
# Flask App Initialization
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Global model state
model_cache = {}
# Tracks, per architecture, whether we're serving a real trained checkpoint
# or silently falling back to untrained/ImageNet-only weights. This is the
# single source of truth the whole site uses to avoid presenting fallback
# predictions or stale metrics as if they were production-validated.
model_status_cache = {}
CONFIG = None
DEVICE = None


def load_config():
    """Load project configuration."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_or_load_model(architecture: str):
    """
    Load a model into cache (or return cached version).
    Falls back to pretrained ImageNet model if no checkpoint exists.
    """
    if architecture in model_cache:
        return model_cache[architecture]

    model, target_layer = get_model(
        architecture=architecture,
        num_classes=CONFIG["model"]["num_classes"],
        dropout=CONFIG["model"]["dropout"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        pretrained=True,
        freeze_backbone=False,
    )

    # Try to load checkpoint
    checkpoint_path = PROJECT_ROOT / CONFIG["training"]["checkpoint_dir"] / f"best_{architecture}.pth"
    if not checkpoint_path.exists():
        repo_id = os.environ.get("HF_MODEL_REPO")
        if repo_id:
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=repo_id,
                    filename=f"best_{architecture}.pth",
                    local_dir=str(checkpoint_path.parent),
                    token=os.environ.get("HF_TOKEN"),
                )
            except Exception as e:
                print(f"[Web] Could not download weights from {repo_id}: {e}")
    if checkpoint_path.exists():
        checkpoint = load_checkpoint(model, str(checkpoint_path), DEVICE)
        print(f"[Web] Loaded trained checkpoint for {architecture}")

        # Pick up a fitted temperature-scaling factor if calibrate.py has been
        # run for this architecture, so predictions shown in the UI report a
        # calibrated (trustworthy) confidence instead of a raw, often
        # overconfident softmax value.
        temperature = 1.0
        calib_path = PROJECT_ROOT / CONFIG["evaluation"]["plots_dir"] / f"calibration_{architecture}.json"
        if calib_path.exists():
            with open(calib_path, "r") as f:
                temperature = json.load(f).get("temperature", 1.0)
            print(f"[Web] Loaded calibration for {architecture}: temperature={temperature:.3f}")

        model_status_cache[architecture] = {
            "trained": True,
            "source": "trained_checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "temperature": temperature,
            "calibrated": calib_path.exists(),
        }
    else:
        # IMPORTANT: the classifier head here is randomly initialized (only the
        # backbone is ImageNet-pretrained). Predictions from this fallback are
        # not meaningful for defect classification — every route that surfaces
        # predictions or metrics must check this status before showing them
        # as trustworthy.
        print(f"[Web] No checkpoint found for {architecture} — using pretrained ImageNet weights")
        print(f"      Expected: {checkpoint_path}")
        print(f"      Train a model first: python src/train.py --architecture {architecture}")
        model_status_cache[architecture] = {
            "trained": False,
            "source": "imagenet_backbone_untrained_head",
            "checkpoint_path": str(checkpoint_path),
            "epoch": None,
            "temperature": 1.0,
            "calibrated": False,
        }

    model = model.to(DEVICE)
    model.eval()

    model_cache[architecture] = (model, target_layer)
    return model, target_layer


def get_model_status(architecture: str) -> dict:
    """Return the trained/fallback status for an architecture (loading it if needed)."""
    if architecture not in model_status_cache:
        get_or_load_model(architecture)
    return model_status_cache[architecture]


def any_model_trained() -> bool:
    """True if at least one architecture currently has a real trained checkpoint loaded."""
    return any(status.get("trained") for status in model_status_cache.values())


def get_evaluation_results():
    """Load pre-computed evaluation results if available."""
    results_path = PROJECT_ROOT / CONFIG["evaluation"]["plots_dir"] / "evaluation_results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            return json.load(f)
    return None


def get_available_plots():
    """Check which evaluation plots are available."""
    plots_dir = PROJECT_ROOT / CONFIG["evaluation"]["plots_dir"]
    available = {}
    for plot_name in ["confusion_matrix", "pr_curve", "roc_curve", "training_history"]:
        plot_path = plots_dir / f"{plot_name}.png"
        if plot_path.exists():
            available[plot_name] = True
    return available


# ============================================================
# Site-wide context (model status banner)
# ============================================================

@app.context_processor
def inject_model_status():
    """
    Make model-trained status available to every template so the site never
    silently presents fallback (untrained) predictions or unverified metrics
    as production-ready — this is checked on every page, not just Dashboard.
    """
    default_arch = CONFIG["model"]["architecture"] if CONFIG else None
    status = model_status_cache.get(default_arch, {}) if default_arch else {}
    return {
        "site_model_trained": bool(status.get("trained")),
        "site_default_architecture": default_arch,
    }


# ============================================================
# Advanced OOD Anomaly Detector
# ============================================================
ood_model_cache = None
ood_ref_embedding = None

def get_ood_model_and_ref():
    """Lazily loads a lightweight feature extractor for robust OOD detection."""
    global ood_model_cache, ood_ref_embedding
    if ood_model_cache is not None:
        return ood_model_cache, ood_ref_embedding
        
    import torch
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    import os
    
    print("[Web] Loading robust OOD Anomaly Detector...")
    # Load lightweight MobileNetV3
    ood_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    ood_model.classifier = torch.nn.Identity()
    ood_model.eval()
    ood_model.to(DEVICE)
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    examples_dir = PROJECT_ROOT / "web" / "examples"
    embeddings = []
    
    if examples_dir.exists():
        # Build reference profile from good examples
        for f in os.listdir(examples_dir):
            if "good" in f.lower() and f.endswith((".png", ".jpg")):
                try:
                    img = Image.open(examples_dir / f).convert("RGB")
                    tensor = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        feats = ood_model(tensor)
                        embeddings.append(feats)
                except Exception:
                    pass
                    
    if embeddings:
        ref = torch.mean(torch.cat(embeddings), dim=0, keepdim=True)
        ood_ref_embedding = F.normalize(ref, p=2, dim=1)
        
    ood_model_cache = (ood_model, transform)
    return ood_model_cache, ood_ref_embedding

# ============================================================
# Routes
# ============================================================

@app.route("/")
def index():
    """Landing page."""
    eval_results = get_evaluation_results()
    return render_template("index.html", results=eval_results)


@app.route("/detect")
def detect():
    """Detection / upload page."""
    # Get example images
    examples_dir = PROJECT_ROOT / "web" / "examples"
    examples = []
    if examples_dir.exists():
        for img_path in sorted(examples_dir.glob("*")):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                examples.append(img_path.name)
    default_arch = CONFIG["model"]["architecture"]
    model_status = get_model_status(default_arch)
    return render_template("detect.html", examples=examples, model_status=model_status)


@app.route("/realtime")
def realtime():
    """Real-time webcam interface page."""
    default_arch = CONFIG["model"]["architecture"]
    model_status = get_model_status(default_arch)
    return render_template("realtime.html", model_status=model_status)


@app.route("/dashboard")
def dashboard():
    """Model metrics dashboard."""
    eval_results = get_evaluation_results()
    available_plots = get_available_plots()
    default_arch = CONFIG["model"]["architecture"]
    model_status = get_model_status(default_arch)
    return render_template(
        "dashboard.html",
        results=eval_results,
        plots=available_plots,
        model_status=model_status,
    )


@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")


# ============================================================
# API Endpoints
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Run inference on an uploaded image.
    
    Accepts: multipart/form-data with 'image' file field.
    Optional: 'architecture' (string), 'threshold' (float).
    
    Returns: JSON with prediction, confidence, probabilities, and
             base64-encoded Grad-CAM images.
    """
    try:
        # Validate input
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Check file extension
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            return jsonify({"error": f"Unsupported file type: {ext}. Allowed: {allowed}"}), 400

        # Load image
        image_bytes = file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # OOD Check
        import numpy as np
        import cv2
        bypass_ood = request.form.get("bypass_ood") == "true"
        if not bypass_ood:
            # 1. Basic image quality check
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            std_dev = np.std(gray)
            mean_val = np.mean(gray)
            
            if std_dev > 80 or std_dev < 5 or mean_val < 10 or mean_val > 240:
                return jsonify({"error": "Image appears to be out-of-domain (too uniform, dark, or bright). Please upload an image of a transistor.", "success": False}), 400
                
            # 2. Advanced Anomaly Detection (Cosine Similarity)
            try:
                import torch
                import torch.nn.functional as F
                (ood_model, transform), ref_emb = get_ood_model_and_ref()
                
                if ref_emb is not None:
                    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        feats = ood_model(img_tensor)
                        feats_norm = F.normalize(feats, p=2, dim=1)
                        # Compute cosine similarity with reference
                        similarity = torch.sum(feats_norm * ref_emb).item()
                        
                    # Usually similarity > 0.6 for in-domain, < 0.4 for completely different objects (faces, forks)
                    if similarity < 0.40:
                        return jsonify({"error": f"Object does not match a transistor (Similarity: {similarity:.2f}). Please scan a transistor.", "success": False}), 400
            except Exception as e:
                import traceback
                traceback.print_exc()
                # Fallback to OpenCV checks if torch OOD fails...
                circles = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10, 
                    param1=50, param2=30, minRadius=2, maxRadius=30
                )
                num_circles = len(circles[0]) if circles is not None else 0
                if num_circles < 5:
                    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cnt = max(contours, key=cv2.contourArea)
                        rect = cv2.minAreaRect(cnt)
                        (x, y), (w, h), angle = rect
                        if w > 0 and h > 0:
                            aspect_ratio = max(w/h, h/w)
                            area = w * h
                            if aspect_ratio > 2.5:
                                return jsonify({"error": "Object shape does not match a transistor (too elongated like a fork or cable). Please scan a transistor.", "success": False}), 400
                            if area < (gray.shape[0] * gray.shape[1] * 0.01):
                                return jsonify({"error": "No significant object detected. Please move closer.", "success": False}), 400

        # Get parameters
        architecture = request.form.get("architecture", CONFIG["model"]["architecture"])
        threshold = float(request.form.get("threshold", 0.65))

        # Load model
        model, target_layer = get_or_load_model(architecture)
        status = get_model_status(architecture)

        # Run inference with Grad-CAM
        result = predict_with_gradcam(
            model=model,
            image=image,
            target_layer=target_layer,
            device=str(DEVICE),
            image_size=CONFIG["dataset"]["image_size"],
            temperature=status.get("temperature", 1.0),
        )

        # Determine status based on threshold slider
        is_defective = result["probabilities"]["defective"] >= threshold
        prediction_text = "Defective" if is_defective else "Good"

        # Flag low-margin predictions as uncertain
        margin = abs(result["probabilities"]["defective"] - threshold)
        is_uncertain = margin < 0.10

        # Build response
        response = {
            "success": True,
            "prediction": prediction_text,
            "confidence": round(result["confidence"], 4),
            "probabilities": {
                k: round(v, 4) for k, v in result["probabilities"].items()
            },
            "is_defective": is_defective,
            "is_uncertain": is_uncertain,
            "threshold": threshold,
            "architecture": architecture,
            "model_trained": status["trained"],
            "calibrated": status.get("calibrated", False),
            "original_image": image_to_base64(result["original_image"]),
            "heatmap": image_to_base64(result["heatmap"]),
            "overlay": image_to_base64(result["overlay"]),
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500



@app.route("/api/health")
def api_health():
    """Health check endpoint. Surfaces real checkpoint status per model."""
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "models_loaded": list(model_cache.keys()),
        "model_status": model_status_cache,
        "any_model_trained": any_model_trained(),
    })


# ============================================================
# Static File Serving
# ============================================================

@app.route("/examples/<filename>")
def serve_example(filename):
    """Serve example images."""
    examples_dir = PROJECT_ROOT / "web" / "examples"
    return send_from_directory(str(examples_dir), filename)


@app.route("/plots/<filename>")
def serve_plot(filename):
    """Serve evaluation plot images."""
    plots_dir = PROJECT_ROOT / CONFIG["evaluation"]["plots_dir"]
    return send_from_directory(str(plots_dir), filename)


# ============================================================
# Main
# ============================================================

def init_app():
    """Initialize the application."""
    global CONFIG, DEVICE

    print("\n" + "=" * 60)
    print("🔬 Transistor Defect Detector — Web Server")
    print("=" * 60)

    CONFIG = load_config()
    DEVICE = get_device()

    # Pre-load the default model
    default_arch = CONFIG["model"]["architecture"]
    print(f"\n[Web] Pre-loading default model: {default_arch}")
    get_or_load_model(default_arch)

    print("\n[Web] Server ready!")
    print(f"[Web] Access at: http://localhost:{CONFIG['web']['port']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    init_app()
    app.run(
        host=CONFIG["web"]["host"],
        port=CONFIG["web"]["port"],
        debug=CONFIG["web"]["debug"],
    )
