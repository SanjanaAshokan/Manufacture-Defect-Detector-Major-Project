"""
Flask Web Application — Transistor Defect Detector
===================================================
Custom website with:
  - Landing page with hero section and feature showcase
  - Detection page with drag-and-drop upload and AJAX inference
  - Dashboard page with model metrics and threshold analysis
  - About page with architecture explanation
  - Batch inspection, three-model consensus, robustness lab and production-line
    simulator pages
  - REST API endpoints for inference

Deployed on Hugging Face Spaces via Docker SDK (port 7860).
"""

import os
import sys
import json
import base64
import threading
import time
import traceback
import zipfile
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageOps

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.gradcam import predict_with_gradcam, image_to_base64
from src.utils import load_checkpoint, get_device
from src.orientation import auto_orient
from src.diagnosis import diagnose

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

# Flask's server is multi-threaded but the models (and their Grad-CAM hooks) are shared
# objects, so every model load / forward / Grad-CAM pass is serialised through this lock.
inference_lock = threading.RLock()
init_lock = threading.Lock()

# Architectures offered in the UI (id, display name). Availability is decided at runtime
# from the checkpoints that really exist — nothing here claims a model is trained.
ARCHITECTURES = [
    ("convnext_tiny", "ConvNeXt-Tiny"),
    ("efficientnet_b0", "EfficientNet-B0"),
    ("resnet50", "ResNet-50"),
]
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
BATCH_MAX_IMAGES = 30
# Firebase Hosting rewrites to Cloud Run end a request after 60 s, so a batch call stops
# starting new images after this many seconds and reports the rest as "skipped".
BATCH_TIME_BUDGET_S = float(os.environ.get("BATCH_TIME_BUDGET", 40))
hf_download_failed = set()


class ApiError(Exception):
    """Raised for client mistakes so they come back as clean JSON 4xx, not a 500."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def load_config():
    """Load project configuration."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def checkpoint_path_for(architecture: str) -> Path:
    return PROJECT_ROOT / CONFIG["training"]["checkpoint_dir"] / f"best_{architecture}.pth"


def arch_available(architecture: str) -> bool:
    """
    Cheap check (no model load): is there a trained checkpoint for this architecture,
    or a model repo configured that may provide one?
    """
    if checkpoint_path_for(architecture).exists():
        return True
    remote = os.environ.get("HF_MODEL_REPO") or os.environ.get("MODEL_WEIGHTS_BASE_URL")
    return bool(remote) and architecture not in hf_download_failed


def download_weights(url: str, destination: Path) -> bool:
    """Fetch one checkpoint over HTTPS (e.g. a Google Cloud Storage object) into `destination`."""
    import urllib.request
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(destination)
        print(f"[Web] Downloaded weights from {url}")
        return True
    except Exception as e:
        print(f"[Web] Could not download weights from {url}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def get_or_load_model(architecture: str):
    """
    Load a model into cache (or return cached version).
    Falls back to an ImageNet-only backbone if no checkpoint exists.
    """
    with inference_lock:
        if architecture in model_cache:
            return model_cache[architecture]

        # Try to find / download the checkpoint first, so ImageNet weights are only
        # fetched when they are really needed (a trained checkpoint overwrites them).
        checkpoint_path = checkpoint_path_for(architecture)
        if not checkpoint_path.exists():
            base_url = os.environ.get("MODEL_WEIGHTS_BASE_URL")
            if base_url and not download_weights(f"{base_url.rstrip('/')}/best_{architecture}.pth", checkpoint_path):
                hf_download_failed.add(architecture)
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
                    hf_download_failed.add(architecture)
                    print(f"[Web] Could not download weights from {repo_id}: {e}")

        model, target_layer = get_model(
            architecture=architecture,
            num_classes=CONFIG["model"]["num_classes"],
            dropout=CONFIG["model"]["dropout"],
            hidden_dim=CONFIG["model"]["hidden_dim"],
            pretrained=not checkpoint_path.exists(),
            freeze_backbone=False,
        )

        if checkpoint_path.exists():
            checkpoint = load_checkpoint(model, str(checkpoint_path), DEVICE)
            print(f"[Web] Loaded trained checkpoint for {architecture}")

            # Pick up a fitted temperature-scaling factor if calibrate.py has been
            # run for this architecture, so predictions shown in the UI report a
            # calibrated (trustworthy) confidence instead of a raw, often
            # overconfident softmax value.
            temperature = 1.0
            calibrated = False
            calib_path = PROJECT_ROOT / CONFIG["evaluation"]["plots_dir"] / f"calibration_{architecture}.json"
            if calib_path.exists():
                try:
                    with open(calib_path, "r") as f:
                        calibration = json.load(f)
                    candidate = float(calibration.get("temperature", 1.0))
                    before = float(calibration["ece_before"])
                    after = float(calibration["ece_after"])
                    same_checkpoint = calibration.get("checkpoint_epoch") == (
                        checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
                    )
                    if candidate > 0 and after < before and same_checkpoint:
                        temperature = candidate
                        calibrated = True
                        print(f"[Web] Loaded validated calibration for {architecture}: temperature={temperature:.3f}")
                    else:
                        print(f"[Web] Ignoring calibration for {architecture}: it did not improve ECE or does not match this checkpoint")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    print(f"[Web] Ignoring invalid calibration for {architecture}: {exc}")

            model_status_cache[architecture] = {
                "trained": True,
                "source": "trained_checkpoint",
                "checkpoint_path": str(checkpoint_path),
                "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
                "temperature": temperature,
                "calibrated": calibrated,
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
    architectures = []
    for arch_id, label in (ARCHITECTURES if CONFIG else []):
        available = arch_available(arch_id)
        architectures.append({
            "id": arch_id,
            "label": label,
            "available": available,
            "default": arch_id == default_arch,
            "option_text": label + (" (default)" if arch_id == default_arch else "")
                           + ("" if available else " — no trained checkpoint"),
        })
    return {
        "site_model_trained": bool(status.get("trained")),
        "site_default_architecture": default_arch,
        "architectures": architectures,
    }


# ============================================================
# Advanced OOD Anomaly Detector
# ============================================================
ood_model_cache = None
ood_ref_embedding = None
ood_load_failed = False   # don't retry a failed weights download on every request

ood_lock = threading.Lock()
ood_extra_refs = None        # embeddings of optional photos in web/ood_references/


def get_ood_model_and_ref():
    """Thread-safe wrapper: the first caller loads the detector, everyone else waits for it."""
    if ood_model_cache is not None:
        return ood_model_cache, ood_ref_embedding
    with ood_lock:
        return _load_ood_model_and_ref()


def _load_ood_model_and_ref():
    """Lazily loads a lightweight feature extractor for robust OOD detection."""
    global ood_model_cache, ood_ref_embedding, ood_load_failed
    if ood_model_cache is not None:
        return ood_model_cache, ood_ref_embedding
    if ood_load_failed:
        raise RuntimeError("OOD feature extractor unavailable (weights could not be loaded)")

    import torch
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    import os
    
    print("[Web] Loading robust OOD Anomaly Detector...")
    # Load lightweight MobileNetV3
    try:
        ood_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        ood_load_failed = True
        raise
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

    # Optional extra reference photos (real-world transistors on other backgrounds / packages):
    # a frame only has to resemble ONE of them, which widens what counts as "a transistor".
    global ood_extra_refs
    extra_dir = PROJECT_ROOT / "web" / "ood_references"
    extra = []
    if extra_dir.exists():
        for f in sorted(os.listdir(extra_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                try:
                    img = Image.open(extra_dir / f).convert("RGB")
                    with torch.no_grad():
                        extra.append(F.normalize(ood_model(transform(img).unsqueeze(0).to(DEVICE)), p=2, dim=1))
                except Exception:
                    pass
    if extra:
        ood_extra_refs = torch.cat(extra)
        print(f"[Web] OOD: {len(extra)} extra reference photo(s) loaded from web/ood_references/")
        
    ood_model_cache = (ood_model, transform)
    return ood_model_cache, ood_ref_embedding

# ============================================================
# Request helpers
# ============================================================

def ensure_initialized():
    """Lazy init so the app also works under gunicorn / flask run (not only __main__)."""
    if CONFIG is None:
        with init_lock:
            if CONFIG is None:
                init_app()


@app.before_request
def _ensure_initialized():
    ensure_initialized()


@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({"error": err.message, "success": False}), err.status


@app.errorhandler(413)
def handle_too_large(_err):
    return jsonify({"error": "Upload too large. Please send smaller images.", "success": False}), 413


def per_file_limit_bytes() -> int:
    return int(CONFIG["web"].get("max_content_length", 16)) * 1024 * 1024


def decode_image(data: bytes, filename: str) -> Image.Image:
    """Validate and decode one uploaded image (EXIF-rotation applied, RGB)."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise ApiError(f"Unsupported file type: {ext or 'none'}. Allowed: {sorted(ALLOWED_IMAGE_EXTS)}")
    if len(data) > per_file_limit_bytes():
        raise ApiError(f"{filename}: file is larger than {CONFIG['web'].get('max_content_length', 16)} MB.", 413)
    try:
        image = Image.open(BytesIO(data))
        image = ImageOps.exif_transpose(image)   # phone photos carry a rotation tag
        return image.convert("RGB")
    except Exception:
        raise ApiError(f"{filename}: could not read this file as an image.")


def read_single_upload(field: str = "image") -> Image.Image:
    if field not in request.files:
        raise ApiError("No image file provided")
    file = request.files[field]
    if file.filename == "":
        raise ApiError("No file selected")
    return decode_image(file.read(), file.filename)


def parse_params(default_gradcam: bool = True) -> dict:
    """Read + validate the optional form fields shared by the inference endpoints."""
    architecture = request.form.get("architecture", CONFIG["model"]["architecture"])
    if architecture not in {a for a, _ in ARCHITECTURES} | {CONFIG["model"]["architecture"]}:
        raise ApiError(f"Unknown architecture: {architecture}")
    try:
        threshold = float(request.form.get("threshold", 0.65))
    except (TypeError, ValueError):
        raise ApiError("threshold must be a number between 0 and 1")
    if not (0.0 <= threshold <= 1.0):
        raise ApiError("threshold must be between 0 and 1")
    gradcam_raw = request.form.get("gradcam")
    return {
        "architecture": architecture,
        "threshold": threshold,
        "auto_orient": request.form.get("auto_orient", "true").lower() != "false",
        "with_gradcam": default_gradcam if gradcam_raw is None else gradcam_raw.lower() != "false",
        "bypass_ood": request.form.get("bypass_ood") == "true",
        # live camera: stricter "is this a transistor?" rule, because people, faces and
        # rooms are constantly in front of a webcam
        "live": request.form.get("live") == "true",
    }


def ood_limits():
    """(reject_below, warn_below) similarity limits; override in configs/config.yaml under `web:`."""
    web = CONFIG.get("web", {})
    # The previous 0.30 default let clearly unrelated objects through to the defect model.
    # 0.45 rejects weak matches while retaining the documented 0.40-0.50 range as a warning
    # when an operator has explicitly configured a lower cutoff.
    return float(web.get("ood_reject_below", 0.45)), float(web.get("ood_warn_below", 0.50))


def live_ood_limit() -> float:
    """
    Similarity below which a live-camera frame is never analysed. Default 0.40 - exactly the
    limit the live camera always used. Tune with the OOD_LIVE_REJECT_BELOW environment variable
    or `ood_live_reject_below` under `web:` in configs/config.yaml.
    """
    web = CONFIG.get("web", {})
    return float(os.environ.get("OOD_LIVE_REJECT_BELOW", web.get("ood_live_reject_below", 0.40)))


def has_transistor_geometry(info: dict) -> bool:
    """Require independent evidence of a transistor body plus visible lead pixels."""
    return bool(
        info
        and info.get("body_box") is not None
        and (info.get("lead_box") is not None or info.get("lead_evidence"))
    )


def ood_check(image: Image.Image, info: dict = None, strict: bool = False, alt_image: Image.Image = None):
    """
    Return an error message if the image does not look like a transistor, else None.

    The feature-similarity test uses one geometrically normalized view. Taking the best of
    four rotations inflated unrelated images' scores. Photos that look different from the
    training photos can still pass with clear transistor body/lead geometry, with a warning.
    If `info` is given it is filled with {"similarity": float, "warning": str}.

    strict=True (live camera): score the frame as-is and reject below the stricter limit.
    `alt_image` is the same frame before auto-orientation; it is used only when orientation
    correction was applied.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # 1. Basic image quality check
    std_dev = np.std(gray)
    mean_val = np.mean(gray)
    # A very bright mean on its own is NOT out-of-domain: catalogue / product photos of a
    # transistor on a plain white backdrop are mostly white (mean > 240) yet carry plenty of
    # contrast from the part itself. Only treat an extreme mean as a failure when the image
    # is also low-contrast (blank sheet, over-exposed or black frame).
    too_flat_bright = mean_val > 240 and std_dev < 20
    if std_dev > 80 or std_dev < 5 or mean_val < 10 or too_flat_bright:
        return "Image appears to be out-of-domain (too uniform, dark, or bright). Please upload an image of a transistor."

    # 2. Advanced Anomaly Detection (Cosine Similarity)
    try:
        import torch.nn.functional as F
        (ood_model, transform), ref_emb = get_ood_model_and_ref()

        if ref_emb is not None or ood_extra_refs is not None:
            score_image = image
            geometry = None
            if strict:
                _, geometry = auto_orient(image)
            else:
                # Normalize only the OOD view. This also works when the user disabled
                # auto-orient for the defect model, without changing that model's input.
                score_image, geometry = auto_orient(image)
            img_tensor = transform(score_image).unsqueeze(0).to(DEVICE)
            views = [img_tensor]
            if strict and alt_image is not None:
                views.append(transform(alt_image).unsqueeze(0).to(DEVICE))
            batch = torch.cat(views)
            with inference_lock, torch.no_grad():
                feats = ood_model(batch)
                feats_norm = F.normalize(feats, p=2, dim=1)
                scores = []
                if ref_emb is not None:
                    scores.append(torch.sum(feats_norm * ref_emb, dim=1))
                if ood_extra_refs is not None:
                    scores.append((feats_norm @ ood_extra_refs.T).flatten())
                similarity = torch.cat(scores).max().item()

            # Usually similarity > 0.6 for training-style photos; real-world photos of genuine
            # transistors often land around 0.4-0.5, while unrelated objects score far lower.
            if info is not None:
                info["similarity"] = round(similarity, 3)
            if strict:
                live_reject = live_ood_limit()
                if info is not None:
                    info["limit"] = live_reject
                geometry_ok = has_transistor_geometry(geometry or {})
                if similarity < live_reject and not geometry_ok:
                    return (f"Object does not match a transistor (Similarity: {similarity:.2f}, "
                            f"needs {live_reject:.2f}). Please scan a transistor.")
                if similarity < max(live_reject, 0.65) and not geometry_ok:
                    return "Could not verify a transistor in this camera frame. Please scan a transistor."
                if similarity < max(live_reject, 0.65) and info is not None:
                    info["warning"] = "Transistor body and leads were detected, but this image differs from the training photos. Review the result carefully."
                return None
            reject_below, warn_below = ood_limits()
            geometry_ok = has_transistor_geometry(geometry or {})
            if similarity < reject_below and not geometry_ok:
                return f"Object does not match a transistor (Similarity: {similarity:.2f}). Please scan a transistor."
            # Borderline visual similarity alone is not enough: screenshots, packaging,
            # and other objects can share ImageNet features with the tiny reference set.
            # Require the independent transistor body/lead geometry for these cases.
            if similarity < max(warn_below, 0.65) and not geometry_ok:
                return (f"Could not verify a transistor body and leads (Similarity: {similarity:.2f}). "
                        "Please upload a clear image of a transistor.")
            if similarity < max(warn_below, 0.65) and info is not None:
                info["warning"] = (f"This photo looks quite different from the training images (similarity {similarity:.2f}) — "
                                   "a different package or background — so the verdict may be less reliable.")
        else:
            # A missing reference profile must not turn the OOD gate into fail-open.
            raise RuntimeError("OOD reference profile is unavailable")
        return None
    except Exception as e:
        if not ood_load_failed:
            traceback.print_exc()
        else:
            print(f"[Web] OOD feature check unavailable, using OpenCV fallback ({e})")
        # If the feature extractor is unavailable, specific body-and-lead geometry
        # takes precedence over generic contours (a transistor with long leads can
        # otherwise look like a cable or fork).
        _, geometry = auto_orient(image)
        if has_transistor_geometry(geometry):
            if info is not None:
                info["warning"] = "Transistor body and leads were detected; feature similarity is unavailable. Review the result carefully."
            return None

        # Fallback to OpenCV checks if the feature extractor is unavailable.
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
                    aspect_ratio = max(w / h, h / w)
                    area = w * h
                    if aspect_ratio > 2.5:
                        return "Object shape does not match a transistor (too elongated like a fork or cable). Please scan a transistor."
                    if area < (gray.shape[0] * gray.shape[1] * 0.01):
                        return "No significant object detected. Please move closer."
        # Do not silently accept an image the fallback could not verify. The transistor
        # geometry check is independent of the learned defect classifier.
        return "Could not verify a transistor in this image. Please upload a clear transistor photo and try again."


def prepare_image(image: Image.Image, use_auto_orient: bool):
    """Make sideways / upside-down transistors upright before anything else sees them."""
    if use_auto_orient:
        return auto_orient(image)
    return image, {"applied": False, "rotation_deg": 0, "leads_pointing": None, "confidence": 0.0,
                   "reason": "auto-orient turned off", "body_box": None, "lead_box": None}


def run_model(image, architecture, threshold, with_gradcam=True):
    """Run one model; returns (response_fields, raw_result). `image` must already be oriented."""
    with inference_lock:
        model, target_layer = get_or_load_model(architecture)
        status = get_model_status(architecture)
        result = predict_with_gradcam(
            model=model,
            image=image,
            target_layer=target_layer,
            device=str(DEVICE),
            image_size=CONFIG["dataset"]["image_size"],
            temperature=status.get("temperature", 1.0),
            auto_orient=False,           # already handled by prepare_image()
            with_gradcam=with_gradcam,
        )

    p_def = result["probabilities"]["defective"]
    # Determine status based on threshold slider
    is_defective = p_def >= threshold
    fields = {
        "prediction": "Defective" if is_defective else "Good",
        "confidence": round(result["confidence"], 4),
        "probabilities": {k: round(v, 4) for k, v in result["probabilities"].items()},
        "is_defective": bool(is_defective),
        # Flag low-margin predictions as uncertain
        "is_uncertain": bool(abs(p_def - threshold) < 0.10),
        "threshold": threshold,
        "architecture": architecture,
        "model_trained": status["trained"],
        "calibrated": status.get("calibrated", False),
    }
    return fields, result


def small_jpeg_b64(image: Image.Image, side: int = 96) -> str:
    thumb = image.copy()
    thumb.thumbnail((side, side))
    buf = BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# Routes
# ============================================================

def list_examples():
    examples_dir = PROJECT_ROOT / "web" / "examples"
    examples = []
    if examples_dir.exists():
        for img_path in sorted(examples_dir.glob("*")):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                examples.append(img_path.name)
    return examples


@app.route("/")
def index():
    """Landing page."""
    eval_results = get_evaluation_results()
    return render_template("index.html", results=eval_results)


@app.route("/detect")
def detect():
    """Detection / upload page."""
    default_arch = CONFIG["model"]["architecture"]
    model_status = get_model_status(default_arch)
    return render_template("detect.html", examples=list_examples(), model_status=model_status)


@app.route("/realtime")
def realtime():
    """Real-time webcam interface page."""
    default_arch = CONFIG["model"]["architecture"]
    model_status = get_model_status(default_arch)
    return render_template("realtime.html", model_status=model_status)


@app.route("/batch")
def batch_page():
    """Batch inspection + report page."""
    default_arch = CONFIG["model"]["architecture"]
    return render_template("batch.html", model_status=get_model_status(default_arch),
                           batch_max=BATCH_MAX_IMAGES)


@app.route("/compare")
def compare_page():
    """Three-model consensus page."""
    default_arch = CONFIG["model"]["architecture"]
    return render_template("compare.html", examples=list_examples(),
                           model_status=get_model_status(default_arch))


@app.route("/lab")
def lab_page():
    """Robustness lab."""
    default_arch = CONFIG["model"]["architecture"]
    return render_template("lab.html", examples=list_examples(),
                           model_status=get_model_status(default_arch))


@app.route("/line")
def line_page():
    """Production-line simulator."""
    default_arch = CONFIG["model"]["architecture"]
    return render_template("line.html", examples=list_examples(),
                           model_status=get_model_status(default_arch))


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
    Optional: 'architecture' (string), 'threshold' (float 0-1),
              'auto_orient' ("false" to disable the sideways-transistor fix),
              'gradcam' ("false" to skip the heat-map for fast screening),
              'bypass_ood' ("true").

    Returns: JSON with prediction, confidence, probabilities, orientation info,
             a suspected-defect diagnosis and base64-encoded Grad-CAM images.
    """
    try:
        image = read_single_upload("image")
        params = parse_params()
        raw_image = image
        image, orientation = prepare_image(image, params["auto_orient"])

        ood_info = {}
        if not params["bypass_ood"]:
            problem = ood_check(image, ood_info, strict=params["live"],
                                alt_image=raw_image if (params["live"] and orientation["applied"]) else None)
            if problem:
                return jsonify({"error": problem, "success": False, "orientation": orientation,
                                "ood_rejected": True, "ood_similarity": ood_info.get("similarity")}), 400

        fields, result = run_model(image, params["architecture"], params["threshold"], params["with_gradcam"])

        response = {"success": True, **fields, "orientation": orientation, "diagnosis": None,
                    "ood": {"similarity": ood_info.get("similarity"), "warning": ood_info.get("warning"),
                            "limit": ood_info.get("limit")}}
        if params["with_gradcam"]:
            response["original_image"] = image_to_base64(result["original_image"])
            response["heatmap"] = image_to_base64(result["heatmap"])
            response["overlay"] = image_to_base64(result["overlay"])
            if fields["is_defective"]:
                response["diagnosis"] = diagnose(result["original_image"], result["heatmap_raw"])
        return jsonify(response)

    except ApiError:
        raise
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


def _collect_batch_items():
    """Gather (name, bytes) pairs from multiple uploaded images and/or zip archives."""
    items = []
    for f in request.files.getlist("images"):
        name = f.filename or "image"
        data = f.read()
        if Path(name).suffix.lower() == ".zip":
            try:
                zf = zipfile.ZipFile(BytesIO(data))
            except zipfile.BadZipFile:
                items.append((name, None))
                continue
            total = 0
            for info in zf.infolist():
                base = Path(info.filename).name
                if info.is_dir() or info.filename.startswith("__MACOSX") or base.startswith("."):
                    continue
                if Path(base).suffix.lower() not in ALLOWED_IMAGE_EXTS:
                    continue
                total += info.file_size
                if info.file_size > per_file_limit_bytes() or total > 200 * 1024 * 1024:
                    items.append((base, None))
                    continue
                items.append((base, zf.read(info)))
                if len(items) > BATCH_MAX_IMAGES:
                    break
        else:
            items.append((name, data))
        if len(items) > BATCH_MAX_IMAGES:
            break
    return items


@app.route("/api/batch", methods=["POST"])
def api_batch():
    """
    Inspect many images at once (multiple files and/or a .zip).
    Returns per-image results plus a summary (yield %, defect-type Pareto, ...).
    """
    try:
        params = parse_params(default_gradcam=False)
        items = _collect_batch_items()
        if not items:
            raise ApiError("No images provided. Upload images or a .zip of images.")
        truncated = len(items) > BATCH_MAX_IMAGES
        items = items[:BATCH_MAX_IMAGES]

        rows = []
        started = time.monotonic()
        for name, data in items:
            row = {"filename": name}
            if time.monotonic() - started > BATCH_TIME_BUDGET_S:
                row.update(status="skipped", message="Time budget for one request reached — please run this image again.")
                rows.append(row)
                continue
            try:
                if data is None:
                    raise ApiError("Could not read this file (corrupt or too large).")
                image = decode_image(data, name)
                image, orientation = prepare_image(image, params["auto_orient"])
                row["rotated"] = bool(orientation["applied"])
                row["rotation_deg"] = orientation["rotation_deg"]
                row["thumb"] = small_jpeg_b64(image)

                ood_info = {}
                problem = None if params["bypass_ood"] else ood_check(image, ood_info)
                if ood_info.get("warning"):
                    row["ood_warning"] = ood_info["warning"]
                if problem:
                    row.update(status="rejected", message=problem)
                else:
                    fields, _ = run_model(image, params["architecture"], params["threshold"], with_gradcam=False)
                    row.update(status="ok", prediction=fields["prediction"],
                               p_defective=fields["probabilities"]["defective"],
                               confidence=fields["confidence"], is_uncertain=fields["is_uncertain"],
                               suspected_type=None, suspected_label=None)
                    if fields["is_defective"]:
                        _, full = run_model(image, params["architecture"], params["threshold"], with_gradcam=True)
                        diag = diagnose(full["original_image"], full["heatmap_raw"])
                        if diag:
                            row["suspected_type"] = diag["type"]
                            row["suspected_label"] = diag["label"]
            except ApiError as e:
                row.update(status="error", message=e.message)
            except Exception as e:  # one bad image must not sink the whole batch
                traceback.print_exc()
                row.update(status="error", message=str(e))
            rows.append(row)

        ok = [r for r in rows if r["status"] == "ok"]
        defective = [r for r in ok if r["prediction"] == "Defective"]
        types = {}
        for r in defective:
            key = r.get("suspected_label") or "Defect not localised"
            types[key] = types.get(key, 0) + 1
        status = get_model_status(params["architecture"])
        summary = {
            "total": len(rows),
            "analysed": len(ok),
            "good": len(ok) - len(defective),
            "defective": len(defective),
            "rejected": sum(1 for r in rows if r["status"] == "rejected"),
            "errors": sum(1 for r in rows if r["status"] == "error"),
            "skipped": sum(1 for r in rows if r["status"] == "skipped"),
            "uncertain": sum(1 for r in ok if r["is_uncertain"]),
            "rotated": sum(1 for r in rows if r.get("rotated")),
            "yield_pct": round(100.0 * (len(ok) - len(defective)) / len(ok), 1) if ok else None,
            "defect_types": dict(sorted(types.items(), key=lambda kv: kv[1], reverse=True)),
            "truncated": truncated,
            "max_images": BATCH_MAX_IMAGES,
        }
        return jsonify({
            "success": True, "results": rows, "summary": summary,
            "architecture": params["architecture"], "threshold": params["threshold"],
            "model_trained": status["trained"],
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    except ApiError:
        raise
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    Run every trained model on one image and report their consensus.
    Architectures without a trained checkpoint are listed but excluded from the vote.
    """
    try:
        image = read_single_upload("image")
        params = parse_params()
        image, orientation = prepare_image(image, params["auto_orient"])
        ood_info = {}
        if not params["bypass_ood"]:
            problem = ood_check(image, ood_info)
            if problem:
                return jsonify({"error": problem, "success": False, "orientation": orientation,
                                "ood_rejected": True, "ood_similarity": ood_info.get("similarity")}), 400

        models_out = []
        for arch_id, label in ARCHITECTURES:
            entry = {"architecture": arch_id, "label": label}
            if not arch_available(arch_id) or not get_model_status(arch_id)["trained"]:
                entry.update(status="no_checkpoint")
                models_out.append(entry)
                continue
            fields, result = run_model(image, arch_id, params["threshold"], with_gradcam=True)
            entry.update(status="ok", **fields,
                         original_image=image_to_base64(result["original_image"]),
                         overlay=image_to_base64(result["overlay"]))
            models_out.append(entry)

        voters = [m for m in models_out if m["status"] == "ok"]
        consensus = None
        if voters:
            n_def = sum(1 for m in voters if m["is_defective"])
            n_good = len(voters) - n_def
            verdict = "Defective" if n_def > n_good else "Good" if n_good > n_def else "Split decision"
            consensus = {
                "verdict": verdict,
                "votes_defective": n_def,
                "votes_good": n_good,
                "voters": len(voters),
                "unanimous": n_def == 0 or n_good == 0,
                "needs_review": not (n_def == 0 or n_good == 0) or any(m["is_uncertain"] for m in voters),
                "mean_p_defective": round(sum(m["probabilities"]["defective"] for m in voters) / len(voters), 4),
            }
        return jsonify({"success": True, "models": models_out, "consensus": consensus,
                        "orientation": orientation, "threshold": params["threshold"],
                        "ood": {"similarity": ood_info.get("similarity"), "warning": ood_info.get("warning")}})
    except ApiError:
        raise
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
        # "loaded" = the similarity check is active; "unavailable" = running on the simpler OpenCV fallback
        "ood_extractor": "loaded" if ood_model_cache is not None else ("unavailable" if ood_load_failed else "not loaded yet"),
        "ood_extra_references": 0 if ood_extra_refs is None else int(ood_extra_refs.shape[0]),
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

    # Server-side request cap (single images are limited per-file in decode_image();
    # the overall cap leaves room for batch uploads).
    app.config["MAX_CONTENT_LENGTH"] = int(CONFIG["web"].get("max_content_length", 16)) * 6 * 1024 * 1024

    # Pre-load the default model
    default_arch = CONFIG["model"]["architecture"]
    print(f"\n[Web] Pre-loading default model: {default_arch}")
    get_or_load_model(default_arch)

    # Warm up in the background so the FIRST scan isn't slow: loads the "is this a
    # transistor?" feature extractor (a weights download on a fresh container) and runs
    # one throw-away prediction so the framework has initialised its kernels.
    def _warm_up():
        try:
            get_ood_model_and_ref()
        except Exception as e:
            print(f"[Web] Warm-up: OOD extractor unavailable ({e})")
        try:
            run_model(Image.new("RGB", (256, 256), (128, 128, 128)), default_arch, 0.65, with_gradcam=True)
        except Exception as e:
            print(f"[Web] Warm-up prediction skipped ({e})")
    threading.Thread(target=_warm_up, name="warm-up", daemon=True).start()

    print("\n[Web] Server ready!")
    print(f"[Web] Access at: http://localhost:{CONFIG['web']['port']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    init_app()
    app.run(
        host=CONFIG["web"]["host"],
        # Cloud Run passes the container port in $PORT; locally the config value is used.
        port=int(os.environ.get("PORT", CONFIG["web"]["port"])),
        debug=CONFIG["web"]["debug"],
    )