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
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from PIL import Image, ImageOps

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import get_model
from src.gradcam import predict_with_gradcam, image_to_base64
from src.utils import load_checkpoint, get_device
from src.orientation import auto_orient, centre_crop, find_transistor_geometry
from src.diagnosis import diagnose
from src.model_info import (epochs_run_is_plausible, history_matches_checkpoint, info_path, read_json,
                            summarize_checkpoint, training_recipe)

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
# Dashboard data: comparison of all models + one page per model
# ============================================================

# URL slug -> architecture id (/dashboard/<slug>)
MODEL_PAGES = {
    "convnext-tiny": "convnext_tiny",
    "efficientnet-b0": "efficientnet_b0",
    "resnet-50": "resnet50",
}
ARCH_TO_SLUG = {arch: slug for slug, arch in MODEL_PAGES.items()}

# Static facts about each architecture (what it is, where it comes from, reference numbers
# from the torchvision model zoo). Everything about OUR trained models - epochs, accuracy,
# parameters - is read from the checkpoints / evaluation files instead, never hard-coded.
ARCH_DETAILS = {
    "convnext_tiny": {
        "label": "ConvNeXt-Tiny",
        "tagline": "A modernised pure-CNN that borrows Transformer-era design",
        "family": "Modern ConvNet",
        "origin": "Liu et al., Meta AI and UC Berkeley - \"A ConvNet for the 2020s\" (CVPR 2022)",
        "pretrained": "ImageNet-1K (torchvision IMAGENET1K_V1)",
        "imagenet_top1": 82.5,
        "gflops": 4.46,
        "gradcam_layer": "Last ConvNeXt block of stage 4",
        "head": "LayerNorm, Dropout 0.3, Linear 768 to 256, GELU, Dropout 0.2, Linear 256 to 2",
        "summary": ("ConvNeXt-Tiny re-designs a classic ResNet with the ideas that made Vision Transformers "
                    "successful - a patchify stem, large 7x7 depth-wise convolutions, inverted-bottleneck blocks, "
                    "LayerNorm and GELU - while staying a purely convolutional network."),
        "how": ["A patchify stem cuts the image into 4x4 patches, like a Vision Transformer, instead of a heavy 7x7 stem.",
                "Four stages (3-3-9-3 blocks) use 7x7 depth-wise convolutions followed by an inverted-bottleneck MLP, with LayerNorm.",
                "The 1000-class ImageNet classifier is replaced by a small 2-class head and the whole network is fine-tuned."],
        "strengths": ["Highest ImageNet accuracy of the three, so it starts from the strongest pretrained features",
                      "Large 7x7 kernels see more of the lead-and-body context in a single layer",
                      "LayerNorm does not depend on batch size, which keeps fine-tuning on a small dataset stable"],
        "tradeoffs": ["About 28M parameters and 4.5 GFLOPs per image - roughly ten times the compute of EfficientNet-B0",
                      "Needs the most memory, so it is the least suited to small edge devices"],
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0",
        "tagline": "Tiny, efficient and fast - built by compound scaling",
        "family": "Mobile ConvNet (NAS)",
        "origin": "Tan and Le, Google Brain - \"EfficientNet: Rethinking Model Scaling for CNNs\" (ICML 2019)",
        "pretrained": "ImageNet-1K (torchvision IMAGENET1K_V1)",
        "imagenet_top1": 77.7,
        "gflops": 0.39,
        "gradcam_layer": "Last convolution of the final feature block",
        "head": "Dropout 0.3, Linear 1280 to 256, BatchNorm, GELU, Dropout 0.2, Linear 256 to 2",
        "summary": ("EfficientNet-B0 is the baseline of a family found by neural architecture search and scaled with one "
                    "compound coefficient (depth, width and resolution together). It is built from mobile inverted-bottleneck "
                    "(MBConv) blocks with squeeze-and-excitation attention."),
        "how": ["MBConv blocks expand the channels, apply a depth-wise convolution, then squeeze them back down.",
                "Squeeze-and-excitation re-weights every channel by how useful it is for the whole image.",
                "Compound scaling keeps network depth, width and input resolution in balance."],
        "strengths": ["Smallest model by far: about 4.3M parameters and 0.39 GFLOPs",
                      "Fastest to run, which suits the live camera, CPU-only servers and edge devices",
                      "Smallest checkpoint, so it is the cheapest to store and deploy"],
        "tradeoffs": ["Lowest ImageNet accuracy of the three, so there is less pretrained knowledge to start from",
                      "Depth-wise convolutions and BatchNorm make it more sensitive to learning-rate and batch-size choices"],
    },
    "resnet50": {
        "label": "ResNet-50",
        "tagline": "The classic residual network - a dependable baseline",
        "family": "Residual ConvNet",
        "origin": "He et al., Microsoft Research - \"Deep Residual Learning for Image Recognition\" (CVPR 2016)",
        "pretrained": "ImageNet-1K (torchvision IMAGENET1K_V2, improved training recipe)",
        "imagenet_top1": 80.9,
        "gflops": 4.09,
        "gradcam_layer": "Last bottleneck block of layer4",
        "head": "Dropout 0.3, Linear 2048 to 256, BatchNorm, GELU, Dropout 0.2, Linear 256 to 2",
        "summary": ("ResNet-50 introduced skip (residual) connections, which let very deep networks train reliably. "
                    "It stacks 16 bottleneck blocks (1x1, 3x3, 1x1 convolutions) in four stages."),
        "how": ["Every block adds its input back to its output (the residual connection), so gradients flow easily.",
                "Bottleneck blocks reduce, process and restore the channels to keep the compute manageable.",
                "Global average pooling feeds the new 2-class head."],
        "strengths": ["Mature, well-understood and supported by every tool, including Grad-CAM",
                      "The V2 pretrained recipe is strong: 80.9% ImageNet top-1",
                      "Simple, robust architecture with predictable fine-tuning behaviour"],
        "tradeoffs": ["About 24M parameters and 4.1 GFLOPs - much heavier than EfficientNet-B0",
                      "An older BatchNorm / ReLU-era design: less accurate than ConvNeXt at a similar cost"],
    },
}

# Chart / accent colour per model (matches the site palette)
ARCH_COLOURS = {"convnext_tiny": "#00d4ff", "efficientnet_b0": "#a78bfa", "resnet50": "#10b981"}

_checkpoint_facts_cache = {}


def _plots_path() -> Path:
    return PROJECT_ROOT / CONFIG["evaluation"]["plots_dir"]


def load_arch_results(architecture: str):
    """
    Test-set evaluation of ONE architecture.
    Prefers evaluation_results_<arch>.json; falls back to the older single evaluation_results.json
    when that file belongs to this architecture. Returns (results | None, plots_are_legacy).
    """
    per_model = read_json(_plots_path() / f"evaluation_results_{architecture}.json")
    if per_model:
        return per_model, False
    legacy = get_evaluation_results()
    if legacy and legacy.get("architecture") == architecture:
        return legacy, True
    return None, False


def arch_plot_urls(architecture: str, legacy: bool) -> dict:
    urls = {}
    plots = _plots_path()
    for stem in ("confusion_matrix", "pr_curve", "roc_curve", "training_history"):
        if (plots / f"{stem}_{architecture}.png").exists():
            urls[stem] = f"/plots/{stem}_{architecture}.png"
        elif legacy and (plots / f"{stem}.png").exists():
            urls[stem] = f"/plots/{stem}.png"
    return urls


def load_checkpoint_facts(architecture: str) -> dict:
    """model_info_<arch>.json when exported; otherwise read from the checkpoint once and cache it."""
    info = read_json(info_path(_plots_path(), architecture))
    if info:
        return info
    if architecture not in _checkpoint_facts_cache:
        facts = {}
        ckpt = checkpoint_path_for(architecture)
        if ckpt.exists():
            try:
                with inference_lock:
                    facts = summarize_checkpoint(ckpt, architecture, CONFIG["training"]["epochs"])
            except Exception:
                traceback.print_exc()
        _checkpoint_facts_cache[architecture] = facts
    return _checkpoint_facts_cache[architecture]


def _ece_of(results, calibration):
    """Test-set expected calibration error, only when it was produced by the current formula."""
    cal = (results or {}).get("calibration") or {}
    return cal.get("ece") if cal.get("version") == 2 else None


def build_model_view(architecture: str) -> dict:
    """Everything the Dashboard shows about one model, gathered from files (never hard-coded)."""
    det = ARCH_DETAILS[architecture]
    plots = _plots_path()
    facts = load_checkpoint_facts(architecture)
    training = read_json(plots / f"training_{architecture}.json") or {}
    results, legacy = load_arch_results(architecture)
    calibration = read_json(plots / f"calibration_{architecture}.json")
    metrics = (results or {}).get("metrics") or {}

    history = training.get("history") or facts.get("history")
    epochs_run = training.get("epochs_run") or facts.get("epochs_run") or (len(history["val_loss"]) if history else None)
    max_epochs = training.get("max_epochs") or facts.get("max_epochs") or CONFIG["training"]["epochs"]
    best_epoch = training.get("best_epoch") or facts.get("checkpoint_epoch")
    early_stopped = training.get("early_stopped")

    # The run record must agree with the saved best checkpoint. A last_<arch>.pth / history from a
    # different run (e.g. 8 epochs run vs a best checkpoint at epoch 17) is ignored, never shown.
    best_val_loss = facts.get("val_loss") if facts.get("val_loss") is not None else training.get("best_val_loss")
    history_mismatch = bool(facts.get("history_mismatch"))
    if history and not history_matches_checkpoint(history, best_epoch, best_val_loss):
        history, history_mismatch = None, True
        epochs_run = training.get("epochs_run") if training.get("history") is None else None
    if epochs_run and best_epoch and not epochs_run_is_plausible(epochs_run, best_epoch, max_epochs):
        epochs_run, history_mismatch = None, True
    if history_mismatch and epochs_run is None:
        early_stopped = None
    if epochs_run:
        history_mismatch = False   # a valid run record exists, so there is nothing left to warn about
    if early_stopped is None and epochs_run and max_epochs:
        early_stopped = epochs_run < max_epochs
    if early_stopped is None:
        stop_reason = None
    else:
        stop_reason = "Early stopping" if early_stopped else "Full epoch budget"
    if training.get("stop_reason") and epochs_run:      # imported from a log: wording limited to what is known
        stop_reason = training["stop_reason"]

    recipe = training.get("recipe")
    recipe_is_run = bool(recipe)
    if not recipe:
        recipe = training_recipe(CONFIG)
    if training.get("best_rule") and epochs_run:        # observed in the real log, unlike the defaults
        recipe = [[k, f"{training['best_rule']} (from the training log)" if k == "Best-model rule" else v] for k, v in recipe]

    params = facts.get("parameters")
    view = {
        "arch": architecture,
        "slug": ARCH_TO_SLUG[architecture],
        "label": det["label"],
        "colour": ARCH_COLOURS[architecture],
        "details": det,
        "is_default": architecture == CONFIG["model"]["architecture"],
        "checkpoint_available": arch_available(architecture),
        "has_checkpoint_facts": bool(facts),
        "has_results": bool(results),
        "results": results,
        "plots": arch_plot_urls(architecture, legacy) if results else {},
        "calibration": calibration,
        "history": history,
        "history_mismatch": history_mismatch,
        "recipe": recipe,
        "recipe_is_run": recipe_is_run,
        "training": training,
        "vals": {
            "parameters_m": (params / 1e6) if params else None,
            "size_mb": facts.get("checkpoint_size_mb"),
            "gflops": det["gflops"],
            "imagenet_top1": det["imagenet_top1"],
            "max_epochs": max_epochs,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "stop_reason": stop_reason,
            "train_minutes": (training.get("total_seconds") / 60.0) if training.get("total_seconds") else None,
            "val_acc": facts.get("val_acc"),
            "val_loss": facts.get("val_loss"),
            "test_samples": (results or {}).get("num_test_samples"),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "avg_precision": metrics.get("avg_precision"),
            "ece": _ece_of(results, calibration),
            "ms_per_image": (results or {}).get("ms_per_image"),
        },
        "weights_dtype": facts.get("weights_dtype"),
        "checkpoint_file": facts.get("checkpoint_file"),
    }
    return view


def _fmt(kind: str, value) -> str:
    if value is None:
        return "\u2014"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "f2":
        return f"{value:.2f}"
    if kind == "f3":
        return f"{value:.3f}"
    if kind == "int":
        return f"{int(value)}"
    if kind == "m":
        return f"{value:.2f} M"
    if kind == "mb":
        return f"{value:.1f} MB"
    if kind == "ms":
        return f"{value:.1f} ms"
    if kind == "min":
        return f"{value:.1f} min"
    return str(value)


# (section, [(label, hint, key, format, better)]) - better: "max" / "min" / None (no ranking)
COMPARE_SECTIONS = [
    ("Model", [
        ("Parameters", "Learnable weights", "parameters_m", "m", "min"),
        ("Checkpoint size", "Saved weights on disk", "size_mb", "mb", "min"),
        ("Compute", "GFLOPs per 224 x 224 image", "gflops", "f2", "min"),
        ("ImageNet top-1", "Pretrained backbone, reference", "imagenet_top1", "f2", "max"),
    ]),
    ("Training", [
        ("Epoch budget", "Maximum epochs allowed", "max_epochs", "int", None),
        ("Best epoch", "Epoch of the saved checkpoint", "best_epoch", "int", None),
        ("Epochs run", "Epochs actually trained", "epochs_run", "int", None),
        ("Training stopped by", "", "stop_reason", "text", None),
    ]),
    ("Validation (best epoch)", [
        ("Accuracy", "Held-out validation images", "val_acc", "pct", "max"),
        ("Loss", "Lower is better", "val_loss", "f3", "min"),
    ]),
    ("Test set", [
        ("Test images", "", "test_samples", "int", None),
        ("Accuracy", "Overall correct predictions", "accuracy", "pct", "max"),
        ("Precision", "Flagged defects that are real", "precision", "pct", "max"),
        ("Recall", "Real defects that were caught", "recall", "pct", "max"),
        ("F1-score", "Balance of precision and recall", "f1_score", "pct", "max"),
        ("ROC AUC", "Ranking quality at every threshold", "roc_auc", "pct", "max"),
        ("Average precision", "Area under the PR curve", "avg_precision", "pct", "max"),
        ("Calibration error", "ECE, lower is better", "ece", "f3", "min"),
        ("Speed", "ms per test image on the evaluating machine", "ms_per_image", "ms", "min"),
    ]),
]


def build_comparison(views: list) -> dict:
    """Rows for the side-by-side table (best value per row flagged) plus chart data."""
    sections = []
    for title, rows in COMPARE_SECTIONS:
        out_rows = []
        for label, hint, key, kind, better in rows:
            vals = [v["vals"].get(key) for v in views]
            present = [x for x in vals if isinstance(x, (int, float))]
            best_value = None
            if better and len(present) >= 2:
                best_value = max(present) if better == "max" else min(present)
            cells = []
            for x in vals:
                is_best = best_value is not None and isinstance(x, (int, float)) and x == best_value
                if best_value is not None and len(set(present)) == 1:
                    is_best = False   # a tie across every model is not a win
                cells.append({"text": _fmt(kind, x), "best": is_best, "missing": x is None})
            out_rows.append({"label": label, "hint": hint, "cells": cells})
        sections.append({"title": title, "rows": out_rows})

    # Which models still lack a held-out test evaluation
    unevaluated = [v["label"] for v in views if not v["has_results"]]

    # Recall per defect type
    defect_types = sorted({d for v in views if v["results"] for d in (v["results"].get("defect_type_breakdown") or {})})
    defect_rows = []
    for d in defect_types:
        cells = []
        for v in views:
            stats = ((v["results"] or {}).get("defect_type_breakdown") or {}).get(d)
            cells.append({"text": f"{stats['recall'] * 100:.0f}%" if stats else "\u2014",
                          "detail": f"{stats['correct']}/{stats['support']}" if stats else "",
                          "missing": stats is None})
        defect_rows.append({"label": d.replace("_", " "), "cells": cells})

    # F1 per decision threshold
    thresholds = sorted({row["threshold"] for v in views if v["results"] for row in (v["results"].get("threshold_analysis") or [])})
    threshold_rows = []
    for th in thresholds:
        cells = []
        for v in views:
            hit = next((r for r in ((v["results"] or {}).get("threshold_analysis") or []) if r["threshold"] == th), None)
            cells.append({"text": f"{hit['f1_score'] * 100:.1f}%" if hit else "\u2014", "missing": hit is None})
        threshold_rows.append({"threshold": th, "cells": cells})

    chart = {
        "models": [{"id": v["arch"], "label": v["label"], "colour": v["colour"]} for v in views],
        "metrics": [
            {"key": key, "label": label, "values": [v["vals"].get(key) for v in views]}
            for key, label in (("val_acc", "Validation accuracy"), ("accuracy", "Test accuracy"),
                               ("precision", "Precision"), ("recall", "Recall"),
                               ("f1_score", "F1-score"), ("roc_auc", "ROC AUC"))
        ],
        "curves": [
            {"id": v["arch"], "label": v["label"], "colour": v["colour"], "history": v["history"],
             "best_epoch": v["vals"]["best_epoch"]}
            for v in views if v["history"]
        ],
    }
    return {"sections": sections, "unevaluated": unevaluated, "defect_rows": defect_rows,
            "threshold_rows": threshold_rows, "chart": chart}


def assign_badges(views: list) -> None:
    """Add short highlight badges (best validation accuracy, smallest, ...) to each model view."""
    for v in views:
        v["badges"] = []
    def mark(key, text, best):
        vals = [(v, v["vals"].get(key)) for v in views if v["vals"].get(key) is not None]
        if len(vals) < 2 or len({x for _, x in vals}) == 1:
            return
        target = (max if best == "max" else min)(x for _, x in vals)
        for v, x in vals:
            if x == target:
                v["badges"].append(text)
    mark("val_acc", "Best validation accuracy", "max")
    mark("f1_score", "Best test F1", "max")
    mark("recall", "Best defect recall", "max")
    mark("parameters_m", "Smallest model", "min")
    mark("gflops", "Lowest compute", "min")
    # Every card gets some "small info": a model that wins no category (e.g. ResNet-50) shows neutral
    # facts about itself instead of an empty row.
    for v in views:
        if not v["badges"]:
            det = v["details"]
            v["neutral_badges"] = [det["family"], f"{det['imagenet_top1']}% ImageNet top-1"]
        else:
            v["neutral_badges"] = []


def rank_summary(views: list, view: dict) -> list:
    """For the per-model page: where this model stands among the three on the key metrics."""
    out = []
    for key, label, kind, better in (("val_acc", "Validation accuracy", "pct", "max"), ("accuracy", "Test accuracy", "pct", "max"),
                                     ("recall", "Defect recall", "pct", "max"), ("f1_score", "F1-score", "pct", "max"),
                                     ("parameters_m", "Parameters", "m", "min"), ("gflops", "Compute (GFLOPs)", "f2", "min")):
        mine = view["vals"].get(key)
        peers = [v["vals"].get(key) for v in views if v["vals"].get(key) is not None]
        if mine is None or len(peers) < 2:
            out.append({"label": label, "text": _fmt(kind, mine), "rank": None, "of": len(peers)})
            continue
        ordered = sorted(peers, reverse=(better == "max"))
        out.append({"label": label, "text": _fmt(kind, mine), "rank": ordered.index(mine) + 1, "of": len(peers)})
    return out


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


def assisted_min_similarity() -> float:
    """
    Minimum feature similarity when the "it has a body and leads" evidence was only found
    thanks to a relaxation (stock-photo bars trimmed, gradient backdrop, live centre crop).
    Ordinary detections are untouched. Tune with OOD_ASSISTED_MIN_SIMILARITY or
    `ood_assisted_min_similarity` under `web:` in configs/config.yaml.
    """
    web = CONFIG.get("web", {})
    return float(os.environ.get("OOD_ASSISTED_MIN_SIMILARITY", web.get("ood_assisted_min_similarity", 0.20)))


def _ood_similarity(image: Image.Image):
    """Best cosine similarity of one image to the reference photos (None if unavailable)."""
    import torch.nn.functional as F
    try:
        (ood_model, transform), ref_emb = get_ood_model_and_ref()
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        with inference_lock, torch.no_grad():
            feats = F.normalize(ood_model(tensor), p=2, dim=1)
            scores = []
            if ref_emb is not None:
                scores.append(torch.sum(feats * ref_emb, dim=1))
            if ood_extra_refs is not None:
                scores.append((feats @ ood_extra_refs.T).flatten())
            return torch.cat(scores).max().item() if scores else None
    except Exception:
        return None


def resolve_geometry(image: Image.Image, geometry: dict, similarity: float, live: bool, info: dict = None) -> bool:
    """
    Decide whether there is independent evidence of a transistor body + leads.

    * Standard evidence is used as before.
    * Evidence that was only found after a relaxation (bars trimmed, gradient backdrop) must
      also reach a small similarity floor, so relaxations cannot let anything through.
    * Live camera only: if nothing was found in the whole frame, look again in the central
      crop (the guide box) - a busy desk or hand around the part hides its outline.
    """
    geometry = geometry or {}
    had_geometry = has_transistor_geometry(geometry)
    ok, method = had_geometry, ("standard" if had_geometry else None)
    floor = assisted_min_similarity()
    if had_geometry and geometry.get("assisted"):
        if similarity < floor:
            ok, method = False, None
        else:
            method = " + ".join(geometry["assisted"])
    if not ok and live and not had_geometry:
        found = find_transistor_geometry(image)
        if found["ok"]:
            crop, _, _ = centre_crop(image, 0.6)
            crop_sim = _ood_similarity(crop)
            if max(similarity, crop_sim if crop_sim is not None else 0.0) >= floor:
                ok, method = True, found["method"]
    if info is not None:
        info["geometry"] = {"ok": ok, "method": method}
    if os.environ.get("OOD_DEBUG"):
        print(f"[OOD] similarity={similarity:.3f} geometry_ok={ok} via={method} live={live}")
    return ok


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
                geometry_ok = resolve_geometry(image, geometry, similarity, strict, info)
                if similarity < live_reject and not geometry_ok:
                    return (f"Object does not match a transistor (Similarity: {similarity:.2f}, "
                            f"needs {live_reject:.2f}). Please scan a transistor.")
                if similarity < max(live_reject, 0.65) and not geometry_ok:
                    return "Could not verify a transistor in this camera frame. Please scan a transistor."
                if similarity < max(live_reject, 0.65) and info is not None:
                    info["warning"] = "Transistor body and leads were detected, but this image differs from the training photos. Review the result carefully."
                return None
            reject_below, warn_below = ood_limits()
            geometry_ok = resolve_geometry(image, geometry, similarity, strict, info)
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
    """Dashboard home: all three models compared side by side."""
    views = [build_model_view(arch) for arch, _ in ARCHITECTURES]
    assign_badges(views)
    comparison = build_comparison(views)
    return render_template("dashboard.html", views=views, comparison=comparison,
                           chart_data=comparison["chart"])


@app.route("/dashboard/<slug>")
def dashboard_model(slug):
    """Dashboard sub-page: everything about one model."""
    arch = MODEL_PAGES.get(slug)
    if arch is None:
        abort(404)
    views = [build_model_view(a) for a, _ in ARCHITECTURES]
    assign_badges(views)
    view = next(v for v in views if v["arch"] == arch)
    position = views.index(view)
    return render_template("dashboard_model.html", view=view, views=views,
                           ranks=rank_summary(views, view), comparison=build_comparison(views),
                           prev_view=views[position - 1] if position > 0 else None,
                           next_view=views[position + 1] if position < len(views) - 1 else None)


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
                            "limit": ood_info.get("limit"), "geometry": ood_info.get("geometry")}}
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