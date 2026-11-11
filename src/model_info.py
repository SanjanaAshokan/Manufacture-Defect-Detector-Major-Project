"""
Model facts for the dashboard
=============================
Reads what a trained checkpoint can tell us about itself - parameter count, the epoch it was
saved at, its validation loss / accuracy and (when ``last_<arch>.pth`` is next to it) the
per-epoch training history - and writes / reads it as small JSON files in ``outputs/plots/``.

The website reads those JSON files, so it does not have to open 50 MB checkpoints on every
visit and keeps working on a server that only has the JSON. Generate / refresh them with:

    python scripts/export_model_info.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# Buffers of BatchNorm layers: stored in the state dict but they are not learnable parameters.
_BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def count_parameters(state_dict: Dict) -> int:
    """Learnable parameters in a state dict (BatchNorm running statistics excluded)."""
    return int(sum(v.numel() for k, v in state_dict.items() if not k.endswith(_BUFFER_SUFFIXES)))


def _round_history(history: Dict) -> Dict:
    return {k: [round(float(x), 5) for x in v] for k, v in history.items() if isinstance(v, (list, tuple))}


def history_matches_checkpoint(history: Dict, best_epoch, best_val_loss, tol: float = 1e-3) -> bool:
    """
    True only when a training history can belong to the run that produced the best checkpoint:
    it must cover at least ``best_epoch`` epochs, and its validation loss at that epoch must equal
    the loss stored in the checkpoint. A ``last_<arch>.pth`` left over from a different (shorter or
    later) run fails this, so its epoch count is never shown next to the wrong checkpoint.
    """
    try:
        losses = history["val_loss"]
        if not best_epoch or len(losses) < int(best_epoch):
            return False
        if best_val_loss is None:
            return True
        return abs(float(losses[int(best_epoch) - 1]) - float(best_val_loss)) <= tol
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def epochs_run_is_plausible(epochs_run, best_epoch, max_epochs) -> bool:
    """The saved epoch can never be later than the last epoch run, nor beyond the epoch budget."""
    try:
        return int(best_epoch) <= int(epochs_run) <= int(max_epochs)
    except (TypeError, ValueError):
        return False


def summarize_checkpoint(best_path: Path, architecture: str, max_epochs: Optional[int] = None) -> Dict:
    """Facts from ``best_<arch>.pth`` (and ``last_<arch>.pth`` for the training history, if present)."""
    import torch  # imported lazily so the JSON can be read without torch

    best_path = Path(best_path)
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    metrics = ckpt.get("metrics", {}) or {}
    info = {
        "architecture": architecture,
        "checkpoint_file": best_path.name,
        "checkpoint_epoch": ckpt.get("epoch"),
        "val_acc": metrics.get("val_acc"),
        "val_loss": metrics.get("val_loss"),
        "parameters": count_parameters(ckpt["model_state_dict"]),
        "checkpoint_size_mb": round(best_path.stat().st_size / 1e6, 1),
        "weights_dtype": str(next(iter(ckpt["model_state_dict"].values())).dtype).replace("torch.", ""),
        "saved_at": ckpt.get("timestamp"),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    last_path = best_path.with_name(best_path.name.replace("best_", "last_", 1))
    if last_path.exists():
        try:
            last = torch.load(last_path, map_location="cpu", weights_only=False)
            history = last.get("history")
            if history and history.get("val_loss") and not history_matches_checkpoint(
                    history, info["checkpoint_epoch"], info["val_loss"]):
                # last_<arch>.pth is from another training run than best_<arch>.pth
                info["history_mismatch"] = (
                    f"last checkpoint has {len(history['val_loss'])} epochs but the best checkpoint "
                    f"is from epoch {info['checkpoint_epoch']}; they are from different runs, so the "
                    f"history was ignored")
            elif history and history.get("val_loss"):
                info["history"] = _round_history(history)
                info["epochs_run"] = len(history["val_loss"])
                if max_epochs:
                    info["max_epochs"] = int(max_epochs)
                    info["early_stopped"] = info["epochs_run"] < int(max_epochs)
        except Exception as exc:  # a damaged last checkpoint must not stop the export
            info["history_error"] = str(exc)
    return info


def info_path(plots_dir: Path, architecture: str) -> Path:
    return Path(plots_dir) / f"model_info_{architecture}.json"


def read_json(path: Path) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def training_recipe(config: Dict) -> list:
    """
    Human-readable training recipe as a list of [label, value] pairs, derived from the config
    the same way src/train.py uses it. Stored in training_<arch>.json for every new run; the
    dashboard falls back to the project defaults when a run did not record it.
    """
    t, m, d = config["training"], config["model"], config["dataset"]
    lr = float(t["learning_rate"])
    es = t.get("early_stopping", {}) or {}
    if t.get("use_weighted_sampler", False):
        loss = "Cross-entropy with a class-balancing sampler"
    else:
        loss = "Focal loss (gamma 2, class weights 1 : 7)"
    return [
        ["Optimiser", f"AdamW, weight decay {t['weight_decay']:g}"],
        ["Learning rate", f"{lr:g} (classifier head), {lr * 0.1:g} (backbone)"],
        ["LR schedule", "OneCycle, cosine annealing, 10% warm-up"],
        ["Batch size", str(t["batch_size"])],
        ["Loss", loss],
        ["Classifier head", f"dropout {m['dropout']:g}, hidden layer {m['hidden_dim']}"],
        ["Input size", f"{d['image_size']} x {d['image_size']} px"],
        ["Pretrained", "ImageNet weights" if m.get("pretrained", True) else "None (from scratch)"],
        ["Early stopping", f"patience {es['patience']} epochs on validation loss" if es.get("enabled") else "off"],
        ["Best-model rule", "Lowest validation loss"],
        ["Mixed precision", "on" if t.get("use_amp") else "off"],
        ["Random seed", str(t["seed"])],
    ]
