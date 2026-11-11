"""
Export the facts the Dashboard shows about each trained model.

    python scripts/export_model_info.py                # all architectures with a checkpoint
    python scripts/export_model_info.py resnet50       # just one

Writes outputs/plots/model_info_<architecture>.json (parameters, best epoch, validation
loss / accuracy, checkpoint size and - when outputs/checkpoints/last_<architecture>.pth exists -
the full per-epoch training history, which powers the training curves).
Run it again after (re)training a model, then commit the JSON files with the checkpoints.
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_info import info_path, summarize_checkpoint, write_json  # noqa: E402

ARCHITECTURES = ["convnext_tiny", "efficientnet_b0", "resnet50"]


def main():
    with open(ROOT / "configs" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    ckpt_dir = ROOT / config["training"]["checkpoint_dir"]
    plots_dir = ROOT / config["evaluation"]["plots_dir"]
    wanted = sys.argv[1:] or ARCHITECTURES
    for arch in wanted:
        best = ckpt_dir / f"best_{arch}.pth"
        if not best.exists():
            print(f"[skip] {arch}: {best} not found")
            continue
        info = summarize_checkpoint(best, arch, max_epochs=config["training"]["epochs"])
        write_json(info_path(plots_dir, arch), info)
        extra = f", {info['epochs_run']} epochs run" if "epochs_run" in info else " (no last_ checkpoint: training history unavailable)"
        print(f"[ok] {arch}: best epoch {info['checkpoint_epoch']}, val acc {info['val_acc']}, "
              f"{info['parameters'] / 1e6:.2f}M parameters{extra}")


if __name__ == "__main__":
    main()
