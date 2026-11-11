"""
Fill a model's "Epochs run" and Training-curve chart from its REAL training log.

Every training run writes one log file (outputs/logs/defect_detector_<time>.log) with a line per epoch:
    Epoch 17/30 | Train Loss: 0.4 | Train Acc: 0.9 | Val Loss: 0.5165 | Val Acc: 0.92 | LR: 0.0007
Pasted console output of a run works too (it contains the same lines).

This script finds, among all runs in the log(s), the one that produced the saved best checkpoint - the run
whose best epoch and validation loss equal the checkpoint's - and writes it to
outputs/plots/training_<architecture>.json, which the Dashboard reads. Nothing is guessed: if no run matches,
nothing is written. It only READS checkpoints and logs; no weights are touched.

    python scripts/import_training_log.py                              # all models, logs in outputs/logs
    python scripts/import_training_log.py convnext_tiny                # one model
    python scripts/import_training_log.py convnext_tiny my_console.txt # log / pasted console file or folder
    python scripts/import_training_log.py convnext_tiny --epochs-run 24   # only the count, no curves
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_info import epochs_run_is_plausible, info_path, read_json, summarize_checkpoint, write_json  # noqa: E402

ARCHITECTURES = ["convnext_tiny", "efficientnet_b0", "resnet50"]
LINE = re.compile(
    r"Epoch\s+(\d+)\s*/\s*(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)\s*\|\s*"
    r"Val Loss:\s*([\d.]+)\s*\|\s*Val Acc:\s*([\d.]+)\s*\|\s*LR:\s*([\d.eE+-]+)")


def parse_runs(text: str) -> list:
    """Split a log into training runs (a run starts whenever the epoch counter goes back to 1)."""
    runs, current, seen = [], [], set()
    for m in LINE.finditer(text):
        epoch = int(m.group(1))
        if epoch == 1 and current:
            runs.append(current)
            current, seen = [], set()
        if epoch in seen:          # the same line printed twice (console + file)
            continue
        seen.add(epoch)
        current.append({"epoch": epoch, "max": int(m.group(2)), "train_loss": float(m.group(3)),
                        "train_acc": float(m.group(4)), "val_loss": float(m.group(5)),
                        "val_acc": float(m.group(6)), "lr": float(m.group(7))})
    if current:
        runs.append(current)
    return runs


def run_matches(run: list, best_epoch: int, best_val_loss: float, best_val_acc=None) -> bool:
    """
    The run's row at the checkpoint's epoch carries the checkpoint's validation loss (and accuracy).
    It does NOT have to be the lowest-loss epoch: older versions of train.py chose the best checkpoint by
    validation accuracy, so the saved epoch can have a higher loss than another epoch of the same run.
    """
    if len(run) < best_epoch or [r["epoch"] for r in run] != list(range(1, len(run) + 1)):
        return False
    row = run[best_epoch - 1]
    if abs(row["val_loss"] - best_val_loss) > 2e-4:
        return False
    return best_val_acc is None or abs(row["val_acc"] - best_val_acc) <= 1e-3


def best_rule_of(run: list, best_epoch: int):
    """Which selection rule explains the saved epoch in this run (None if neither does)."""
    losses, accs = [r["val_loss"] for r in run], [r["val_acc"] for r in run]
    if losses.index(min(losses)) + 1 == best_epoch:
        return "Lowest validation loss"
    if accs.index(max(accs)) + 1 == best_epoch:
        return "Highest validation accuracy"
    return None


def summarize_run(run: list) -> dict:
    losses = [r["val_loss"] for r in run]
    best = losses.index(min(losses))
    return {"epochs": len(run), "max": run[0]["max"], "best_epoch": run[best]["epoch"], "min_val_loss": losses[best]}


def near_matches(runs: list, best_epoch: int, best_loss: float):
    """
    Runs whose lowest validation loss equals the checkpoint's, but whose best epoch is not the stored one.
    Returns (offset_by_one, others). Older checkpoints may have stored the epoch counted from 0, so a run
    whose best epoch is exactly one higher is accepted as the same run.
    """
    offset, other = [], []
    for f, r in runs:
        info = summarize_run(r)
        if [x["epoch"] for x in r] != list(range(1, len(r) + 1)) or abs(info["min_val_loss"] - best_loss) > 2e-4:
            continue
        (offset if info["best_epoch"] == best_epoch + 1 else other).append((f, r))
    return offset, other


def collect_sources(paths: list, logs_dir: Path) -> list:
    files = []
    for p in (paths or [logs_dir]):
        p = Path(p)
        files += sorted(p.glob("*.log")) + sorted(p.glob("*.txt")) if p.is_dir() else [p]
    return [f for f in files if f.exists()]


def main():
    args = sys.argv[1:]
    epochs_only = None
    if "--epochs-run" in args:
        i = args.index("--epochs-run")
        epochs_only = int(args[i + 1])
        del args[i:i + 2]
    wanted = [args.pop(0)] if args and args[0] in ARCHITECTURES else ARCHITECTURES
    with open(ROOT / "configs" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    ckpt_dir, plots_dir = ROOT / config["training"]["checkpoint_dir"], ROOT / config["evaluation"]["plots_dir"]
    files = collect_sources(args, ROOT / config["training"]["log_dir"])
    runs = []   # (file, run) newest file last
    for f in files:
        for r in parse_runs(f.read_text(encoding="utf-8", errors="ignore")):
            runs.append((f, r))
    print(f"Read {len(files)} log file(s), found {len(runs)} training run(s).")

    for arch in wanted:
        best = ckpt_dir / f"best_{arch}.pth"
        if not best.exists():
            print(f"[skip] {arch}: no checkpoint")
            continue
        facts = read_json(info_path(plots_dir, arch)) or summarize_checkpoint(best, arch, config["training"]["epochs"])
        best_epoch, best_loss = facts["checkpoint_epoch"], facts["val_loss"]
        max_epochs = config["training"]["epochs"]

        if epochs_only is not None:
            if len(wanted) != 1:
                sys.exit("--epochs-run needs one architecture, e.g. convnext_tiny --epochs-run 24")
            if not epochs_run_is_plausible(epochs_only, best_epoch, max_epochs):
                sys.exit(f"[error] {arch}: {epochs_only} epochs is impossible - the saved checkpoint is from epoch "
                         f"{best_epoch} and the budget is {max_epochs}.")
            write_json(plots_dir / f"training_{arch}.json", {
                "architecture": arch, "max_epochs": max_epochs, "epochs_run": epochs_only, "best_epoch": best_epoch,
                "early_stopped": epochs_only < max_epochs, "best_val_loss": best_loss, "best_val_acc": facts["val_acc"],
                "source": "entered by hand (no curves)"})
            print(f"[ok] {arch}: recorded {epochs_only} epochs run (no history, so no curves)")
            continue

        hits = [(f, r) for f, r in runs if run_matches(r, best_epoch, best_loss, facts.get("val_acc"))]
        log_best = best_epoch
        if not hits:
            offset, other = near_matches(runs, best_epoch, best_loss)
            if offset:
                hits, log_best = offset, best_epoch + 1
                print(f"[note] {arch}: the log counts the best epoch as {log_best}, the checkpoint stores {best_epoch} "
                      f"(older checkpoints counted epochs from 0). Using the log's numbering.")
        if not hits:
            print(f"[no match] {arch}: no logged run has best epoch {best_epoch} with validation loss {best_loss:.4f}.")
            if runs:
                print("  Closest logged runs (file | epochs run | best epoch | lowest validation loss):")
                ranked = sorted(runs, key=lambda fr: abs(summarize_run(fr[1])["min_val_loss"] - best_loss))[:5]
                for f, r in ranked:
                    i = summarize_run(r)
                    print(f"   {f.name} | {i['epochs']}/{i['max']} | {i['best_epoch']} | {i['min_val_loss']:.4f}")
            for f, r in near_matches(runs, best_epoch, best_loss)[1]:
                i = summarize_run(r)
                print(f"  Same loss but best epoch {i['best_epoch']} (not {best_epoch}): {f.name} - "
                      f"if this is the run, tell Claude and share this line.")
            print(f"  The log of the run that produced best_{arch}.pth is probably not in the folders read. "
                  f"Look for it elsewhere (Colab / Kaggle / another PC) and pass its path or pasted console output, "
                  f"or use --epochs-run N.")
            continue
        f, run = hits[-1]
        full = len(run) >= run[0]["max"]
        record = {
            "architecture": arch, "max_epochs": run[0]["max"], "epochs_run": len(run), "best_epoch": log_best,
            # The log file cannot tell an early-stopping exit from a manual stop, so say only what is known.
            "stop_reason": "Full epoch budget" if full else "Ended before the epoch budget",
            "best_val_loss": best_loss, "best_val_acc": facts["val_acc"],
            "history": {k: [r[k] for r in run] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "lr")},
            "source": f"imported from {f.name}"}
        rule = best_rule_of(run, log_best)
        if rule:
            record["best_rule"] = rule
        write_json(plots_dir / f"training_{arch}.json", record)
        print(f"[ok] {arch}: {len(run)} epochs run of {run[0]['max']}, best epoch {log_best} "
              f"({'full budget' if full else 'ended before the budget'}; best epoch chosen by "
              f"{(rule or 'an unidentified rule').lower()}) - from {f.name}")


if __name__ == "__main__":
    main()
