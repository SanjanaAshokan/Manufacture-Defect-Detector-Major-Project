#!/bin/bash
# ============================================================
# Evaluate every trained model and refresh the Dashboard data
# ============================================================
# Writes, for each architecture, outputs/plots/evaluation_results_<arch>.json and
# confusion_matrix_/pr_curve_/roc_curve_<arch>.png, then exports the model facts
# (parameters, best epoch, training history) the Dashboard shows.
if [ -d "venv" ]; then source venv/bin/activate; elif [ -d ".venv" ]; then source .venv/bin/activate; fi

# Default model (ConvNeXt-Tiny) LAST: the home page reads the un-suffixed evaluation_results.json,
# which is overwritten by whichever model is evaluated last.
for arch in efficientnet_b0 resnet50 convnext_tiny; do
    if [ -f "outputs/checkpoints/best_${arch}.pth" ]; then
        echo "=== Evaluating ${arch} ==="
        python src/evaluate.py --architecture "${arch}" || exit $?
    else
        echo "[skip] no checkpoint for ${arch}"
    fi
done

python scripts/export_model_info.py
echo "[done] Reload the Dashboard to see all three models."
