#!/bin/bash
# ============================================================
# One-Click Evaluation Launcher (Linux/macOS)
# ============================================================
echo "============================================================"
echo "Starting Transistor Defect Detector Evaluation..."
echo "============================================================"

# Check for virtual environment
if [ -d "venv" ]; then
    echo "[✓] Found virtual environment in venv/"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "[✓] Found virtual environment in .venv/"
    source .venv/bin/activate
else
    echo "[!] No virtual environment found. Running with global python..."
fi

# Run evaluation and generate Grad-CAM samples
python src/evaluate.py --generate-gradcam "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[✗] Evaluation failed with error code $EXIT_CODE."
    exit $EXIT_CODE
fi

echo "[✓] Evaluation complete! Check outputs/plots/ and outputs/gradcam_samples/."
