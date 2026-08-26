#!/bin/bash
# ============================================================
# One-Click Training Launcher (Linux/macOS)
# ============================================================
echo "============================================================"
echo "Starting Transistor Defect Detector Training..."
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

# Run training
python src/train.py "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[✗] Training failed with error code $EXIT_CODE."
    exit $EXIT_CODE
fi

echo "[✓] Training complete! Check outputs/checkpoints/ for saved weights."
