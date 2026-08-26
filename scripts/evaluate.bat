@echo off
:: ============================================================
:: One-Click Evaluation Launcher (Windows)
:: ============================================================
echo ============================================================
echo Starting Transistor Defect Detector Evaluation...
echo ============================================================

:: Check for virtual environment
if exist venv\Scripts\activate.bat (
    echo [✓] Found virtual environment in venv/
    call venv\Scripts\activate.bat
) else (
    echo [!] No virtual environment found. Running with global python...
)

:: Run evaluation and generate Grad-CAM samples
python src/evaluate.py --generate-gradcam %*

if %errorlevel% neq 0 (
    echo [✗] Evaluation failed with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo [✓] Evaluation complete! Check outputs/plots/ and outputs/gradcam_samples/.
pause
