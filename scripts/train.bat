@echo off
:: ============================================================
:: One-Click Training Launcher (Windows)
:: ============================================================
echo ============================================================
echo Starting Transistor Defect Detector Training...
echo ============================================================

:: Check for virtual environment
if exist venv\Scripts\activate.bat (
    echo [✓] Found virtual environment in venv/
    call venv\Scripts\activate.bat
) else (
    echo [!] No virtual environment found. Running with global python...
)

:: Run training
python src/train.py %*

if %errorlevel% neq 0 (
    echo [✗] Training failed with error code %errorlevel%.
    pause
    exit /b %errorlevel%
)

echo [✓] Training complete! Check outputs/checkpoints/ for saved weights.
pause
