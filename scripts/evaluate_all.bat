@echo off
REM ============================================================
REM Evaluate every trained model and refresh the Dashboard data
REM ============================================================
REM Writes, for each architecture, outputs\plots\evaluation_results_<arch>.json and
REM confusion_matrix_ / pr_curve_ / roc_curve_<arch>.png, then exports the model facts
REM (parameters, best epoch, training history) the Dashboard shows.
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

REM Default model (ConvNeXt-Tiny) LAST: the home page reads the un-suffixed evaluation_results.json,
REM which is overwritten by whichever model is evaluated last.
for %%A in (efficientnet_b0 resnet50 convnext_tiny) do (
    if exist outputs\checkpoints\best_%%A.pth (
        echo === Evaluating %%A ===
        python src\evaluate.py --architecture %%A
        if errorlevel 1 exit /b 1
    ) else (
        echo [skip] no checkpoint for %%A
    )
)

python scripts\export_model_info.py
echo [done] Reload the Dashboard to see all three models.
