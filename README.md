# 🔬 Transistor Defect Detector

An end-to-end, production-ready quality control inspection system for transistors. The system uses transfer learning with fine-tuned **EfficientNet-B0** and **ResNet-50** classifiers combined with **Grad-CAM (Gradient-weighted Class Activation Mapping)** for explainable anomaly localization. 

This repository includes a custom-designed, fully responsive dark-mode web application for local or Hugging Face Spaces deployment using Docker.

## 🚀 Key Features

- **Multi-Arch Support**: Toggle between **EfficientNet-B0**, balanced **ConvNeXt-Tiny**, and heavy **ResNet-50** backbones.
- **Explainable AI (XAI)**: Generates real-time Grad-CAM heatmap overlays indicating defect hotspots.
- **Interactive Web Interface**: Custom glassmorphism dashboard with drag-and-drop file upload, light/dark mode toggle, real-time prediction confidence, recent detection history, and visual explanations.
- **Production Dashboard**: Complete performance metrics with Confusion Matrix, Precision-Recall Curve, ROC Curve, per-defect-type recall breakdown, and dynamic confidence threshold analysis table — all pulled live from the latest evaluation run, never hardcoded.
- **Live Model-Status Banner**: The site automatically flags itself as "demo mode" on any page whenever no trained checkpoint is loaded, so an untrained deployment can never be mistaken for a validated one.
- **Confidence Calibration**: Optional temperature-scaling step (`src/calibrate.py`) so the confidence % shown in the UI reflects real-world accuracy rather than raw, often overconfident softmax output. Low-margin predictions are also flagged as "uncertain" in the UI.
- **Containerized Deployment**: Pre-configured Docker files optimized for immediate deployment on HF Spaces.

---

## 📂 Project Structure

```
transistor-defect-detector/
├── README.md                         # Project info & HF Space metadata
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker container configuration
├── .gitignore                        # Git ignore rules
│
├── configs/
│   └── config.yaml                   # Centralized configuration (hyperparams, layers, port)
│
├── data/
│   ├── download_dataset.py           # Dataset downloader & split parser
│   └── README.md                     # Dataset details & MVTec license
│
├── src/
│   ├── dataset.py                    # PyTorch Dataset, augmentation, dataloaders
│   ├── models.py                     # EfficientNet-B0 / ConvNeXt-Tiny / ResNet-50 implementations
│   ├── gradcam.py                    # Hook-based Grad-CAM computation
│   ├── train.py                      # Training loop with AMP & OneCycleLR
│   ├── evaluate.py                   # Testing metrics & plots generation
│   ├── predict.py                    # CLI prediction tool
│   ├── realtime.py                   # Real-time webcam inference with live Grad-CAM
│   └── utils.py                      # Saving/loading checkpoints & helper classes
│
├── web/
│   ├── app.py                        # Flask server backend
│   ├── templates/                    # HTML structure files (index, dashboard, detect, realtime)
│   └── static/                       # Custom CSS, JS, and image assets
│
├── scripts/
│   ├── train.bat / train.sh          # One-click training launchers
│   └── evaluate.bat / evaluate.sh    # One-click evaluation launchers
│
└── docs/
    ├── SETUP.md                      # Installation & setup guide
    ├── TRAINING.md                   # Model training & fine-tuning manual
    ├── DEPLOYMENT.md                 # Hugging Face Spaces deployment manual
    └── ARCHITECTURE.md               # Pipeline architecture design decisions
```

---

## 🛠️ Quick Start

### 1. Installation
Ensure you have Python 3.9+ installed. Clone the repository and run:
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 2. Download and Split Dataset
The model trains on the **MVTec AD Transistor** dataset:
```bash
python data/download_dataset.py
```
*(If automated download fails, download the transistor tarball manually from MVTec, place it in `data/mvtec_transistor` and run the script with `--manual`)*

### 3. Model Training & Evaluation
Train the default model (ConvNeXt-Tiny) with early stopping and mixed precision:
```bash
# Windows
scripts\train.bat

# Linux/macOS
chmod +x scripts/train.sh
./scripts/train.sh
```

Evaluate the trained checkpoint and generate visualization curves, metrics JSON, and the per-defect-type breakdown:
```bash
# Windows
scripts\evaluate.bat

# Linux/macOS
chmod +x scripts/evaluate.sh
./scripts/evaluate.sh
```
This writes `outputs/plots/evaluation_results.json` — the website reads this file directly, so the landing page and dashboard update automatically the moment it exists. Until you run this step, the site stays in "demo mode" and shows no fabricated numbers.

### 4. (Optional) Calibrate Confidence
Fit a temperature-scaling correction so the confidence % shown in the UI matches real accuracy instead of an overconfident raw softmax value:
```bash
python src/calibrate.py
```
Saves `outputs/plots/calibration_<architecture>.json`; the web app and `evaluate.py` both pick it up automatically on the next run.

### 5. Run the Web Server Locally
Start the custom Flask dashboard:
```bash
python web/app.py
```
Open **[http://localhost:7860](http://localhost:7860)** in your browser. If you haven't trained/evaluated yet, the site will clearly show a "Demo mode" banner instead of pretending to be a validated model.

### 6. Real-time Webcam Inference
To run real-time defect detection using your webcam with live Grad-CAM overlay:
```bash
python src/realtime.py
```
*Note: The project uses `opencv-python-headless` by default. To use the desktop window for real-time capture, you must have the full OpenCV version installed. If the window doesn't appear, run `pip uninstall opencv-python-headless -y` and `pip install opencv-python`.*

---

## 📊 Model Performance

Accuracy, precision, recall, F1, per-defect-type recall, and calibration error are **generated by `src/evaluate.py`**, saved to `outputs/plots/evaluation_results.json`, and rendered live on the landing page and `/dashboard` — they are never hardcoded in the website or this README. Run evaluation against your own trained checkpoint to see current numbers in the app.

> Two real training bugs were fixed in this version that were previously producing unreliable predictions:
> 1. The `OneCycleLR` scheduler was never actually stepped, so the learning rate stayed frozen near its starting value for the whole run.
> 2. Class imbalance was being corrected twice — once via `WeightedRandomSampler` and again via class-weighted loss — which skewed precision/recall. This is now a single config flag (`training.use_weighted_sampler` in `configs/config.yaml`) driving both.
>
> Re-train with `scripts/train.sh` (or `train.bat`) to get a checkpoint reflecting these fixes.

---

## 📝 Documentation
For detailed guides on specific components:
- [Installation Guide](docs/SETUP.md)
- [Training Manual](docs/TRAINING.md)
- [Deployment Instructions](docs/DEPLOYMENT.md)
- [System Architecture](docs/ARCHITECTURE.md)