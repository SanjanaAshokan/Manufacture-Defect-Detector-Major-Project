# 🛠️ Setup & Installation Guide

This document describes the environment setup, dataset acquisition, and folder preparation steps for running the **Transistor Defect Detector** system.

---

## System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS Catalina+
- **Python Version**: 3.9, 3.10 (recommended), or 3.11
- **Hardware Prerequisites**:
  - Minimum: 8 GB RAM, Quad-core CPU
  - Recommended: 16 GB RAM, NVIDIA GPU with 6+ GB VRAM for accelerated training (CUDA 11.7+)

---

## 1. Environment Preparation

We recommend using a clean virtual environment or a conda environment to prevent package version conflicts.

### Using Python virtualenv
```bash
# Create the environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate.bat

# Activate on Linux/macOS
source venv/bin/activate
```

### Using Conda
```bash
# Create the environment
conda create -n defect_detector python=3.10 -y

# Activate
conda activate defect_detector
```

---

## 2. Installing Dependencies

Install the locked package dependencies using pip:
```bash
pip install -r requirements.txt
```

### GPU Support Note (CUDA)
If you have a compatible NVIDIA GPU and want to train with CUDA acceleration, verify PyTorch detects your GPU:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```
If this prints `False` but you have a GPU, reinstall PyTorch matching your CUDA version from the [PyTorch Get Started](https://pytorch.org/get-started/locally/) guide.

---

## 3. Dataset Setup

The Transistor Defect Detector uses the **MVTec Anomaly Detection (AD)** benchmark dataset. 

### Automated Setup (Recommended)
Run the dataset utility script:
```bash
python data/download_dataset.py
```
This script will attempt to retrieve and download the dataset, extract the contents, parse defect subdirectories, perform a validation partition split, and copy example assets into `web/examples/` for the web dashboard.

### Manual Setup Fallback
If the automated download fails due to network constraints:
1. Navigate to the [MVTec AD Dataset request form](https://www.mvtec.com/company/research/datasets/mvtec-ad).
2. Agree to the CC BY-NC-SA 4.0 license conditions.
3. Download the **Transistor** archive file (~300 MB).
4. Save the archive file to `data/transistor.tar.xz`.
5. Run the script again to extract and organize:
   ```bash
   python data/download_dataset.py
   ```
6. Verify your folder structure matches:
   - `data/processed/train/{good, defective}`
   - `data/processed/val/{good, defective}`
   - `data/processed/test/{good, defective}`

---

## 4. Verification Check

Run a quick test to make sure everything is installed correctly:
```bash
python -c "import torch; import cv2; import gradcam; print('All libraries loaded correctly!')"
```
If you see the validation print without import errors, you are ready to train!
