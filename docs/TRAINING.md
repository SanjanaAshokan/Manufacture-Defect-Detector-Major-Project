# 🧠 Model Training & Fine-Tuning Guide

This guide covers model training parameters, backbone configuration, early stopping metrics, and learning rate scheduling techniques.

---

## Centralized Configuration (`configs/config.yaml`)

All training hyperparameters are declared in the central YAML configuration file:
- `model.architecture`: Determines whether to train `efficientnet_b0` or `resnet50`.
- `training.epochs`: Number of learning loops (default: `30`).
- `training.batch_size`: Batch dimensions (default: `32`). Reduce to `16` or `8` if experiencing GPU Out-of-Memory (OOM) errors.
- `training.learning_rate`: Maximum learning rate bounds (default: `0.001`).

---

## Fine-Tuning Strategy

The training execution follows a two-phase transfer learning routine:

### Phase 1: Feature Extraction (Warmup)
- Backbone network weights are frozen.
- Only the newly attached custom classifier head parameters are trainable.
- Duration is controlled by `model.warmup_epochs` (default: `3`).
- This builds a strong base mapping without destroying pre-trained ImageNet representations.

### Phase 2: Full Fine-Tuning
- After the warmup phase, all layers in the backbone are unfrozen.
- The learning rate is cut by a factor of 10 (`learning_rate * 0.1`) to prevent rapid model drift.
- The entire model is fine-tuned to capture subtle transistor lead and casing defects.

---

## Running Training

### Quick Launch
Using the pre-configured scripts:
```bash
# Windows
scripts\train.bat

# Linux/macOS
chmod +x scripts/train.sh
./scripts/train.sh
```

### CLI Execution & Overrides
You can execute the training loop directly with python to override configuration values on the fly:
```bash
# Override architecture to ResNet-50
python src/train.py --architecture resnet50

# Override epochs and batch size
python src/train.py --epochs 40 --batch-size 16

# Disable half-precision training
python src/train.py --no-amp
```

---

## Convergence & TensorBoard Monitoring

Logs are saved under `outputs/logs/`. To monitor real-time training progress:
```bash
tensorboard --logdir outputs/logs
```
Open the provided loopback URL (usually `http://localhost:6006`) to view validation accuracy curves and learning rate decay trends.

---

## Troubleshooting

### GPU Out-of-Memory (OOM)
If you see `RuntimeError: CUDA out of memory`:
1. Open `configs/config.yaml`.
2. Lower `training.batch_size` to `16` or `8`.
3. Re-run training.

### Class Imbalance
The dataset contains fewer defective samples than good ones. The training script automatically calculates inverse-frequency class weights and applies them to `nn.CrossEntropyLoss(weight=class_weights)` to prevent the model from biasing towards the majority class.
