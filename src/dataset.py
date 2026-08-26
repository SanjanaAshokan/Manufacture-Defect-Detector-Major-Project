"""
Dataset & DataLoader for MVTec AD Transistor Binary Classification
==================================================================
Provides a PyTorch Dataset with augmentation pipelines and
a factory function for creating train/val/test DataLoaders.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np


class TransistorDataset(Dataset):
    """
    Binary classification dataset for transistor defect detection.
    
    Expects directory structure:
        root/
        ├── good/        (label = 0)
        └── defective/   (label = 1)
    """

    CLASS_NAMES = ["good", "defective"]
    CLASS_TO_IDX = {"good": 0, "defective": 1}

    def __init__(self, root: str, transform: Optional[transforms.Compose] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.targets = []
        self.imgs = []

        for cls_name, cls_idx in self.CLASS_TO_IDX.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                    self.samples.append((str(img_path), cls_idx))
                    self.targets.append(cls_idx)

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}. Check directory structure.")

        print(f"  Pre-loading {len(self.samples)} images from {root} into RAM...")
        for img_path, _ in self.samples:
            img = Image.open(img_path).convert("RGB")
            img.load()  # Load pixel data to RAM
            self.imgs.append(img)

        print(f"  Loaded {len(self.samples)} images from {root}")
        for cls_name, cls_idx in self.CLASS_TO_IDX.items():
            count = self.targets.count(cls_idx)
            print(f"    {cls_name}: {count}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.imgs[idx]
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for imbalanced data."""
        class_counts = np.bincount(self.targets, minlength=len(self.CLASS_NAMES))
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts.astype(np.float32)
        weights = weights / weights.sum() * len(self.CLASS_NAMES)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.FloatTensor([class_weights[t] for t in self.targets])


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training augmentation pipeline."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation / test transform pipeline (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_dataloaders(
    processed_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        processed_dir: Path to processed dataset with train/val/test splits.
        batch_size: Batch size for training.
        image_size: Target image size (square).
        num_workers: Number of DataLoader workers.
        pin_memory: Pin memory for CUDA transfers.
        use_weighted_sampler: Use WeightedRandomSampler for class imbalance.
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    processed_dir = Path(processed_dir)

    print("\n[Dataset] Loading datasets...")

    # Create datasets
    train_dataset = TransistorDataset(
        root=str(processed_dir / "train"),
        transform=get_train_transforms(image_size)
    )
    val_dataset = TransistorDataset(
        root=str(processed_dir / "val"),
        transform=get_val_transforms(image_size)
    )
    test_dataset = TransistorDataset(
        root=str(processed_dir / "test"),
        transform=get_val_transforms(image_size)
    )

    # Weighted sampler for imbalanced training data
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampler and len(set(train_dataset.targets)) > 1:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_shuffle = False  # Sampler handles shuffling

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
    }

    return dataloaders
