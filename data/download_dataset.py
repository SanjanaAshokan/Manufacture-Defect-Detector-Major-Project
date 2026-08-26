"""
MVTec AD Transistor Dataset Downloader & Preprocessor
=====================================================
Downloads the MVTec Anomaly Detection dataset (Transistor category),
extracts it, and creates a binary classification directory structure
(good / defective) with train/val/test splits.

Usage:
    python data/download_dataset.py
    python data/download_dataset.py --manual  (if you downloaded manually)
"""

import os
import sys
import shutil
import random
import argparse
import tarfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_directory_structure(processed_dir: Path):
    """Create the processed binary classification directory structure."""
    for split in ["train", "val", "test"]:
        for cls in ["good", "defective"]:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)
    print(f"[✓] Created directory structure at {processed_dir}")


def organize_mvtec_to_binary(raw_dir: Path, processed_dir: Path, val_split: float = 0.2, seed: int = 42):
    """
    Convert MVTec AD transistor structure to binary classification:
        - train/good/    ← from raw train/good (split into train + val)
        - val/good/      ← held-out portion of train/good
        - test/good/     ← from raw test/good
        - test/defective/ ← from raw test/{bent_lead, cut_lead, damaged_case, misplaced}
    
    For training defective samples, we also need some defective images in train/defective.
    Since MVTec only has defective images in test, we split them across train/val/test.
    """
    random.seed(seed)
    create_directory_structure(processed_dir)

    # ------- Good images -------
    train_good_src = raw_dir / "train" / "good"
    test_good_src = raw_dir / "test" / "good"

    if not train_good_src.exists():
        print(f"[✗] Training good directory not found: {train_good_src}")
        print("    Make sure the MVTec transistor dataset is extracted correctly.")
        return False

    # Split train/good into train + val
    good_images = sorted(list(train_good_src.glob("*.png")))
    if not good_images:
        good_images = sorted(list(train_good_src.glob("*.jpg")) + list(train_good_src.glob("*.bmp")))

    random.shuffle(good_images)
    val_count = max(1, int(len(good_images) * val_split))
    val_good = good_images[:val_count]
    train_good = good_images[val_count:]

    # Copy train good
    for img in train_good:
        shutil.copy2(img, processed_dir / "train" / "good" / img.name)
    print(f"[✓] Train/good: {len(train_good)} images")

    # Copy val good
    for img in val_good:
        shutil.copy2(img, processed_dir / "val" / "good" / img.name)
    print(f"[✓] Val/good: {len(val_good)} images")

    # Copy test good
    if test_good_src.exists():
        test_good_images = sorted(list(test_good_src.glob("*.png")))
        if not test_good_images:
            test_good_images = sorted(
                list(test_good_src.glob("*.jpg")) + list(test_good_src.glob("*.bmp"))
            )
        for img in test_good_images:
            shutil.copy2(img, processed_dir / "test" / "good" / img.name)
        print(f"[✓] Test/good: {len(test_good_images)} images")

    # ------- Defective images -------
    defect_types = ["bent_lead", "cut_lead", "damaged_case", "misplaced"]
    all_defective = []

    for defect in defect_types:
        defect_dir = raw_dir / "test" / defect
        if defect_dir.exists():
            imgs = sorted(list(defect_dir.glob("*.png")))
            if not imgs:
                imgs = sorted(list(defect_dir.glob("*.jpg")) + list(defect_dir.glob("*.bmp")))
            # Prefix with defect type to avoid name collisions
            all_defective.extend([(img, defect) for img in imgs])
            print(f"    Found {len(imgs)} images in {defect}")
        else:
            print(f"    [!] Defect directory not found: {defect_dir}")

    if not all_defective:
        print("[✗] No defective images found!")
        return False

    # Shuffle and split defective images: 60% train, 20% val, 20% test
    random.shuffle(all_defective)
    total = len(all_defective)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    train_defective = all_defective[:train_end]
    val_defective = all_defective[train_end:val_end]
    test_defective = all_defective[val_end:]

    # Copy with prefixed names to avoid collisions
    for img_path, defect_type in train_defective:
        dest_name = f"{defect_type}_{img_path.name}"
        shutil.copy2(img_path, processed_dir / "train" / "defective" / dest_name)

    for img_path, defect_type in val_defective:
        dest_name = f"{defect_type}_{img_path.name}"
        shutil.copy2(img_path, processed_dir / "val" / "defective" / dest_name)

    for img_path, defect_type in test_defective:
        dest_name = f"{defect_type}_{img_path.name}"
        shutil.copy2(img_path, processed_dir / "test" / "defective" / dest_name)

    print(f"[✓] Train/defective: {len(train_defective)} images")
    print(f"[✓] Val/defective: {len(val_defective)} images")
    print(f"[✓] Test/defective: {len(test_defective)} images")

    # ------- Summary -------
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Train: {len(train_good)} good + {len(train_defective)} defective = {len(train_good) + len(train_defective)}")
    print(f"  Val:   {len(val_good)} good + {len(val_defective)} defective = {len(val_good) + len(val_defective)}")

    test_good_count = len(test_good_images) if test_good_src.exists() else 0
    print(f"  Test:  {test_good_count} good + {len(test_defective)} defective = {test_good_count + len(test_defective)}")
    print("=" * 50)

    return True


def extract_tar(tar_path: Path, extract_dir: Path):
    """Extract a tar.xz file."""
    print(f"[...] Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r:xz") as tar:
        tar.extractall(path=extract_dir)
    print(f"[✓] Extracted to {extract_dir}")


def copy_example_images(processed_dir: Path, examples_dir: Path, count: int = 3):
    """Copy a few sample images to the web examples directory for the demo."""
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Copy some good and defective examples
    for cls in ["good", "defective"]:
        src_dir = processed_dir / "test" / cls
        if src_dir.exists():
            images = sorted(list(src_dir.glob("*")))[:count]
            for img in images:
                dest_name = f"example_{cls}_{img.name}"
                shutil.copy2(img, examples_dir / dest_name)

    copied = len(list(examples_dir.glob("*")))
    print(f"[✓] Copied {copied} example images to {examples_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare MVTec AD Transistor dataset")
    parser.add_argument(
        "--manual", action="store_true",
        help="Skip download — use this if you manually placed the dataset in data/mvtec_transistor/"
    )
    parser.add_argument(
        "--raw-dir", type=str, default=str(PROJECT_ROOT / "data" / "mvtec_transistor"),
        help="Path to raw MVTec transistor directory"
    )
    parser.add_argument(
        "--processed-dir", type=str, default=str(PROJECT_ROOT / "data" / "processed"),
        help="Path to output processed directory"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    examples_dir = PROJECT_ROOT / "web" / "examples"

    print("=" * 60)
    print("MVTec AD Transistor Dataset — Download & Preparation")
    print("=" * 60)

    # Check if already processed
    if processed_dir.exists() and any(processed_dir.rglob("*.png")):
        print(f"[!] Processed dataset already exists at {processed_dir}")
        response = input("    Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("[!] Skipping. Use existing dataset.")
            return

    # Check for raw dataset
    if not raw_dir.exists() or not (raw_dir / "train").exists():
        # Check for tar file
        tar_path = PROJECT_ROOT / "data" / "transistor.tar.xz"
        if tar_path.exists():
            extract_tar(tar_path, PROJECT_ROOT / "data")
            # MVTec extracts to a 'transistor' directory
            extracted = PROJECT_ROOT / "data" / "transistor"
            if extracted.exists() and not raw_dir.exists():
                extracted.rename(raw_dir)
        else:
            print("\n" + "=" * 60)
            print("MANUAL DOWNLOAD REQUIRED")
            print("=" * 60)
            print("""
The MVTec AD dataset requires accepting a license agreement.

Steps to download:
  1. Go to: https://www.mvtec.com/company/research/datasets/mvtec-ad
  2. Click 'Download' and accept the license agreement
  3. Download the 'transistor' category (~300 MB)
  4. Extract to: data/mvtec_transistor/

  Expected structure after extraction:
    data/mvtec_transistor/
    ├── train/
    │   └── good/          (213 images)
    ├── test/
    │   ├── good/          (60 images)
    │   ├── bent_lead/     (defective)
    │   ├── cut_lead/      (defective)
    │   ├── damaged_case/  (defective)
    │   └── misplaced/     (defective)
    └── ground_truth/
        ├── bent_lead/
        ├── cut_lead/
        ├── damaged_case/
        └── misplaced/

Alternative: Download from Kaggle:
  https://www.kaggle.com/datasets/ipythonx/mvtec-ad

After downloading, run again:
  python data/download_dataset.py --manual
""")
            return

    # Process the dataset
    print(f"\n[...] Processing raw dataset: {raw_dir}")
    success = organize_mvtec_to_binary(raw_dir, processed_dir, args.val_split, args.seed)

    if success:
        # Copy examples for the web demo
        copy_example_images(processed_dir, examples_dir)
        print("\n[✓] Dataset preparation complete!")
        print(f"    Processed dataset: {processed_dir}")
        print(f"    Example images:    {examples_dir}")
    else:
        print("\n[✗] Dataset preparation failed. Check the raw dataset directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
