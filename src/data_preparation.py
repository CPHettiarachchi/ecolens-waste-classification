"""
src/data_preparation.py
-----------------------
Handles dataset validation, auto-detection, and train/val/test splitting.
Dataset must be downloaded manually from Kaggle (see instructions below).
"""

import os
import shutil
import random
import yaml
import json
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def find_class_dirs(raw_dir: Path, classes: list) -> Path:
    """
    Handles nested zip extractions like:
      data/raw/RealWaste/cardboard/
      data/raw/archive/RealWaste/cardboard/
    Returns the directory that directly contains the class subfolders.
    """
    if any((raw_dir / cls).exists() for cls in classes):
        return raw_dir

    for subdir in sorted(raw_dir.iterdir()):
        if subdir.is_dir():
            if any((subdir / cls).exists() for cls in classes):
                print(f"[INFO] Found class folders inside: {subdir.name}/")
                return subdir

    for subdir in sorted(raw_dir.iterdir()):
        if subdir.is_dir():
            for subsubdir in sorted(subdir.iterdir()):
                if subsubdir.is_dir():
                    if any((subsubdir / cls).exists() for cls in classes):
                        print(f"[INFO] Found class folders inside: {subdir.name}/{subsubdir.name}/")
                        return subsubdir

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(raw_dir.rglob("*")):
        if p.is_dir():
            if any(f.suffix.lower() in image_exts for f in p.iterdir() if f.is_file()):
                print(f"[INFO] Fallback: using image parent dir: {p.parent}")
                return p.parent

    return raw_dir


def validate_images(directory: Path, min_size: int = 32) -> list:
    valid_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_files = list(directory.rglob("*"))
    for fp in tqdm(all_files, desc=f"  Validating {directory.name}", leave=False, ncols=80):
        if fp.suffix.lower() not in image_extensions:
            continue
        try:
            with Image.open(fp) as img:
                w, h = img.size
                if w >= min_size and h >= min_size:
                    valid_paths.append(fp)
        except Exception:
            print(f"  [WARN] Corrupt image skipped: {fp.name}")
    return valid_paths


def split_dataset(raw_dir, processed_dir, classes, splits, seed=42):
    set_seed(seed)
    train_r, val_r, test_r = splits
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6

    for split in ("train", "val", "test"):
        for cls in classes:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {"train": Counter(), "val": Counter(), "test": Counter()}

    for cls in classes:
        class_dir = raw_dir / cls
        if not class_dir.exists():
            print(f"  [SKIP] {cls} — folder not found")
            continue

        valid_images = validate_images(class_dir)
        if not valid_images:
            print(f"  [SKIP] {cls} — no valid images found")
            continue

        random.shuffle(valid_images)
        n = len(valid_images)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)

        splits_map = {
            "train": valid_images[:n_train],
            "val":   valid_images[n_train : n_train + n_val],
            "test":  valid_images[n_train + n_val :],
        }

        for split_name, files in splits_map.items():
            dest_dir = processed_dir / split_name / cls
            for src in files:
                shutil.copy2(src, dest_dir / src.name)
            summary[split_name][cls] = len(files)

        print(
            f"  {cls:25s}  "
            f"train={summary['train'][cls]:4d}  "
            f"val={summary['val'][cls]:4d}  "
            f"test={summary['test'][cls]:4d}"
        )

    return summary


def save_summary(summary, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({split: dict(counts) for split, counts in summary.items()}, f, indent=2)
    print(f"[INFO] Summary saved to {output_path}")


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    raw_dir       = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    classes       = config["dataset"]["classes"]
    splits = (
        config["dataset"]["train_split"],
        config["dataset"]["val_split"],
        config["dataset"]["test_split"],
    )

    raw_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    has_images = any(f.suffix.lower() in image_exts for f in raw_dir.rglob("*") if f.is_file())

    if not has_images:
        print("\n" + "="*60)
        print("  DATASET NOT FOUND — Manual download required")
        print("="*60)
        print("""
  1. Go to:
     https://www.kaggle.com/datasets/joebeachcapital/realwaste

  2. Click [Download] (login required) -> realwaste.zip (~185 MB)

  3. Extract the zip. You will see folders like:
       cardboard/  glass/  plastic/  paper/ ...

  4. Copy ALL those class folders into:
       D:\\clp\\EcoLens\\data\\raw\\

     Final structure:
       data/raw/cardboard/   <- contains .jpg images
       data/raw/glass/
       data/raw/plastic/
       ... etc

  5. Re-run:  python data_preparation.py
""")
        print("="*60)
        return

    actual_root = find_class_dirs(raw_dir, classes)
    print(f"[INFO] Data root detected: {actual_root}\n")
    print("[INFO] Class inventory:")

    found_classes = []
    for cls in classes:
        cls_dir = actual_root / cls
        if cls_dir.exists():
            n = len([f for f in cls_dir.rglob("*") if f.suffix.lower() in image_exts])
            print(f"  OK  {cls:25s} ({n:4d} images)")
            found_classes.append(cls)
        else:
            print(f"  --  {cls:25s} (not found)")

    if not found_classes:
        print("\n[ERROR] None of the expected class folders were found.")
        print("  Folders currently in raw dir:")
        for p in sorted(actual_root.iterdir()):
            print(f"    -> {p.name}")
        print("\n  Update config.yaml > dataset > classes to match the names above.")
        return

    print(f"\n[INFO] Found {len(found_classes)}/{len(classes)} classes. Splitting...\n")
    summary = split_dataset(actual_root, processed_dir, found_classes, splits)
    save_summary(summary, Path("reports/dataset_summary.json"))

    total = {split: sum(counts.values()) for split, counts in summary.items()}
    print(f"\n{'='*50}")
    print(f"  train={total['train']}  val={total['val']}  test={total['test']}")
    print(f"  Total: {sum(total.values())} images processed")
    print(f"{'='*50}")
    print("[INFO] Data preparation complete.\n")


if __name__ == "__main__":
    main()
