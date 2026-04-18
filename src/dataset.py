"""
src/dataset.py
--------------
Custom Dataset class with augmentation pipeline.
Windows-safe: num_workers=0 by default.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Optional, Callable

import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def get_train_transforms(image_size, cfg):
    norm = cfg.get("augmentation", {}).get("normalize", {})
    mean = norm.get("mean", [0.485, 0.456, 0.406])
    std  = norm.get("std",  [0.229, 0.224, 0.225])

    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
            A.OneOf([A.GaussNoise(), A.GaussianBlur(blur_limit=3)], p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class WasteDataset(Dataset):
    def __init__(self, root, transform=None, use_albumentations=True):
        self.root = Path(root)
        self.use_alb = use_albumentations and ALBUMENTATIONS_AVAILABLE
        self._folder = datasets.ImageFolder(str(self.root))
        self.classes      = self._folder.classes
        self.class_to_idx = self._folder.class_to_idx
        self.samples      = self._folder.samples
        self.transform    = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            if self.use_alb and ALBUMENTATIONS_AVAILABLE and hasattr(self.transform, '__call__'):
                try:
                    image_np = np.array(image)
                    augmented = self.transform(image=image_np)
                    image = augmented["image"]
                except Exception:
                    image = transforms.ToTensor()(image)
            else:
                image = self.transform(image)

        return image, label

    def get_sample_weights(self):
        from collections import Counter
        labels = [label for _, label in self.samples]
        counts = Counter(labels)
        weights = [1.0 / counts[label] for _, label in self.samples]
        return torch.tensor(weights, dtype=torch.float)


def build_dataloaders(config):
    processed_dir = Path(config["paths"]["processed_dir"])
    image_size    = config["dataset"]["image_size"]
    batch_size    = config["training"]["batch_size"]
    num_workers   = config["training"]["num_workers"]

    train_transform = get_train_transforms(image_size, config)
    val_transform   = get_val_transforms(image_size)

    train_ds = WasteDataset(processed_dir / "train", train_transform, ALBUMENTATIONS_AVAILABLE)
    val_ds   = WasteDataset(processed_dir / "val",   val_transform,   False)
    test_ds  = WasteDataset(processed_dir / "test",  val_transform,   False)

    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"[INFO] Classes: {train_ds.classes}")

    return {"train": train_loader, "val": val_loader, "test": test_loader,
            "classes": train_ds.classes}
