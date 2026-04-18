"""
src/train.py
------------
Production training loop with:
  - Two-phase fine-tuning (freeze -> unfreeze)
  - Mixed precision (AMP) for speed (GPU only)
  - Cosine annealing LR scheduler with linear warmup
  - Label smoothing cross-entropy loss
  - Gradient clipping
  - Early stopping
  - MLflow experiment tracking
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import yaml
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from dataset import build_dataloaders
from model import build_model, save_checkpoint


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[DEVICE] Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("[DEVICE] CPU (training will be slow — consider setting epochs: 5 to test)")
    return device


class EarlyStopping:
    def __init__(self, patience=7, mode="max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False
        improved = (metric > self.best) if self.mode == "max" else (metric < self.best)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            print(f"  [ES] No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("  [ES] Early stopping triggered.")
                self.stop = True
        return self.stop


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=eta_min
        )

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale
        else:
            self._cosine.step()

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def run_epoch(model, loader, criterion, optimizer, device, grad_clip, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"  {phase.upper():5s}", leave=False, ncols=80)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


def train(config):
    set_seed(config["training"]["seed"])
    device = get_device()

    loaders = build_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = build_model(config)
    model.to(device)
    model.freeze_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["optimizer"]["lr_head"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    total_epochs = config["training"]["epochs"]
    warmup_epochs = config["scheduler"]["warmup_epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs, total_epochs,
        eta_min=config["scheduler"]["eta_min"],
    )

    early_stop = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        mode=config["early_stopping"]["mode"],
    )

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(config["paths"]["mlflow_uri"])
            mlflow.set_experiment(config["project"]["name"])
            mlflow.start_run(run_name=f"efficientnet_b3_{int(time.time())}")
            mlflow.log_params({
                "architecture": config["model"]["architecture"],
                "epochs": total_epochs,
                "batch_size": config["training"]["batch_size"],
            })
        except Exception:
            pass

    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    history = defaultdict(list)
    unfreeze_epoch = config["model"]["unfreeze_epoch"]
    best_val_acc = 0.0

    print(f"\n{'='*55}")
    print(f"  EcoLens Training — {config['dataset']['num_classes']} classes")
    print(f"  Epochs: {total_epochs} | Batch: {config['training']['batch_size']} | Device: {device}")
    print(f"{'='*55}\n")

    for epoch in range(total_epochs):
        print(f"Epoch [{epoch+1:3d}/{total_epochs}]  LR={scheduler.get_last_lr()[0]:.6f}")

        if epoch == unfreeze_epoch:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": config["optimizer"]["lr_backbone"]},
                {"params": model.classifier.parameters(), "lr": config["optimizer"]["lr_head"] * 0.1},
            ], weight_decay=config["optimizer"]["weight_decay"])
            scheduler = WarmupCosineScheduler(
                optimizer, 0, max(total_epochs - unfreeze_epoch, 1),
                eta_min=config["scheduler"]["eta_min"],
            )

        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                          config["training"]["gradient_clip"], "train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device,
                                      config["training"]["gradient_clip"], "val")
        elapsed = time.time() - t0
        scheduler.step(epoch)

        for k, v in [("train_loss", train_loss), ("train_acc", train_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[k].append(v)

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({"train_loss": train_loss, "train_acc": train_acc,
                                    "val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            except Exception:
                pass

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        save_checkpoint(model, optimizer, epoch + 1, val_acc, config,
                        models_dir / f"checkpoint_epoch{epoch+1:03d}.pth", is_best=is_best)

        print(f"  Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"{'BEST' if is_best else '':4s} [{elapsed:.0f}s]")

        if early_stop(val_acc):
            print(f"\n[INFO] Early stopped at epoch {epoch+1}")
            break

    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metric("best_val_accuracy", best_val_acc)
            mlflow.end_run()
        except Exception:
            pass

    print(f"\n{'='*55}")
    print(f"  Training complete! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {config['paths']['best_model']}")
    print(f"{'='*55}\n")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    history = train(config)

    Path("reports").mkdir(exist_ok=True)
    with open("reports/training_history.json", "w") as f:
        json.dump(dict(history), f, indent=2)
    print("[INFO] Training history saved to reports/training_history.json")
