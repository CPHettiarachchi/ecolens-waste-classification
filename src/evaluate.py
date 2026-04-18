"""
src/evaluate.py
---------------
Model evaluation: accuracy, confusion matrix, per-class F1, confidence plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import yaml
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score

from dataset import build_dataloaders
from model import load_checkpoint


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in tqdm(loader, desc="Evaluating", ncols=80):
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return all_labels, all_preds, all_probs


def plot_confusion_matrix(y_true, y_pred, classes, output_dir, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, title = ".2f", "Normalized Confusion Matrix"
    else:
        fmt, title = "d", "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    suffix = "normalized" if normalize else "raw"
    path = output_dir / f"confusion_matrix_{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_per_class_metrics(report, classes, output_dir):
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / "per_class_metrics.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_confidence_distribution(probs, labels, output_dir):
    probs_np  = np.array(probs)
    labels_np = np.array(labels)
    max_probs = probs_np.max(axis=1)
    correct   = (probs_np.argmax(axis=1) == labels_np)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(max_probs[correct],  bins=30, alpha=0.7, label="Correct",   color="steelblue")
    ax.hist(max_probs[~correct], bins=30, alpha=0.7, label="Incorrect", color="salmon")
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Max Softmax Probability")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "confidence_distribution.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    model_path = config["paths"]["best_model"]
    if not Path(model_path).exists():
        print(f"[ERROR] Model not found at {model_path}")
        print("        Run training first: python src/train.py")
        return

    model, checkpoint = load_checkpoint(model_path, device)
    classes = checkpoint.get("classes", config["dataset"]["classes"])

    loaders = build_dataloaders(config)
    y_true, y_pred, y_probs = run_inference(model, loaders["test"], device)

    acc   = accuracy_score(y_true, y_pred)
    k     = min(3, len(classes))
    top3  = top_k_accuracy_score(y_true, y_probs, k=k)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_str  = classification_report(y_true, y_pred, target_names=classes)

    print(f"\n{'='*55}")
    print(f"  Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Top-{k} Accuracy : {top3:.4f} ({top3*100:.2f}%)")
    print(f"  Macro F1       : {report_dict['macro avg']['f1-score']:.4f}")
    print(f"{'='*55}")
    print("\n" + report_str)

    plot_confusion_matrix(y_true, y_pred, classes, output_dir, normalize=True)
    plot_confusion_matrix(y_true, y_pred, classes, output_dir, normalize=False)
    plot_per_class_metrics(report_dict, classes, output_dir)
    plot_confidence_distribution(y_probs, y_true, output_dir)

    full_report = {
        "test_accuracy":   acc,
        "top3_accuracy":   top3,
        "macro_f1":        report_dict["macro avg"]["f1-score"],
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall":    report_dict["macro avg"]["recall"],
        "per_class":       {cls: report_dict[cls] for cls in classes},
    }
    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[INFO] All reports saved to {output_dir}/")
    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    evaluate(config)
