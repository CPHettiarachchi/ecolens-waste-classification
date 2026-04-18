"""
src/model.py
------------
EfficientNet-B3 with two-phase training strategy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn as nn
from pathlib import Path

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class EcoLensClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4, pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                "efficientnet_b3", pretrained=pretrained,
                num_classes=0, global_pool="avg",
            )
            in_features = self.backbone.num_features
        else:
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            _model = efficientnet_b3(weights=weights)
            in_features = _model.classifier[1].in_features
            self.backbone = nn.Sequential(*list(_model.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[MODEL] Backbone frozen — training classifier head only.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[MODEL] Backbone unfrozen — full fine-tuning.")

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        return self.classifier(features)

    def get_features(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() == 4:
                features = features.mean(dim=[2, 3])
        return features

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(config):
    model_cfg = config["model"]
    num_classes = config["dataset"]["num_classes"]
    model = EcoLensClassifier(
        num_classes=num_classes,
        dropout_rate=model_cfg.get("dropout_rate", 0.4),
        pretrained=model_cfg.get("pretrained", True),
    )
    params = model.count_parameters()
    print(f"[MODEL] Architecture : EfficientNet-B3 + Custom Head")
    print(f"[MODEL] Total params : {params['total']:,}")
    print(f"[MODEL] Trainable    : {params['trainable']:,}")
    return model


def save_checkpoint(model, optimizer, epoch, val_accuracy, config, path, is_best=False):
    checkpoint = {
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "num_classes": model.num_classes,
        "classes": config["dataset"]["classes"],
        "config": config,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = Path(config["paths"]["best_model"])
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, best_path)
        print(f"  [CKPT] Best model saved (val_acc={val_accuracy:.4f})")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = EcoLensClassifier(
        num_classes=checkpoint["num_classes"],
        dropout_rate=config["model"].get("dropout_rate", 0.4),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"[CKPT] Loaded from {path} | Epoch {checkpoint['epoch']} | Val acc {checkpoint['val_accuracy']:.4f}")
    return model, checkpoint
