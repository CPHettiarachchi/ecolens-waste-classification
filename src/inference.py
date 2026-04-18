"""
src/inference.py
----------------
Production inference pipeline — single image prediction with confidence scores.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from typing import Union
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import load_checkpoint


def get_inference_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class WastePredictor:
    WASTE_TIPS = {
        "cardboard": "Flatten and place in paper/cardboard recycling bin.",
        "glass":     "Rinse and place in glass recycling container.",
        "metal":     "Rinse cans and recycle in metal bin.",
        "paper":     "Recycle in paper bin. Keep dry.",
        "plastic":   "Check resin code — recycle if accepted locally.",
        "trash":     "General waste bin. Check if any parts are recyclable.",
        "food_organics": "Compost or use organic waste bin.",
        "textile_trash": "Donate if wearable, else textile recycling drop-off.",
        "vegetation":    "Compost or green waste bin.",
        "miscellaneous_trash": "General waste bin.",
    }

    def __init__(self, model_path, device=None, image_size=224):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.checkpoint = load_checkpoint(model_path, self.device)
        self.classes   = self.checkpoint["classes"]
        self.image_size = image_size
        self.transform = get_inference_transform(image_size)

    def _load_image(self, source):
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, bytes):
            return Image.open(io.BytesIO(source)).convert("RGB")
        return Image.open(source).convert("RGB")

    @torch.no_grad()
    def predict(self, source, top_k=3):
        image  = self._load_image(source)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_k = min(top_k, len(self.classes))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_class   = self.classes[top_indices[0]]
        confidence  = float(probs[top_indices[0]])

        return {
            "top_class":    top_class,
            "confidence":   confidence,
            "top_k": [
                {"class": self.classes[i], "confidence": float(probs[i]),
                 "bar_pct": float(probs[i]) * 100}
                for i in top_indices
            ],
            "disposal_tip": self.WASTE_TIPS.get(top_class, "Check local recycling guidelines."),
            "all_probs":    {cls: float(p) for cls, p in zip(self.classes, probs)},
        }


if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser(description="EcoLens Inference")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    predictor = WastePredictor(model_path=config["paths"]["best_model"])
    result = predictor.predict(args.image, top_k=args.top_k)

    print(f"\n  Prediction : {result['top_class'].upper()}")
    print(f"  Confidence : {result['confidence']*100:.1f}%")
    print(f"  Tip        : {result['disposal_tip']}")
    print(f"\n  Top-{args.top_k}:")
    for item in result["top_k"]:
        bar = "█" * int(item["bar_pct"] / 5)
        print(f"    {item['class']:20s} {item['bar_pct']:5.1f}% {bar}")
