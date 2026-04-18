"""
tests/test_model.py
-------------------
Unit tests for model architecture and inference pipeline.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import numpy as np
from PIL import Image

from model import EcoLensClassifier


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Create an untrained model for architecture tests."""
    return EcoLensClassifier(num_classes=9, dropout_rate=0.4, pretrained=False)


@pytest.fixture
def dummy_batch():
    """224×224 batch of 4 random images."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def dummy_image():
    """A random PIL image."""
    arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ─────────────────────────────────────────────
#  Model Architecture Tests
# ─────────────────────────────────────────────

class TestModelArchitecture:

    def test_output_shape(self, model, dummy_batch):
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)
        assert output.shape == (4, 9), f"Expected (4, 9), got {output.shape}"

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        frozen = all(not p.requires_grad for p in model.backbone.parameters())
        head_trainable = any(p.requires_grad for p in model.classifier.parameters())
        assert frozen, "Backbone should be frozen"
        assert head_trainable, "Classifier head should be trainable"

    def test_unfreeze_backbone(self, model):
        model.freeze_backbone()
        model.unfreeze_backbone()
        trainable = all(p.requires_grad for p in model.backbone.parameters())
        assert trainable, "Backbone should be trainable after unfreeze"

    def test_parameter_count(self, model):
        params = model.count_parameters()
        assert params["total"] > 1_000_000, "Model should have > 1M parameters"
        assert params["trainable"] > 0, "Should have trainable parameters"

    def test_output_not_nan(self, model, dummy_batch):
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_get_features(self, model, dummy_batch):
        features = model.get_features(dummy_batch)
        assert features.shape[0] == 4
        assert features.dim() == 2


# ─────────────────────────────────────────────
#  Inference Transform Tests
# ─────────────────────────────────────────────

class TestInferenceTransforms:

    def test_single_transform(self, dummy_image):
        from inference import get_inference_transform
        transform = get_inference_transform(image_size=224)
        tensor = transform(dummy_image)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_tta_transforms(self, dummy_image):
        from inference import get_tta_transforms
        transforms = get_tta_transforms(image_size=224)
        assert len(transforms) == 5
        for t in transforms:
            tensor = t(dummy_image)
            assert tensor.shape == (3, 224, 224)

    def test_normalization_range(self, dummy_image):
        from inference import get_inference_transform
        transform = get_inference_transform(224)
        tensor = transform(dummy_image)
        # After ImageNet normalization, values should be roughly in [-3, 3]
        assert tensor.min() > -5.0 and tensor.max() < 5.0


# ─────────────────────────────────────────────
#  Dataset Tests (no actual files needed)
# ─────────────────────────────────────────────

class TestDatasetUtilities:

    def test_weighted_sampler_weights_positive(self):
        """WeightedRandomSampler requires all weights > 0."""
        from collections import Counter
        import torch
        labels = [0, 0, 1, 2, 2, 2]
        counts = Counter(labels)
        n_classes = 3
        weights = torch.zeros(n_classes)
        for idx, count in counts.items():
            weights[idx] = 1.0 / count
        assert (weights > 0).all()

    def test_split_ratios_sum_to_one(self):
        train, val, test = 0.75, 0.15, 0.10
        assert abs(train + val + test - 1.0) < 1e-6
