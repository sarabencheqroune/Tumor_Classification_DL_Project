"""
Unit tests for CNN model and inference tool.

Tests run without a trained checkpoint by building a random-weight model.
"""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import CLASS_NAMES, NUM_CLASSES
from src.models.cnn_model import build_resnet18
from src.tools.cnn_inference_tool import CNNInferenceTool


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def create_dummy_checkpoint(path: str):
    """Save a freshly initialised (random) ResNet18 state dict to path."""
    model = build_resnet18(pretrained=False)
    torch.save(model.state_dict(), path)


def create_test_image(path: str, size=(256, 256)):
    img = Image.fromarray(
        np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    )
    img.save(path)


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def dummy_model_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    p = str(d / "resnet18_test.pt")
    create_dummy_checkpoint(p)
    return p


@pytest.fixture(scope="module")
def test_image_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgs")
    p = str(d / "test_mri.jpg")
    create_test_image(p)
    return p


# --------------------------------------------------------------------------- #
#  ResNet18 model tests                                                         #
# --------------------------------------------------------------------------- #

class TestBuildResnet18:

    def test_model_has_correct_output_size(self):
        model = build_resnet18(pretrained=False)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, NUM_CLASSES)

    def test_model_fc_layer_is_replaced(self):
        model = build_resnet18(pretrained=False)
        assert model.fc.out_features == NUM_CLASSES


# --------------------------------------------------------------------------- #
#  CNNInferenceTool tests                                                       #
# --------------------------------------------------------------------------- #

class TestCNNInferenceTool:

    def test_successful_classification(self, dummy_model_path, test_image_path):
        tool   = CNNInferenceTool(model_path=dummy_model_path, confidence_threshold=0.80)
        result = tool.classify_mri(test_image_path)

        assert result["status"] == "success"
        assert result["prediction"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0
        assert "all_scores" in result
        assert len(result["all_scores"]) == NUM_CLASSES
        assert "needs_human_review" in result

    def test_all_scores_sum_to_one(self, dummy_model_path, test_image_path):
        tool   = CNNInferenceTool(model_path=dummy_model_path)
        result = tool.classify_mri(test_image_path)
        total  = sum(result["all_scores"].values())
        assert abs(total - 1.0) < 1e-4

    def test_nonexistent_image_returns_error(self, dummy_model_path):
        tool   = CNNInferenceTool(model_path=dummy_model_path)
        result = tool.classify_mri("/does/not/exist.jpg")
        assert result["status"] == "error"
        assert result["needs_human_review"] is True

    def test_low_confidence_flags_human_review(self, dummy_model_path, test_image_path):
        # Set threshold to 1.01 so every prediction triggers review
        tool   = CNNInferenceTool(model_path=dummy_model_path, confidence_threshold=1.01)
        result = tool.classify_mri(test_image_path)
        assert result["needs_human_review"] is True

    def test_high_threshold_never_exceeded(self, dummy_model_path, test_image_path):
        """With threshold=0 all predictions should pass without triggering review."""
        tool   = CNNInferenceTool(model_path=dummy_model_path, confidence_threshold=0.0)
        result = tool.classify_mri(test_image_path)
        assert result["needs_human_review"] is False

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            CNNInferenceTool(model_path=str(tmp_path / "missing.pt"))
