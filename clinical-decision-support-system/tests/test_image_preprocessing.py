"""
Unit tests for MRIPreprocessor.
"""
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.preprocessing import MRIPreprocessor


@pytest.fixture
def preprocessor():
    return MRIPreprocessor(image_size=224)


@pytest.fixture
def valid_rgb_image(tmp_path):
    """Create a valid RGB JPEG test image."""
    img = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    path = tmp_path / "valid.jpg"
    img.save(str(path))
    return str(path)


@pytest.fixture
def blank_image(tmp_path):
    """Create an all-black (blank) MRI-like image."""
    img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    path = tmp_path / "blank.jpg"
    img.save(str(path))
    return str(path)


class TestMRIPreprocessor:

    def test_valid_image_returns_success(self, preprocessor, valid_rgb_image):
        result = preprocessor.preprocess(valid_rgb_image)
        assert result["status"] == "success"
        assert "tensor" in result

    def test_output_tensor_shape(self, preprocessor, valid_rgb_image):
        result = preprocessor.preprocess(valid_rgb_image)
        tensor = result["tensor"]
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_tensor_values_are_normalised(self, preprocessor, valid_rgb_image):
        result = preprocessor.preprocess(valid_rgb_image)
        tensor = result["tensor"]
        # After ImageNet normalisation, values can be negative
        assert tensor.min().item() < 0 or tensor.max().item() > 1  # not in [0,1]

    def test_nonexistent_file_returns_error(self, preprocessor):
        result = preprocessor.preprocess("/nonexistent/path/mri.jpg")
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    def test_unsupported_extension_returns_error(self, preprocessor, tmp_path):
        path = tmp_path / "file.pdf"
        path.write_bytes(b"fake pdf content")
        result = preprocessor.preprocess(str(path))
        assert result["status"] == "error"

    def test_corrupted_file_returns_error(self, preprocessor, tmp_path):
        path = tmp_path / "corrupted.jpg"
        path.write_bytes(b"this is definitely not a jpeg")
        result = preprocessor.preprocess(str(path))
        assert result["status"] == "error"

    def test_blank_image_is_processed_successfully(self, preprocessor, blank_image):
        result = preprocessor.preprocess(blank_image)
        assert result["status"] == "success"

    def test_png_image_is_supported(self, preprocessor, tmp_path):
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        path = tmp_path / "mri.png"
        img.save(str(path))
        result = preprocessor.preprocess(str(path))
        assert result["status"] == "success"

    def test_grayscale_converted_to_rgb(self, preprocessor, tmp_path):
        """Grayscale MRI scans must be silently promoted to RGB."""
        img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L")
        path = tmp_path / "gray.jpg"
        img.save(str(path))
        result = preprocessor.preprocess(str(path))
        assert result["status"] == "success"
        assert result["tensor"].shape == (3, 224, 224)
