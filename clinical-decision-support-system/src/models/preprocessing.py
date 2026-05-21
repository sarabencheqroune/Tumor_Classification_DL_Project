"""
MRI image preprocessing pipeline.

Handles loading, validation, resizing, and normalization so every
tensor passed to the CNN has identical shape and statistics.
"""
from pathlib import Path
from typing import Union

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


class MRIPreprocessor:
    """
    Loads a JPEG/PNG MRI scan and converts it to a normalised float tensor
    ready for ResNet18 inference.

    Output tensor shape: (3, IMAGE_SIZE, IMAGE_SIZE)
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, image_size: int = IMAGE_SIZE):
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ]
        )

    def preprocess(self, image_path: Union[str, Path]) -> dict:
        """
        Load and preprocess an MRI image file.

        Returns:
            {"status": "success", "tensor": torch.Tensor}
            {"status": "error",   "message": str}
        """
        path = Path(image_path)

        # --- Validation ---
        if not path.exists():
            return {"status": "error", "message": f"File not found: {image_path}"}

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return {
                "status": "error",
                "message": (
                    f"Unsupported file type '{path.suffix}'. "
                    f"Supported: {self.SUPPORTED_EXTENSIONS}"
                ),
            }

        try:
            img = Image.open(path).convert("RGB")
        except UnidentifiedImageError:
            return {
                "status": "error",
                "message": f"Cannot identify image file (corrupted?): {image_path}",
            }
        except Exception as exc:
            return {"status": "error", "message": f"Image open error: {exc}"}

        try:
            tensor = self.transform(img)
        except Exception as exc:
            return {"status": "error", "message": f"Transform error: {exc}"}

        return {"status": "success", "tensor": tensor}
