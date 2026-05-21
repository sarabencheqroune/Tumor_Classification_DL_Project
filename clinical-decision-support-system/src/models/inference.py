"""
Low-level inference wrapper around the trained ResNet18.

This module is intentionally thin — it only handles forward-pass
and softmax probability extraction. Business logic (confidence
thresholds, HITL flags) lives in the agent layer.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from src.config import CLASS_NAMES
from src.models.cnn_model import load_trained_model
from src.models.preprocessing import MRIPreprocessor


class BrainTumorInference:
    """
    Loads the trained model once and exposes a predict() method.
    """

    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model = load_trained_model(model_path, self.device)
        self.preprocessor = MRIPreprocessor()
        self.class_names = CLASS_NAMES

    def predict(self, image_path: str) -> dict:
        """
        Run inference on a single MRI image file.

        Returns:
            {
                "status": "success",
                "prediction": "glioma",
                "confidence": 0.94,
                "all_scores": {"glioma": 0.94, "meningioma": 0.03, ...}
            }
            or
            {"status": "error", "message": "..."}
        """
        prep = self.preprocessor.preprocess(image_path)
        if prep["status"] != "success":
            return prep  # propagate error

        tensor: torch.Tensor = prep["tensor"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        result = probs.max(dim=0)
        confidence = result.values.item()
        pred_idx   = result.indices.item()

        return {
            "status": "success",
            "prediction": self.class_names[pred_idx],
            "confidence": confidence,
            "all_scores": {
                self.class_names[i]: probs[i].item() for i in range(len(self.class_names))
            },
        }
