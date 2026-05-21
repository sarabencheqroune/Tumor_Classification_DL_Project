"""
CNN Inference Tool — CrewAI-compatible wrapper around BrainTumorInference.

The tool accepts an image path (str) and returns a structured result dict.
Agent 1 calls this as its primary tool.
"""
from pathlib import Path
from typing import Optional

from src.config import CONFIDENCE_THRESHOLD, MODEL_PATH
from src.models.inference import BrainTumorInference


class CNNInferenceTool:
    """
    Wraps ResNet18 inference for use inside the ImageClassifierAgent.

    Adds confidence-threshold logic so the agent knows whether to flag
    the case for mandatory human review.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        mp = str(model_path or MODEL_PATH)
        self.engine = BrainTumorInference(model_path=mp, device=device)

    # ------------------------------------------------------------------ #

    def classify_mri(self, image_path: str) -> dict:
        """
        Classify a brain MRI image.

        Args:
            image_path: absolute or relative path to a JPEG/PNG MRI scan

        Returns:
            {
                "status": "success",
                "prediction": "glioma",
                "confidence": 0.94,
                "all_scores": {"glioma": 0.94, ...},
                "needs_human_review": False
            }
            or
            {"status": "error", "message": "...", "needs_human_review": True}
        """
        if not image_path or not Path(image_path).exists():
            return {
                "status": "error",
                "message": f"Image not found: {image_path}",
                "needs_human_review": True,
            }

        result = self.engine.predict(image_path)

        if result["status"] != "success":
            result["needs_human_review"] = True
            return result

        result["needs_human_review"] = result["confidence"] < self.confidence_threshold
        return result
