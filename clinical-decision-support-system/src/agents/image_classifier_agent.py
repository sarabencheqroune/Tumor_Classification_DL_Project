"""
Agent 1 — Image Classifier Agent

Responsibilities:
  - Accept an MRI image path
  - Run the CNN inference tool (ResNet18)
  - Apply the confidence threshold
  - Return prediction + confidence + needs_human_review flag
  - Log every action with timestamp

If confidence < CONFIDENCE_THRESHOLD (default 0.80), the result is flagged
for mandatory human review before report generation can proceed.
If the CNN itself fails for any reason, the case is also escalated to HITL.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.config import CONFIDENCE_THRESHOLD
from src.logger import JSONLogger
from src.tools.cnn_inference_tool import CNNInferenceTool
from src.tools.error_handlers import safe_image_path


class ImageClassifierAgent(BaseAgent):
    """
    Classifies brain MRI images using a fine-tuned ResNet18.

    Output classes: glioma | meningioma | pituitary | no_tumor
    """

    def __init__(
        self,
        model_path: str,
        logger: JSONLogger,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
    ):
        super().__init__(name="ImageClassifierAgent", logger=logger)
        self.confidence_threshold = confidence_threshold
        self.cnn_tool = CNNInferenceTool(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    # ------------------------------------------------------------------ #

    def classify(self, image_path: str) -> dict:
        """
        Main entry point.

        Args:
            image_path: path to a JPEG/PNG MRI scan

        Returns:
            {
                "status": "success",
                "agent": "ImageClassifierAgent",
                "timestamp": "...",
                "image_path": "...",
                "prediction": "glioma",
                "confidence": 0.94,
                "all_scores": {...},
                "needs_human_review": False
            }
        """
        self.logger.info(
            "ImageClassifierAgent: classification started",
            image_path=image_path,
        )

        # --- 1. Validate image path ---
        valid, err_msg = safe_image_path(image_path)
        if not valid:
            self.logger.error(
                f"ImageClassifierAgent: invalid image — {err_msg}",
                image_path=image_path,
            )
            return {
                "status": "error",
                "agent": self.name,
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path,
                "message": err_msg,
                "needs_human_review": True,
            }

        # --- 2. Run CNN ---
        result = self._safe_run(
            self.cnn_tool.classify_mri,
            image_path,
            error_extra={"image_path": image_path, "needs_human_review": True},
        )

        if result["status"] != "success":
            self.logger.error(
                f"ImageClassifierAgent: CNN failed — {result.get('message')}",
                image_path=image_path,
            )
            result["agent"] = self.name
            result["image_path"] = image_path
            return result

        # --- 3. Log result ---
        flagged = result["needs_human_review"]
        self.logger.info(
            f"ImageClassifierAgent: classified as '{result['prediction']}' "
            f"(confidence={result['confidence']:.2%}, "
            f"needs_review={flagged})",
            prediction=result["prediction"],
            confidence=result["confidence"],
            needs_human_review=flagged,
            image_path=image_path,
        )

        return {
            **result,
            "agent": self.name,
            "image_path": image_path,
            "timestamp": datetime.now().isoformat(),
        }
