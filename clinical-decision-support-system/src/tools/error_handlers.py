"""
Centralised error-handling utilities shared across agents and tools.
"""
from pathlib import Path
from typing import Optional


def safe_image_path(image_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate that an image path exists and has a supported extension.

    Returns:
        (True, None)           if valid
        (False, error_message) if invalid
    """
    SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    p = Path(image_path)

    if not p.exists():
        return False, f"File not found: {image_path}"
    if p.suffix.lower() not in SUPPORTED:
        return False, f"Unsupported file type '{p.suffix}'. Supported: {SUPPORTED}"
    if p.stat().st_size == 0:
        return False, f"File is empty: {image_path}"
    return True, None


def wrap_tool_error(tool_name: str, exc: Exception) -> dict:
    """Return a standardised error dict for a tool exception."""
    return {
        "status": "error",
        "tool": tool_name,
        "message": str(exc),
        "needs_human_review": True,
    }


def validate_classification_result(result: dict) -> tuple[bool, Optional[str]]:
    """
    Ensure a classification result dict has the required keys and values.

    Returns (True, None) if valid, else (False, reason).
    """
    required = {"status", "prediction", "confidence"}
    valid_classes = {"glioma", "meningioma", "pituitary", "no_tumor"}

    missing = required - result.keys()
    if missing:
        return False, f"Missing keys in classification result: {missing}"

    if result["status"] != "success":
        return False, f"Classification status is '{result['status']}'"

    if result["prediction"] not in valid_classes:
        return False, f"Unknown prediction class: '{result['prediction']}'"

    conf = result.get("confidence", -1)
    if not (0.0 <= conf <= 1.0):
        return False, f"Confidence out of range [0,1]: {conf}"

    return True, None
