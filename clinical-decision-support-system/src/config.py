"""
Configuration and constants for the Clinical Decision Support System.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))
RUNS_DIR = LOG_DIR / "runs"

# Model configuration
MODEL_PATH = Path(os.getenv("MODEL_PATH", SRC_DIR / "checkpoints" / "resnet18_braintumor.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.80))
IMAGE_SIZE = 224
NUM_CLASSES = 4
# Must match ImageFolder alphabetical order of dataset folders:
# glioma(0) meningioma(1) notumor(2) pituitary(3)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Human-readable display names
CLASS_DISPLAY = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "notumor":    "No Tumor",
    "pituitary":  "Pituitary Adenoma",
}

# Image normalization (ImageNet stats used for ResNet18 pretrained weights)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# RAG configuration
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", SRC_DIR / "vector_store"))
GUIDELINES_PATH   = Path(os.getenv("GUIDELINES_PATH",   SRC_DIR / "data" / "medical_guidelines.txt"))
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
RAG_TOP_K         = 3

# HITL configuration
HITL_LOG_PATH = Path(os.getenv("HITL_LOG_PATH", LOG_DIR / "hitl_decisions.json"))

# Gemini (LLM) - optional for extended report narratives
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure runtime directories exist
for d in [LOG_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
