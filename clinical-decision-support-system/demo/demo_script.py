"""
demo_script.py — Automated demo runner for the Clinical Decision Support System.

Runs all 4 sample MRI classes through the pipeline in non-interactive mode
(auto-approves HITL) and prints the generated reports.

Usage:
    python demo/demo_script.py

Requires:
    - Trained model at src/checkpoints/resnet18_braintumor.pt
    - RAG index built (run notebooks/04_rag_setup.ipynb or call build_index())
    - Sample MRI images in demo/sample_mris/
"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.clinical_report_agent import ClinicalReportAgent
from src.agents.image_classifier_agent import ImageClassifierAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.config import CONFIDENCE_THRESHOLD, MODEL_PATH, VECTOR_STORE_PATH
from src.hitl.review_interface import HumanInTheLoopInterface
from src.logger import get_logger
from src.tools.rag_retrieval_tool import RAGRetrievalTool
from src.tools.report_formatter_tool import ReportFormatterTool


SAMPLE_IMAGES = {
    "glioma_01.jpg":     "DEMO_GLIOMA",
    "meningioma_01.jpg": "DEMO_MENINGIOMA",
    "pituitary_01.jpg":  "DEMO_PITUITARY",
    "notumor_01.jpg":    "DEMO_NOTUMOR",
}

OUTPUT_DIR = Path(__file__).parent / "expected_outputs"
IMAGES_DIR = Path(__file__).parent / "sample_mris"


def run_demo():
    logger    = get_logger("Demo")
    rag       = RAGRetrievalTool(vector_store_path=str(VECTOR_STORE_PATH))
    formatter = ReportFormatterTool()
    reporter  = ClinicalReportAgent(rag_tool=rag, logger=logger, formatter=formatter)
    hitl      = HumanInTheLoopInterface(logger=logger, interactive=False)

    try:
        classifier = ImageClassifierAgent(
            model_path=str(MODEL_PATH),
            logger=logger,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("Please train the model first using notebooks/02_model_training.ipynb")
        sys.exit(1)

    orchestrator = OrchestratorAgent(
        classifier_agent=classifier,
        report_agent=reporter,
        hitl=hitl,
        logger=logger,
        output_dir=str(OUTPUT_DIR),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  CLINICAL DECISION SUPPORT SYSTEM — AUTOMATED DEMO")
    print("=" * 70)

    for image_name, patient_id in SAMPLE_IMAGES.items():
        image_path = IMAGES_DIR / image_name

        if not image_path.exists():
            print(f"\n[SKIP] Sample image not found: {image_path}")
            print("       Copy sample MRI images from the dataset into demo/sample_mris/")
            continue

        print(f"\n{'─'*70}")
        print(f"  Processing: {image_name}  (Patient: {patient_id})")
        print(f"{'─'*70}")

        start = time.time()
        result = orchestrator.process_mri(str(image_path), patient_id=patient_id)
        elapsed = time.time() - start

        if result["status"] == "success":
            cls    = result["classification"]
            print(f"  Prediction : {cls['prediction'].upper()}")
            print(f"  Confidence : {cls['confidence']:.2%}")
            print(f"  HITL       : {result['hitl_decision']['approval_label']}")
            print(f"  Elapsed    : {elapsed:.2f}s")
            if result.get("pdf_path"):
                print(f"  PDF saved  : {result['pdf_path']}")
        elif result["status"] == "rejected":
            print(f"  [REJECTED] {result['reason']}")
        else:
            print(f"  [ERROR] {result.get('message')}")

    print("\n" + "=" * 70)
    print("  Demo complete. Reports saved to:", OUTPUT_DIR)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
