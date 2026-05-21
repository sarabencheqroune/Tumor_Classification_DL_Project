"""
main.py — Entry point for the Clinical Decision Support System.

Usage:
    python -m src.main --image path/to/mri.jpg --patient_id ANON_001

Interactive mode (default): prompts the radiologist at the terminal.
Test mode (--non_interactive): auto-approves HITL for automated demos.
"""
import argparse
import sys
from pathlib import Path

from src.agents.clinical_report_agent import ClinicalReportAgent
from src.agents.image_classifier_agent import ImageClassifierAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.config import CONFIDENCE_THRESHOLD, MODEL_PATH, VECTOR_STORE_PATH
from src.hitl.review_interface import HumanInTheLoopInterface
from src.logger import get_logger
from src.tools.rag_retrieval_tool import RAGRetrievalTool
from src.tools.report_formatter_tool import ReportFormatterTool


def build_pipeline(interactive: bool = True, output_dir: str = "demo/expected_outputs"):
    """Instantiate all agents and return the orchestrator."""
    run_id = None  # auto-generated per run
    logger = get_logger("ClinicalDSS", run_id)

    logger.info("Initialising Clinical Decision Support System")

    # Agent 1
    classifier = ImageClassifierAgent(
        model_path=str(MODEL_PATH),
        logger=logger,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # RAG tool
    rag = RAGRetrievalTool(vector_store_path=str(VECTOR_STORE_PATH))

    # Agent 2
    formatter = ReportFormatterTool()
    reporter  = ClinicalReportAgent(rag_tool=rag, logger=logger, formatter=formatter)

    # HITL
    hitl = HumanInTheLoopInterface(logger=logger, interactive=interactive)

    # Orchestrator
    orchestrator = OrchestratorAgent(
        classifier_agent=classifier,
        report_agent=reporter,
        hitl=hitl,
        logger=logger,
        output_dir=output_dir,
    )

    return orchestrator, logger


def main():
    parser = argparse.ArgumentParser(description="Clinical Decision Support System — MRI Brain Tumor Diagnosis")
    parser.add_argument("--image",           required=True,  help="Path to MRI image (JPG/PNG)")
    parser.add_argument("--patient_id",      default="ANONYMOUS", help="Patient identifier")
    parser.add_argument("--non_interactive", action="store_true",  help="Auto-approve HITL (for demos/tests)")
    parser.add_argument("--output_dir",      default="demo/expected_outputs", help="Directory for saved reports")
    args = parser.parse_args()

    orchestrator, logger = build_pipeline(
        interactive=not args.non_interactive,
        output_dir=args.output_dir,
    )

    print(f"\nProcessing MRI: {args.image}")
    print(f"Patient ID    : {args.patient_id}\n")

    result = orchestrator.process_mri(
        image_path=args.image,
        patient_id=args.patient_id,
    )

    # ------------------------------------------------------------------ #
    # Display result                                                        #
    # ------------------------------------------------------------------ #
    if result["status"] == "success":
        print("\n" + result.get("text_report", ""))
        if result.get("pdf_path"):
            print(f"\nPDF report saved → {result['pdf_path']}")
        if result.get("json_path"):
            print(f"JSON report saved → {result['json_path']}")

    elif result["status"] == "rejected":
        print("\n[PIPELINE STOPPED]")
        print(f"Reason : {result['reason']}")
        print(f"Notes  : {result.get('radiologist_notes', '')}")

    else:
        print(f"\n[ERROR] {result.get('message', 'Unknown error')}")
        logger.error("Pipeline ended with error", **{k: v for k, v in result.items() if k != "audit_log"})
        sys.exit(1)


if __name__ == "__main__":
    main()
