"""
Orchestrator Agent — coordinates the full clinical decision-support pipeline.

Pipeline:
  1. Receive MRI image path
  2. Agent 1 (ImageClassifierAgent) → CNN prediction
  3. Confidence check → if < threshold OR CNN failed → mandatory HITL
  4. HITL checkpoint (radiologist approve / reject / override)
  5. If rejected → return rejection result (no report)
  6. Agent 2 (ClinicalReportAgent) → structured report + optional PDF
  7. Return final result with full audit trail

All steps are logged as JSON with timestamps.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.agents.clinical_report_agent import ClinicalReportAgent
from src.agents.image_classifier_agent import ImageClassifierAgent
from src.config import CONFIDENCE_THRESHOLD
from src.hitl.review_interface import HumanInTheLoopInterface
from src.logger import JSONLogger


class OrchestratorAgent(BaseAgent):
    """
    Top-level controller.  Owns the pipeline and error-handling strategy.
    """

    def __init__(
        self,
        classifier_agent: ImageClassifierAgent,
        report_agent: ClinicalReportAgent,
        hitl: HumanInTheLoopInterface,
        logger: JSONLogger,
        output_dir: Optional[str] = None,
    ):
        super().__init__(name="OrchestratorAgent", logger=logger)
        self.classifier_agent = classifier_agent
        self.report_agent     = report_agent
        self.hitl             = hitl
        self.output_dir       = output_dir

    # ------------------------------------------------------------------ #

    def process_mri(
        self,
        image_path: str,
        patient_id: str = "ANONYMOUS",
    ) -> dict:
        """
        Run the full pipeline from MRI upload to final report.

        Args:
            image_path: path to JPEG/PNG MRI scan
            patient_id: anonymised patient identifier

        Returns:
            {
                "status": "success" | "rejected" | "error",
                "run_id": "...",
                "classification": {...},
                "hitl_decision": {...},
                "report": {...},
                "text_report": "...",
                "pdf_path": "...",
                "json_path": "...",
                "audit_log": [...]
            }
        """
        run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_log = []

        self.logger.info(
            "Pipeline started",
            run_id=run_id,
            patient_id=patient_id,
            image_path=image_path,
        )

        # ============================================================== #
        # STEP 1 — Image Classification (Agent 1)                         #
        # ============================================================== #
        self.logger.info("Step 1: Image classification", run_id=run_id)

        classification = self.classifier_agent.classify(image_path)
        audit_log.append({"step": "classification", "result": classification})

        if classification["status"] != "success":
            self.logger.error(
                f"Pipeline aborted: classification failed — {classification.get('message')}",
                run_id=run_id,
            )
            return {
                "status": "error",
                "run_id": run_id,
                "message": f"Classification failed: {classification.get('message')}",
                "audit_log": audit_log,
            }

        # ============================================================== #
        # STEP 2 — HITL Checkpoint (always; triggered if low confidence)  #
        # ============================================================== #
        self.logger.info(
            "Step 2: HITL checkpoint",
            run_id=run_id,
            low_confidence=classification.get("needs_human_review"),
        )

        hitl_result = self.hitl.request_review(image_path, classification)
        audit_log.append({"step": "hitl_review", "result": hitl_result})

        if not hitl_result.get("approved"):
            self.logger.warning(
                "Pipeline stopped: radiologist rejected classification",
                run_id=run_id,
            )
            return {
                "status": "rejected",
                "run_id": run_id,
                "reason": "Radiologist rejected the CNN classification.",
                "radiologist_notes": hitl_result.get("radiologist_notes", ""),
                "classification": classification,
                "hitl_decision": hitl_result,
                "audit_log": audit_log,
            }

        # Apply override if radiologist corrected the class
        if hitl_result.get("override_class"):
            original = classification["prediction"]
            classification["prediction"] = hitl_result["override_class"]
            self.logger.info(
                f"Prediction overridden: {original} → {classification['prediction']}",
                run_id=run_id,
            )
            audit_log.append({
                "step": "prediction_override",
                "original": original,
                "new": classification["prediction"],
            })

        # ============================================================== #
        # STEP 3 — Clinical Report Generation (Agent 2)                   #
        # ============================================================== #
        self.logger.info("Step 3: Clinical report generation", run_id=run_id)

        report_result = self.report_agent.generate_report(
            approved_prediction=classification,
            patient_id=patient_id,
            run_id=run_id,
            radiologist_notes=hitl_result.get("radiologist_notes", ""),
            radiologist_approval=hitl_result.get("approval_label", "Approved"),
            output_dir=self.output_dir,
        )
        audit_log.append({"step": "report_generation", "status": report_result["status"]})

        if report_result["status"] != "success":
            self.logger.error("Report generation failed", run_id=run_id)
            return {
                "status": "error",
                "run_id": run_id,
                "message": report_result.get("message", "Report generation failed"),
                "classification": classification,
                "hitl_decision": hitl_result,
                "audit_log": audit_log,
            }

        # ============================================================== #
        # DONE                                                             #
        # ============================================================== #
        self.logger.info("Pipeline completed successfully", run_id=run_id)

        return {
            "status": "success",
            "run_id": run_id,
            "classification": classification,
            "hitl_decision": hitl_result,
            "report": report_result["report"],
            "text_report": report_result.get("text_report", ""),
            "pdf_path": report_result.get("pdf_path"),
            "json_path": report_result.get("json_path"),
            "audit_log": audit_log,
        }
