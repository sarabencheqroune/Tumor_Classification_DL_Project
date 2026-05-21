"""
Integration tests for the agent pipeline.

- ImageClassifierAgent
- ClinicalReportAgent
- OrchestratorAgent (with mocked HITL)

All tests use a dummy ResNet18 checkpoint (random weights) so they run
without a real trained model.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.agents.clinical_report_agent import ClinicalReportAgent
from src.agents.image_classifier_agent import ImageClassifierAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.config import CLASS_NAMES
from src.hitl.review_interface import HumanInTheLoopInterface
from src.logger import get_logger
from src.models.cnn_model import build_resnet18
from src.tools.rag_retrieval_tool import RAGRetrievalTool
from src.tools.report_formatter_tool import ReportFormatterTool


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def dummy_model_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    p = d / "resnet18_test.pt"
    model = build_resnet18(pretrained=False)
    torch.save(model.state_dict(), str(p))
    return str(p)


@pytest.fixture(scope="module")
def test_image(tmp_path_factory):
    d = tmp_path_factory.mktemp("imgs")
    p = str(d / "mri.jpg")
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(p)
    return p


@pytest.fixture(scope="module")
def logger():
    return get_logger("TestSuite")


@pytest.fixture(scope="module")
def rag_tool(tmp_path_factory):
    store = tmp_path_factory.mktemp("chroma")
    tool  = RAGRetrievalTool(
        vector_store_path=str(store),
        guidelines_path="src/data/medical_guidelines.txt",
        top_k=2,
    )
    tool.build_index()
    return tool


# --------------------------------------------------------------------------- #
#  ImageClassifierAgent                                                        #
# --------------------------------------------------------------------------- #

class TestImageClassifierAgent:

    def test_classify_returns_success(self, dummy_model_path, test_image, logger):
        agent  = ImageClassifierAgent(dummy_model_path, logger, confidence_threshold=0.80)
        result = agent.classify(test_image)
        assert result["status"] == "success"
        assert result["prediction"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_missing_image(self, dummy_model_path, logger):
        agent  = ImageClassifierAgent(dummy_model_path, logger)
        result = agent.classify("/does/not/exist.jpg")
        assert result["status"] == "error"
        assert result["needs_human_review"] is True

    def test_classify_low_confidence_triggers_review(self, dummy_model_path, test_image, logger):
        agent  = ImageClassifierAgent(dummy_model_path, logger, confidence_threshold=1.01)
        result = agent.classify(test_image)
        assert result["needs_human_review"] is True


# --------------------------------------------------------------------------- #
#  ClinicalReportAgent                                                         #
# --------------------------------------------------------------------------- #

class TestClinicalReportAgent:

    def test_report_generated_for_glioma(self, rag_tool, logger):
        agent  = ClinicalReportAgent(rag_tool=rag_tool, logger=logger)
        pred   = {"prediction": "glioma", "confidence": 0.94, "all_scores": {c: 0.25 for c in CLASS_NAMES}}
        result = agent.generate_report(pred, patient_id="TEST_001")
        assert result["status"] == "success"
        assert result["report"]["classification"]["tumor_type"] == "glioma"

    def test_report_contains_all_fields(self, rag_tool, logger):
        agent  = ClinicalReportAgent(rag_tool=rag_tool, logger=logger)
        pred   = {"prediction": "no_tumor", "confidence": 0.99, "all_scores": {}}
        result = agent.generate_report(pred)
        report = result["report"]
        for key in ["patient_id", "timestamp", "classification", "clinical_summary",
                    "recommended_next_steps", "medical_guidelines", "disclaimer"]:
            assert key in report, f"Missing key: {key}"

    def test_text_report_is_string(self, rag_tool, logger):
        agent  = ClinicalReportAgent(rag_tool=rag_tool, logger=logger)
        pred   = {"prediction": "meningioma", "confidence": 0.85, "all_scores": {}}
        result = agent.generate_report(pred)
        assert isinstance(result.get("text_report"), str)
        assert "MENINGIOMA" in result["text_report"].upper()

    def test_report_saved_to_disk(self, rag_tool, logger, tmp_path):
        agent  = ClinicalReportAgent(rag_tool=rag_tool, logger=logger)
        pred   = {"prediction": "pituitary", "confidence": 0.91, "all_scores": {}}
        result = agent.generate_report(pred, output_dir=str(tmp_path))
        assert result.get("json_path") and Path(result["json_path"]).exists()


# --------------------------------------------------------------------------- #
#  OrchestratorAgent (end-to-end)                                              #
# --------------------------------------------------------------------------- #

class TestOrchestratorAgent:

    def _build_orchestrator(self, dummy_model_path, rag_tool, logger,
                             auto_approve=True, output_dir=None):
        classifier = ImageClassifierAgent(dummy_model_path, logger, confidence_threshold=0.80)
        reporter   = ClinicalReportAgent(rag_tool=rag_tool, logger=logger)
        hitl       = HumanInTheLoopInterface(logger=logger, interactive=False)
        return OrchestratorAgent(classifier, reporter, hitl, logger, output_dir=output_dir)

    def test_full_pipeline_success(self, dummy_model_path, rag_tool, logger, test_image):
        orch   = self._build_orchestrator(dummy_model_path, rag_tool, logger)
        result = orch.process_mri(test_image, patient_id="PATIENT_X")
        assert result["status"] == "success"
        assert "report" in result
        assert "classification" in result
        assert "hitl_decision" in result
        assert result["hitl_decision"]["approved"] is True

    def test_pipeline_rejects_when_hitl_rejects(self, dummy_model_path, rag_tool, logger, test_image):
        classifier = ImageClassifierAgent(dummy_model_path, logger)
        reporter   = ClinicalReportAgent(rag_tool=rag_tool, logger=logger)
        # Mock HITL to always reject
        hitl = MagicMock()
        hitl.request_review.return_value = {
            "approved": False,
            "override_class": None,
            "radiologist_notes": "Poor image quality",
            "approval_label": "Rejected",
            "timestamp": "2024-01-01T00:00:00",
            "decision_code": "R",
        }
        orch   = OrchestratorAgent(classifier, reporter, hitl, logger)
        result = orch.process_mri(test_image)
        assert result["status"] == "rejected"
        assert "reason" in result

    def test_pipeline_with_override(self, dummy_model_path, rag_tool, logger, test_image):
        classifier = ImageClassifierAgent(dummy_model_path, logger)
        reporter   = ClinicalReportAgent(rag_tool=rag_tool, logger=logger)
        hitl = MagicMock()
        hitl.request_review.return_value = {
            "approved": True,
            "override_class": "meningioma",
            "radiologist_notes": "Radiologist correction",
            "approval_label": "Approved with Override",
            "timestamp": "2024-01-01T00:00:00",
            "decision_code": "O",
        }
        orch   = OrchestratorAgent(classifier, reporter, hitl, logger)
        result = orch.process_mri(test_image)
        assert result["status"] == "success"
        assert result["classification"]["prediction"] == "meningioma"

    def test_pipeline_error_on_bad_image(self, dummy_model_path, rag_tool, logger):
        orch   = self._build_orchestrator(dummy_model_path, rag_tool, logger)
        result = orch.process_mri("/nonexistent/mri.jpg")
        assert result["status"] == "error"

    def test_pipeline_saves_reports(self, dummy_model_path, rag_tool, logger, test_image, tmp_path):
        orch   = self._build_orchestrator(dummy_model_path, rag_tool, logger, output_dir=str(tmp_path))
        result = orch.process_mri(test_image)
        assert result["status"] == "success"
        assert result.get("json_path") and Path(result["json_path"]).exists()

    def test_audit_log_populated(self, dummy_model_path, rag_tool, logger, test_image):
        orch   = self._build_orchestrator(dummy_model_path, rag_tool, logger)
        result = orch.process_mri(test_image)
        assert isinstance(result.get("audit_log"), list)
        assert len(result["audit_log"]) >= 3  # classification + hitl + report
