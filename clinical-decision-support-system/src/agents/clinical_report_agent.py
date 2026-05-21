"""
Agent 2 — Clinical Report Agent

Responsibilities:
  - Accept an APPROVED classification result (post-HITL)
  - Retrieve relevant medical guidelines via RAG
  - Fill the structured report template
  - Optionally export PDF
  - Log every action

This agent only runs AFTER the radiologist has approved the classification.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.base_agent import BaseAgent
from src.logger import JSONLogger
from src.tools.rag_retrieval_tool import RAGRetrievalTool
from src.tools.report_formatter_tool import ReportFormatterTool


# Static fallback summaries / recommendations if RAG fails
_SUMMARIES = {
    "glioma": (
        "Glioma detected. This is a primary brain tumor arising from glial cells. "
        "High-grade lesions are fast-growing and require urgent neurosurgical assessment "
        "and likely multimodal treatment (surgery, radiation, chemotherapy)."
    ),
    "meningioma": (
        "Meningioma detected. Typically an extra-axial, slow-growing benign tumor. "
        "Assessment of surgical candidacy, symptom burden, and proximity to critical "
        "structures is required before deciding between observation and intervention."
    ),
    "pituitary": (
        "Pituitary adenoma detected. Endocrine workup, visual field testing, and "
        "neurosurgery or endocrinology consultation are recommended depending on "
        "tumor size and functional status."
    ),
    "notumor": (
        "No intracranial neoplasm identified. Findings are within normal limits. "
        "Clinical correlation with presenting symptoms is recommended."
    ),
}

_RECOMMENDATIONS = {
    "glioma": [
        "URGENT neurosurgery consultation",
        "MRI with gadolinium contrast + DTI/PWI for surgical planning",
        "Neuro-oncology consultation for adjuvant therapy planning",
        "Molecular profiling (IDH, MGMT) for prognosis",
        "Consider enrollment in clinical trials",
    ],
    "meningioma": [
        "Neurosurgery consultation",
        "CT to evaluate calcification and bone involvement",
        "MR or CT angiography if adjacent to major venous sinuses",
        "Ophthalmology referral if near optic chiasm",
        "Follow-up MRI at 3-6 months if watchful waiting is chosen",
    ],
    "pituitary": [
        "Endocrinology consultation",
        "Full hormone panel (prolactin, IGF-1, cortisol, TSH, LH/FSH)",
        "Formal visual field testing (Humphrey perimetry)",
        "Dedicated pituitary MRI protocol if not already done",
        "Neurosurgery referral if macro-adenoma or causing visual symptoms",
    ],
    "notumor": [
        "Reassure patient — no intracranial neoplasm identified",
        "Clinical correlation with presenting symptoms",
        "Consider alternative diagnoses (migraine, demyelination, vascular)",
        "Repeat imaging only if symptoms progress or new deficits emerge",
    ],
}


class ClinicalReportAgent(BaseAgent):
    """
    Generates a structured clinical report from an approved CNN prediction.
    Integrates RAG-retrieved medical guidelines into the report.
    Patient metadata (age, sex, comorbidities, symptoms) enriches the RAG
    query and adds personalised clinical considerations to the report.
    """

    def __init__(
        self,
        rag_tool: RAGRetrievalTool,
        logger: JSONLogger,
        formatter: Optional[ReportFormatterTool] = None,
    ):
        super().__init__(name="ClinicalReportAgent", logger=logger)
        self.rag_tool  = rag_tool
        self.formatter = formatter or ReportFormatterTool()

    # ------------------------------------------------------------------ #

    def generate_report(
        self,
        approved_prediction: dict,
        patient_id: str = "ANONYMOUS",
        run_id: Optional[str] = None,
        radiologist_notes: str = "",
        radiologist_approval: str = "Approved",
        output_dir: Optional[str] = None,
        patient_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Generate and optionally save a clinical report.

        Args:
            approved_prediction: result dict from ImageClassifierAgent (post-HITL)
            patient_id: anonymised patient identifier
            run_id: pipeline run ID for audit trail
            radiologist_notes: free-text notes from the HITL review
            radiologist_approval: "Approved" | "Approved with Override"
            output_dir: if provided, saves PDF and JSON to this directory

        Returns:
            {
                "status": "success",
                "report": {...},
                "text_report": "...",
                "pdf_path": "..." (if output_dir provided),
                "json_path": "..." (if output_dir provided)
            }
        """
        patient_metadata = patient_metadata or {}
        age          = patient_metadata.get("age")
        sex          = patient_metadata.get("sex")
        comorbidities = patient_metadata.get("comorbidities", [])
        symptoms     = patient_metadata.get("symptoms", [])

        self.logger.info(
            "ClinicalReportAgent: generating report",
            prediction=approved_prediction.get("prediction"),
            patient_id=patient_id,
            age=age,
            sex=sex,
            comorbidities=comorbidities,
        )

        tumor_type = approved_prediction.get("prediction", "unknown")
        confidence = approved_prediction.get("confidence", 0.0)

        # --- 1. Build metadata-enriched RAG query ---
        query_parts = [f"Clinical guidelines for {tumor_type} brain tumor diagnosis and treatment"]
        if age:
            query_parts.append(f"patient age {age} years")
        if sex == "M":
            query_parts.append("male patient")
        elif sex == "F":
            query_parts.append("female patient")
        if comorbidities:
            query_parts.append(f"with comorbidities: {', '.join(comorbidities)}")
        if symptoms:
            query_parts.append(f"presenting with {', '.join(symptoms)}")
        rag_query = " ".join(query_parts)

        self.logger.info(f"ClinicalReportAgent: RAG query — {rag_query}")

        rag_result = self._safe_run(
            self.rag_tool.retrieve_guidelines,
            rag_query,
            error_extra={},
        )

        if rag_result.get("status") == "success" and rag_result.get("guidelines"):
            guidelines_text = "\n\n".join(rag_result["guidelines"])
            self.logger.info(
                f"ClinicalReportAgent: RAG retrieved {rag_result['retrieved_count']} sections"
            )
        else:
            # Fallback to static text
            guidelines_text = (
                f"[RAG unavailable — static fallback]\n\n"
                + self._get_static_guidelines(tumor_type)
            )
            self.logger.warning("ClinicalReportAgent: RAG failed, using static fallback")

        # --- 2. Build report dict ---
        sex_display = {"M": "Male", "F": "Female"}.get(sex, "Not specified") if sex else "Not specified"
        patient_specific_notes = self._get_patient_specific_notes(tumor_type, patient_metadata)

        report = {
            "patient_id": patient_id,
            "patient_metadata": {
                "name":               patient_metadata.get("name", "Anonymous"),
                "age":                age,
                "sex":                sex_display,
                "date_of_birth":      patient_metadata.get("dob"),
                "comorbidities":      comorbidities,
                "presenting_symptoms": symptoms,
            },
            "run_id": run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            "timestamp": datetime.now().isoformat(),
            "rag_query": rag_query,
            "classification": {
                "tumor_type": tumor_type,
                "confidence": confidence,
                "all_predictions": approved_prediction.get("all_scores", {}),
            },
            "clinical_summary": _SUMMARIES.get(tumor_type, "See clinical guidelines."),
            "patient_specific_notes": patient_specific_notes,
            "recommended_next_steps": _RECOMMENDATIONS.get(tumor_type, []),
            "medical_guidelines": guidelines_text,
            "radiologist_approval": radiologist_approval,
            "radiologist_notes": radiologist_notes,
            "disclaimer": (
                "This report is generated by an AI-assisted clinical decision-support system. "
                "It is NOT a substitute for professional medical diagnosis. "
                "All findings must be confirmed by a qualified radiologist or physician."
            ),
        }

        # --- 3. Format text ---
        text_report = self.formatter.format_text_report(report)

        result = self._success(report=report, text_report=text_report)

        # --- 4. Optionally save ---
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = out / f"{ts}_{patient_id}_report.json"
            pdf_path  = out / f"{ts}_{patient_id}_report.pdf"

            json_r = self.formatter.save_json_report(report, str(json_path))
            pdf_r  = self.formatter.save_pdf_report(report, str(pdf_path))

            result["json_path"] = json_r.get("path")
            result["pdf_path"]  = pdf_r.get("path")

            self.logger.info(
                f"ClinicalReportAgent: report saved",
                json_path=result["json_path"],
                pdf_path=result["pdf_path"],
            )

        self.logger.info("ClinicalReportAgent: report generation complete", patient_id=patient_id)
        return result

    # ------------------------------------------------------------------ #
    #  Fallback helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_static_guidelines(tumor_type: str) -> str:
        guidelines_dir = Path(__file__).resolve().parents[1] / "data" / "medical_guidelines.txt"
        if guidelines_dir.exists():
            text = guidelines_dir.read_text()
            sections = [s.strip() for s in text.split("---")]
            for section in sections:
                if tumor_type.upper() in section.upper()[:30]:
                    return section
        return f"Clinical guidelines for {tumor_type} — please consult standard neuro-oncology references."

    @staticmethod
    def _get_patient_specific_notes(tumor_type: str, patient_metadata: dict) -> str:
        """
        Generate patient-specific clinical considerations based on demographics,
        comorbidities, and presenting symptoms.
        """
        notes = []
        age           = patient_metadata.get("age")
        sex           = patient_metadata.get("sex")
        comorbidities = patient_metadata.get("comorbidities", [])
        symptoms      = patient_metadata.get("symptoms", [])

        # Age-specific considerations
        if age:
            if tumor_type == "glioma":
                if age < 40:
                    notes.append(
                        "Young patient (<40 yrs): IDH-mutant glioma is more likely — "
                        "favourable prognosis. Molecular profiling (IDH, 1p/19q, MGMT) is essential."
                    )
                elif age > 65:
                    notes.append(
                        "Elderly patient (>65 yrs): treatment tolerance may be reduced. "
                        "Consider hypofractionated radiotherapy ± temozolomide (NORDIC/NOA-08 regimens) "
                        "and geriatric oncology assessment."
                    )
            if tumor_type == "pituitary" and age > 60:
                notes.append(
                    "Older patient: non-functional macro-adenomas are common in this age group. "
                    "Conservative management with serial MRI is often preferred if asymptomatic."
                )

        # Sex-specific considerations
        if sex == "F":
            if tumor_type == "meningioma":
                notes.append(
                    "Female patient: meningiomas are 2× more common in women and may express "
                    "progesterone receptors. Avoid exogenous progesterone/HRT. "
                    "Progesterone-receptor status should be assessed at histology."
                )
            if tumor_type == "pituitary":
                notes.append(
                    "Female patient: rule out prolactinoma as a cause of menstrual irregularities "
                    "or galactorrhoea. Serum prolactin is mandatory."
                )

        # Comorbidity-specific considerations
        if "anticoagulation" in comorbidities:
            notes.append(
                "Anticoagulation: elevated surgical haemorrhagic risk. "
                "Perioperative bridging protocol required; involve haematology. "
                "Review indication for anticoagulation before holding therapy."
            )
        if "diabetes" in comorbidities:
            notes.append(
                "Diabetes mellitus: corticosteroid use (e.g., dexamethasone for cerebral oedema) "
                "will cause significant hyperglycaemia. Close glucose monitoring and insulin "
                "dose adjustment are required."
            )
        if "renal" in comorbidities:
            notes.append(
                "Renal impairment: gadolinium contrast agents carry risk of nephrogenic systemic "
                "fibrosis if GFR <30 mL/min. Check current GFR before contrast MRI; "
                "consider pre-hydration and macrocyclic agents."
            )
        if "cardiac" in comorbidities:
            notes.append(
                "Cardiac disease: formal cardiology pre-operative risk assessment required. "
                "Neurosurgical procedures carry moderate-to-high cardiovascular risk; "
                "optimise cardiac status before elective surgery."
            )
        if "hypertension" in comorbidities:
            notes.append(
                "Hypertension: maintain strict blood-pressure control peri-operatively "
                "(target <140/90 mmHg). Uncontrolled hypertension increases intracranial "
                "haemorrhage risk and may confound MRI perfusion sequences."
            )

        # Symptom-specific considerations
        if "seizure" in symptoms and tumor_type != "notumor":
            notes.append(
                "Seizure presentation: initiate anti-epileptic therapy — levetiracetam "
                "(Keppra) is preferred (no hepatic enzyme induction, no interaction with "
                "chemotherapy). EEG and neurology review recommended."
            )
        if "vision loss" in symptoms and tumor_type in ("pituitary", "meningioma"):
            notes.append(
                "Visual symptoms: urgent ophthalmology referral for formal visual field "
                "testing (Humphrey perimetry). Optic pathway compression may require "
                "time-critical surgical decompression."
            )
        if "headache" in symptoms:
            notes.append(
                "Headache: assess for raised intracranial pressure (papilloedema, "
                "morning predominance, positional worsening). Consider dexamethasone "
                "if significant perilesional oedema is present on imaging."
            )
        if "cognitive" in symptoms:
            notes.append(
                "Cognitive changes: baseline neuropsychological assessment recommended "
                "before treatment. Cognitive impairment may influence consent capacity "
                "and treatment goals discussions."
            )
        if "weakness" in symptoms:
            notes.append(
                "Motor weakness: document laterality and onset. "
                "Urgent physiotherapy referral and falls-risk assessment. "
                "Consider corticosteroids if acute deterioration due to oedema."
            )

        if not notes:
            return ""
        return "\n".join(f"• {n}" for n in notes)
