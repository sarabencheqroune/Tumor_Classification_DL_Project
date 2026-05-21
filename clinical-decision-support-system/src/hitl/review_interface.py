"""
Human-in-the-Loop (HITL) Review Interface

The radiologist sees:
  - MRI image path
  - CNN prediction + confidence
  - All class probability scores

Then decides:
  [A]pprove  — accept the CNN prediction and continue to report generation
  [R]eject   — stop the pipeline; no report is generated
  [O]verride — accept with correction (radiologist provides the correct class)

Every decision is timestamped and appended to a JSON audit log.

Supports two modes:
  interactive=True  — prompts the user at the terminal (default)
  interactive=False — auto-approves (for automated testing / CI)
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import CLASS_NAMES, HITL_LOG_PATH
from src.logger import JSONLogger


class HumanInTheLoopInterface:
    """Radiologist review checkpoint."""

    def __init__(
        self,
        logger: JSONLogger,
        log_path: Optional[str] = None,
        interactive: bool = True,
    ):
        self.logger      = logger
        self.log_path    = Path(log_path or HITL_LOG_PATH)
        self.interactive = interactive
        self._decisions: list[dict] = self._load_existing_log()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def request_review(self, image_path: str, classification: dict) -> dict:
        """
        Present classification to the radiologist and collect a decision.

        Returns:
            {
                "approved": True/False,
                "override_class": "meningioma" | None,
                "radiologist_notes": "...",
                "timestamp": "...",
                "decision_code": "A" | "R" | "O"
            }
        """
        self.logger.info(
            "HITL checkpoint triggered",
            prediction=classification.get("prediction"),
            confidence=classification.get("confidence"),
            low_confidence=classification.get("needs_human_review", False),
        )

        if self.interactive:
            return self._interactive_review(image_path, classification)
        else:
            return self._auto_approve(classification)

    # ------------------------------------------------------------------ #
    #  Interactive mode                                                    #
    # ------------------------------------------------------------------ #

    def _interactive_review(self, image_path: str, classification: dict) -> dict:
        self._print_review_panel(image_path, classification)

        while True:
            raw = input("\nDecision [A/R/O]: ").strip().upper()
            if raw in ("A", "R", "O"):
                break
            print("  Please enter A (Approve), R (Reject), or O (Override).")

        override_class = None
        if raw == "O":
            override_class = self._get_override_class()

        notes = ""
        if raw == "R":
            notes = input("Clinical notes (reason for rejection): ").strip()
        elif raw == "O" and override_class:
            notes = input("Clinical notes (optional): ").strip()

        return self._record_and_return(raw, classification, override_class, notes)

    def _print_review_panel(self, image_path: str, classification: dict) -> None:
        print("\n" + "=" * 70)
        print("  HUMAN-IN-THE-LOOP REVIEW  —  Radiologist Checkpoint")
        print("=" * 70)
        print(f"  Image      : {image_path}")
        print(f"\n  CNN Result :")
        print(f"    Predicted class : {classification.get('prediction', 'N/A').upper()}")
        print(f"    Confidence      : {classification.get('confidence', 0):.2%}")
        if classification.get("needs_human_review"):
            print("    *** LOW CONFIDENCE — mandatory human review ***")
        print("\n  All class scores:")
        for cls, score in classification.get("all_scores", {}).items():
            bar = "█" * int(score * 20)
            print(f"    {cls:<20} {score:.3f}  {bar}")
        print("\n  Options:")
        print("    [A] Approve  — accept CNN prediction, continue to report")
        print("    [R] Reject   — stop pipeline, no report generated")
        print("    [O] Override — correct the predicted class")

    def _get_override_class(self) -> Optional[str]:
        print(f"\n  Valid classes: {', '.join(CLASS_NAMES)}")
        while True:
            cls = input("  Enter correct class: ").strip().lower()
            if cls in CLASS_NAMES:
                return cls
            print(f"  Invalid class '{cls}'. Choose from: {CLASS_NAMES}")

    # ------------------------------------------------------------------ #
    #  Auto-approve mode (testing)                                         #
    # ------------------------------------------------------------------ #

    def _auto_approve(self, classification: dict) -> dict:
        self.logger.info("HITL: auto-approve mode (non-interactive)")
        return self._record_and_return("A", classification, None, "auto-approved (test mode)")

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #

    def _record_and_return(
        self,
        decision_code: str,
        classification: dict,
        override_class: Optional[str],
        notes: str,
    ) -> dict:
        approved = decision_code in ("A", "O")
        approval_label = {
            "A": "Approved",
            "R": "Rejected",
            "O": "Approved with Override",
        }[decision_code]

        record = {
            "timestamp": datetime.now().isoformat(),
            "decision_code": decision_code,
            "approved": approved,
            "approval_label": approval_label,
            "cnn_prediction": classification.get("prediction"),
            "cnn_confidence": classification.get("confidence"),
            "override_class": override_class,
            "radiologist_notes": notes,
        }

        self._decisions.append(record)
        self._save_log()

        self.logger.info(
            f"HITL decision: {approval_label}",
            decision_code=decision_code,
            override_class=override_class,
        )

        return {
            "approved": approved,
            "override_class": override_class,
            "radiologist_notes": notes,
            "approval_label": approval_label,
            "timestamp": record["timestamp"],
            "decision_code": decision_code,
        }

    def _save_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as fh:
            json.dump(self._decisions, fh, indent=2)

    def _load_existing_log(self) -> list[dict]:
        if self.log_path.exists():
            try:
                with open(self.log_path) as fh:
                    return json.load(fh)
            except Exception:
                pass
        return []
