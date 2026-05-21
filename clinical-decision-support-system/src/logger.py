"""
JSON-structured logger for the Clinical Decision Support System.

Every agent action, model prediction, and HITL decision is logged
with an ISO-8601 timestamp for full audit-trail traceability.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import LOG_DIR, RUNS_DIR, LOG_LEVEL


class JSONLogger:
    """
    Writes structured JSON log entries to both a rolling run file and stdout.
    """

    def __init__(self, name: str, run_id: Optional[str] = None):
        self.name = name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_path = RUNS_DIR / f"{self.run_id}_run.json"
        self._entries: list[dict] = []

        # Standard Python logger for console output
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def info(self, message: str, **extra: Any) -> None:
        self._log("INFO", message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        self._log("WARNING", message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        self._log("ERROR", message, **extra)

    def debug(self, message: str, **extra: Any) -> None:
        self._log("DEBUG", message, **extra)

    def get_run_log(self) -> list[dict]:
        """Return all entries collected in this run."""
        return list(self._entries)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _log(self, level: str, message: str, **extra: Any) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "run_id": self.run_id,
            "message": message,
            **extra,
        }
        self._entries.append(entry)
        self._flush()

        # Mirror to console
        log_fn = getattr(self._logger, level.lower(), self._logger.info)
        log_fn(message)

    def _flush(self) -> None:
        """Persist current entries to the run JSON file."""
        try:
            with open(self.run_log_path, "w") as fh:
                json.dump(self._entries, fh, indent=2, default=str)
        except Exception:
            pass  # Never let logging crash the main pipeline


def get_logger(name: str, run_id: Optional[str] = None) -> JSONLogger:
    """Factory — returns a JSONLogger with the given name."""
    return JSONLogger(name, run_id)
