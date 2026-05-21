"""
Base class shared by all agents in the clinical decision-support pipeline.

Provides:
  - Structured logging via JSONLogger
  - A consistent result-dict schema helper
  - Error-wrapping so individual agents never crash the pipeline
"""
from datetime import datetime
from typing import Any, Optional

from src.logger import JSONLogger


class BaseAgent:
    """Abstract base for all clinical decision-support agents."""

    def __init__(self, name: str, logger: JSONLogger):
        self.name   = name
        self.logger = logger

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _success(self, **kwargs: Any) -> dict:
        return {"status": "success", "agent": self.name,
                "timestamp": datetime.now().isoformat(), **kwargs}

    def _error(self, message: str, **kwargs: Any) -> dict:
        self.logger.error(message, agent=self.name)
        return {"status": "error", "agent": self.name, "message": message,
                "timestamp": datetime.now().isoformat(), **kwargs}

    def _safe_run(self, fn, *args, error_extra: Optional[dict] = None, **kwargs):
        """
        Call fn(*args, **kwargs).  If it raises, return a structured error dict
        instead of propagating the exception.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            extra = error_extra or {}
            return self._error(str(exc), **extra)
