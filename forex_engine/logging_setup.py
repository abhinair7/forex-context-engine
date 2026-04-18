"""Structured JSON logging with a dedicated rejection sink.

Two logical streams:

- ``audit`` — every decision point on the happy path (Node entered,
  Node produced N signals, conflict detected, hypothesis ranked, ...).
- ``rejections`` — one line per rejected payload with full context,
  written to a separate file so compliance can tail it without noise.

Both use JSON-per-line so downstream tools (Splunk, Datadog) ingest
them without custom parsers.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from .config import EngineConfig
from .time_utils import now_est

_AUDIT_LOGGER_NAME = "forex.audit"
_REJECTION_LOGGER_NAME = "forex.rejection"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):            # Pydantic v2 models
        return obj.model_dump(mode="json")
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": now_est().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=_json_default, separators=(",", ":"))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def configure_logging(cfg: EngineConfig) -> None:
    """Idempotent: safe to call many times (e.g. in tests)."""
    _ensure_parent(cfg.audit_log_path)
    _ensure_parent(cfg.rejection_log_path)

    formatter = _JsonFormatter()

    audit = logging.getLogger(_AUDIT_LOGGER_NAME)
    audit.setLevel(logging.INFO)
    audit.handlers.clear()
    audit_file = logging.FileHandler(cfg.audit_log_path, encoding="utf-8")
    audit_file.setFormatter(formatter)
    audit_stream = logging.StreamHandler(sys.stderr)
    audit_stream.setFormatter(formatter)
    audit.addHandler(audit_file)
    audit.addHandler(audit_stream)
    audit.propagate = False

    reject = logging.getLogger(_REJECTION_LOGGER_NAME)
    reject.setLevel(logging.WARNING)
    reject.handlers.clear()
    reject_file = logging.FileHandler(cfg.rejection_log_path, encoding="utf-8")
    reject_file.setFormatter(formatter)
    reject.addHandler(reject_file)
    reject.propagate = False


def audit(message: str, **fields: Any) -> None:
    logging.getLogger(_AUDIT_LOGGER_NAME).info(
        message, extra={"extra_fields": fields}
    )


def reject(message: str, **fields: Any) -> None:
    """Record a rejected payload. Always includes ``reason`` and ``source``."""
    logging.getLogger(_REJECTION_LOGGER_NAME).warning(
        message, extra={"extra_fields": fields}
    )
