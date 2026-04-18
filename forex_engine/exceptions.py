"""Typed exceptions for the Forex Context Engine.

Any caller can react to a specific failure mode without string-matching.
Every exception carries an auditable ``context`` dict that logging layers
serialize verbatim into the rejection log.
"""
from __future__ import annotations

from typing import Any


class ForexEngineError(Exception):
    """Base class. Never raised directly."""

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context


class SchemaValidationError(ForexEngineError):
    """Raw payload failed Pydantic validation at Node A."""


class ExtractionError(ForexEngineError):
    """Source-specific parsing failed before schema validation."""


class EvolutionError(ForexEngineError):
    """Node B could not produce a coherent next state."""


class PersistenceError(ForexEngineError):
    """Node D failed to write to the backing store."""


class ValidationError(ForexEngineError):
    """Node C detected an unrecoverable conflict."""


class DisproofError(ForexEngineError):
    """Node E could not rank hypotheses (primary was not the strongest)."""


class GenAIBackendError(ForexEngineError):
    """Pluggable backend returned an error or malformed response."""


class ConfigurationError(ForexEngineError):
    """Missing / malformed config at startup."""
