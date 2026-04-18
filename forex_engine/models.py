"""Pydantic v2 schemas — the contract between every node.

Design rules
------------
- All timestamps are timezone-aware EST (see ``time_utils.EST``).
  A naive datetime — or any other offset — is a bug.
- Financial magnitudes use ``int`` basis-points (rates/yields) or
  ``Decimal`` (prices, volatilities, ratios). Never ``float``.
- Every state-bearing model is ``frozen=True``. A new state is a new
  object; mutation is a category error.
- ``extra="forbid"`` — unknown keys are a loud failure.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .time_utils import now_est, require_est


# --------------------------------------------------------------------------- #
# Enums                                                                       #
# --------------------------------------------------------------------------- #
class SignalSource(str, Enum):
    FRED = "fred"
    TIINGO = "tiingo"
    CME = "cme"
    CFTC = "cftc"
    ECB = "ecb"
    BOJ = "boj"
    CBOE = "cboe"
    BIS = "bis"
    CALENDAR = "calendar"
    CROSSASSET = "crossasset"


class SignalKind(str, Enum):
    POLICY_RATE = "policy_rate"
    TREASURY_YIELD = "treasury_yield"
    REAL_YIELD = "real_yield"
    FX_SPOT = "fx_spot"
    FX_SPREAD = "fx_spread"
    FX_VOLUME = "fx_volume"
    IMPLIED_RATE_PATH = "implied_rate_path"
    COT_POSITIONING = "cot_positioning"
    FORWARD_GUIDANCE = "forward_guidance"
    VOLATILITY = "volatility"
    ECON_RELEASE = "econ_release"
    CROSS_CORRELATION = "cross_correlation"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_BASE = ConfigDict(frozen=True, extra="forbid", strict=True)


# --------------------------------------------------------------------------- #
# Signal                                                                      #
# --------------------------------------------------------------------------- #
class Signal(BaseModel):
    """One observation from one source at one instant."""

    model_config = _BASE

    signal_id: UUID = Field(default_factory=uuid4)
    source: SignalSource
    kind: SignalKind
    entity: str = Field(min_length=1, max_length=64)
    value_bps: Optional[int] = None
    value_decimal: Optional[Decimal] = None
    unit: str = Field(min_length=1, max_length=32)
    observed_at: datetime
    ingested_at: datetime = Field(
        default_factory=now_est
    )
    confidence: Confidence = Confidence.HIGH
    raw_payload: dict = Field(default_factory=dict)

    @field_validator("observed_at", "ingested_at")
    @classmethod
    def _est(cls, v: datetime) -> datetime:
        return require_est(v)

    @model_validator(mode="after")
    def _exactly_one_value(self) -> "Signal":
        has_bps = self.value_bps is not None
        has_dec = self.value_decimal is not None
        if has_bps == has_dec:
            raise ValueError(
                "exactly one of value_bps or value_decimal must be set"
            )
        return self

    @model_validator(mode="after")
    def _ingested_after_observed(self) -> "Signal":
        if self.ingested_at < self.observed_at:
            raise ValueError("ingested_at must be >= observed_at")
        return self


# --------------------------------------------------------------------------- #
# Event                                                                       #
# --------------------------------------------------------------------------- #
class Event(BaseModel):
    """A domain event derived from one or more signals (e.g. rate hike)."""

    model_config = _BASE

    event_id: UUID = Field(default_factory=uuid4)
    event_type: str = Field(min_length=1, max_length=64)
    entity: str = Field(min_length=1, max_length=64)
    magnitude_bps: Optional[int] = None
    magnitude_decimal: Optional[Decimal] = None
    occurred_at: datetime
    source_signal_ids: tuple[UUID, ...]
    description: str = Field(max_length=512)

    @field_validator("occurred_at")
    @classmethod
    def _est(cls, v: datetime) -> datetime:
        return require_est(v)

    @field_validator("source_signal_ids")
    @classmethod
    def _non_empty(cls, v: tuple[UUID, ...]) -> tuple[UUID, ...]:
        if not v:
            raise ValueError("event must cite at least one source signal")
        return v


# --------------------------------------------------------------------------- #
# Relationship                                                                #
# --------------------------------------------------------------------------- #
class Relationship(BaseModel):
    """Directed edge: source_entity --(type, strength)--> target_entity."""

    model_config = _BASE

    relationship_id: UUID = Field(default_factory=uuid4)
    source_entity: str = Field(min_length=1, max_length=64)
    target_entity: str = Field(min_length=1, max_length=64)
    relationship_type: str = Field(min_length=1, max_length=64)
    strength: Decimal                          # -1 .. 1
    observed_at: datetime
    evidence_signal_ids: tuple[UUID, ...] = Field(default_factory=tuple)

    @field_validator("observed_at")
    @classmethod
    def _est(cls, v: datetime) -> datetime:
        return require_est(v)

    @field_validator("strength")
    @classmethod
    def _bounded(cls, v: Decimal) -> Decimal:
        if v < Decimal("-1") or v > Decimal("1"):
            raise ValueError("strength must be in [-1, 1]")
        return v

    @model_validator(mode="after")
    def _no_self_loop(self) -> "Relationship":
        if self.source_entity == self.target_entity:
            raise ValueError("self-loops are not meaningful here")
        return self


# --------------------------------------------------------------------------- #
# ContextState                                                                #
# --------------------------------------------------------------------------- #
class ContextState(BaseModel):
    """Immutable snapshot. A new state is a new record (see Node D)."""

    model_config = _BASE

    state_id: UUID = Field(default_factory=uuid4)
    version: int = Field(ge=1)
    created_at: datetime = Field(
        default_factory=now_est
    )
    parent_state_id: Optional[UUID] = None
    signals: tuple[Signal, ...] = Field(default_factory=tuple)
    events: tuple[Event, ...] = Field(default_factory=tuple)
    relationships: tuple[Relationship, ...] = Field(default_factory=tuple)
    metadata: dict = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _est(cls, v: datetime) -> datetime:
        return require_est(v)

    @model_validator(mode="after")
    def _genesis_has_no_parent(self) -> "ContextState":
        if self.version == 1 and self.parent_state_id is not None:
            raise ValueError("genesis state (version=1) must have no parent")
        if self.version > 1 and self.parent_state_id is None:
            raise ValueError("non-genesis state must have a parent_state_id")
        return self

    def checksum(self) -> str:
        """Deterministic sha256 over canonical JSON — used in audit trail."""
        canonical = self.model_dump(mode="json", exclude={"state_id"})
        blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
class Conflict(BaseModel):
    model_config = _BASE
    conflict_id: UUID = Field(default_factory=uuid4)
    conflict_type: str = Field(min_length=1, max_length=64)
    description: str = Field(max_length=512)
    signal_ids: tuple[UUID, ...]
    severity: Severity


class ValidationResult(BaseModel):
    model_config = _BASE
    state_id: UUID
    passed: bool
    conflicts: tuple[Conflict, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    validated_at: datetime = Field(
        default_factory=now_est
    )

    @field_validator("validated_at")
    @classmethod
    def _est(cls, v: datetime) -> datetime:
        return require_est(v)

    @model_validator(mode="after")
    def _passed_implies_no_critical(self) -> "ValidationResult":
        if self.passed and any(
            c.severity == Severity.CRITICAL for c in self.conflicts
        ):
            raise ValueError("passed=True is incompatible with critical conflicts")
        return self


# --------------------------------------------------------------------------- #
# Disproof                                                                    #
# --------------------------------------------------------------------------- #
class Hypothesis(BaseModel):
    model_config = _BASE
    hypothesis_id: UUID = Field(default_factory=uuid4)
    is_primary: bool
    claim: str = Field(min_length=1, max_length=512)
    supporting_signal_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    confidence_score: Decimal                  # 0 .. 1
    counter_evidence: tuple[str, ...] = Field(default_factory=tuple)
    rank: int = Field(ge=1)

    @field_validator("confidence_score")
    @classmethod
    def _unit(cls, v: Decimal) -> Decimal:
        if v < Decimal("0") or v > Decimal("1"):
            raise ValueError("confidence_score must be in [0, 1]")
        return v


class DisproofResult(BaseModel):
    model_config = _BASE
    state_id: UUID
    primary: Hypothesis
    alternatives: tuple[Hypothesis, ...]
    primary_survived: bool
    rationale: str = Field(max_length=2048)
    evaluated_at: datetime = Field(
        default_factory=now_est
    )

    @field_validator("evaluated_at")
    @classmethod
    def _est(cls, v: datetime) -> datetime:
        return require_est(v)

    @field_validator("alternatives")
    @classmethod
    def _exactly_three(
        cls, v: tuple[Hypothesis, ...]
    ) -> tuple[Hypothesis, ...]:
        # The spec requires exactly three alternative hypotheses.
        if len(v) != 3:
            raise ValueError("disproof requires exactly 3 alternatives")
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "DisproofResult":
        if not self.primary.is_primary:
            raise ValueError("primary.is_primary must be True")
        if any(a.is_primary for a in self.alternatives):
            raise ValueError("alternatives must have is_primary=False")
        if self.primary_survived and any(
            a.confidence_score > self.primary.confidence_score
            for a in self.alternatives
        ):
            raise ValueError(
                "primary_survived=True requires no alternative to outrank it"
            )
        return self


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #
class InferenceRequest(BaseModel):
    model_config = _BASE
    request_id: UUID = Field(default_factory=uuid4)
    state_id: UUID
    # Already-curated payload (post-validation) — nodes A..E decide what goes in.
    context_payload: dict
    question: str = Field(min_length=1, max_length=4096)
    thinking_budget_tokens: int = Field(ge=5_000, le=10_000)


class InferenceResponse(BaseModel):
    model_config = _BASE
    request_id: UUID
    response_id: UUID = Field(default_factory=uuid4)
    provider: str = Field(min_length=1, max_length=32)
    model: str = Field(min_length=1, max_length=64)
    answer: str
    thinking: Optional[str] = None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    thinking_tokens: Optional[int] = Field(default=None, ge=0)
    cost_usd: Decimal = Field(ge=Decimal("0"))
    generated_at: datetime = Field(
        default_factory=now_est
    )

    @field_validator("generated_at")
    @classmethod
    def _est(cls, v: datetime) -> datetime:
        return require_est(v)


# --------------------------------------------------------------------------- #
# Delta (Node B output)                                                        #
# --------------------------------------------------------------------------- #
class StateDelta(BaseModel):
    """What changed between parent state and new state."""

    model_config = _BASE

    added_signal_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    removed_signal_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    added_event_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    added_relationship_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    decayed_signal_ids: tuple[UUID, ...] = Field(default_factory=tuple)
    notes: tuple[str, ...] = Field(default_factory=tuple)
