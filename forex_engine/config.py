"""Runtime configuration for the Forex Context Engine.

Single ``EngineConfig`` dataclass — pulled from environment at startup,
passed explicitly through the call graph so every unit test can
instantiate a deterministic instance without monkey-patching ``os.environ``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class EngineConfig:
    # --- Persistence ---------------------------------------------------
    postgres_dsn: str
    # --- Gen AI --------------------------------------------------------
    genai_provider: str = "anthropic"           # "anthropic" | "openai"
    genai_model: str = "claude-3-5-sonnet-20241022"
    genai_api_key: str = ""
    genai_thinking_budget_tokens: int = 8000    # clamp: 5_000..10_000
    # --- Evolution -----------------------------------------------------
    # Temporal decay: weight = 0.5 ** (age_hours / half_life_hours).
    signal_half_life_hours: Decimal = Decimal("24")
    # --- Validation ----------------------------------------------------
    # Two rate observations for the same entity disagreeing by > this
    # tolerance (in basis points) are flagged as conflicts.
    rate_conflict_tolerance_bps: int = 2
    # --- Logging -------------------------------------------------------
    rejection_log_path: Path = field(default_factory=lambda: Path("./logs/rejections.jsonl"))
    audit_log_path: Path = field(default_factory=lambda: Path("./logs/audit.jsonl"))

    def __post_init__(self) -> None:
        if not self.postgres_dsn:
            raise ConfigurationError("postgres_dsn is required")
        if self.genai_provider not in {"anthropic", "openai"}:
            raise ConfigurationError(
                "unsupported genai_provider",
                provider=self.genai_provider,
            )
        if not (5_000 <= self.genai_thinking_budget_tokens <= 10_000):
            raise ConfigurationError(
                "genai_thinking_budget_tokens must be in [5000, 10000]",
                value=self.genai_thinking_budget_tokens,
            )
        if self.signal_half_life_hours <= 0:
            raise ConfigurationError("signal_half_life_hours must be positive")
        if self.rate_conflict_tolerance_bps < 0:
            raise ConfigurationError("rate_conflict_tolerance_bps must be >= 0")


def load_from_env() -> EngineConfig:
    """Build an ``EngineConfig`` from environment variables.

    Kept separate from the dataclass so tests can construct configs
    directly without touching the environment.
    """
    try:
        return EngineConfig(
            postgres_dsn=os.environ["FOREX_PG_DSN"],
            genai_provider=os.environ.get("FOREX_GENAI_PROVIDER", "anthropic"),
            genai_model=os.environ.get(
                "FOREX_GENAI_MODEL", "claude-3-5-sonnet-20241022"
            ),
            genai_api_key=os.environ.get("FOREX_GENAI_API_KEY", ""),
            genai_thinking_budget_tokens=int(
                os.environ.get("FOREX_THINKING_BUDGET", "8000")
            ),
            signal_half_life_hours=Decimal(
                os.environ.get("FOREX_SIGNAL_HALF_LIFE_HOURS", "24")
            ),
            rate_conflict_tolerance_bps=int(
                os.environ.get("FOREX_RATE_TOLERANCE_BPS", "2")
            ),
            rejection_log_path=Path(
                os.environ.get("FOREX_REJECTION_LOG", "./logs/rejections.jsonl")
            ),
            audit_log_path=Path(
                os.environ.get("FOREX_AUDIT_LOG", "./logs/audit.jsonl")
            ),
        )
    except KeyError as exc:
        raise ConfigurationError(f"missing env var: {exc.args[0]}") from exc
