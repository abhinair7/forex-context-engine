"""Worked example: process a Fed rate hike + market reaction.

Runs end-to-end against an in-memory repo and the ``EchoClient`` so no
vendor keys or PostgreSQL are required. Swap two lines (see bottom) to
hit Anthropic / OpenAI / real Postgres.

  $ python examples/fed_rate_hike.py

Expected: a ``PipelineResult`` with ``stopped_at == "ok"``, printed as
JSON so the audit trail is greppable.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from forex_engine import (
    EchoClient,
    EngineConfig,
    InMemoryStateRepository,
    RawPayload,
    SignalSource,
    build_default_orchestrator,
    configure_logging,
)


def _seed_prior_state(orch, repo) -> None:
    """Seed the repo with a prior state so the rate move is a delta."""
    prior = [
        RawPayload(
            source=SignalSource.FRED,
            payload={
                "series_id": "DFF",
                "unit": "pct",
                "observations": [
                    {"date": "2026-04-16T15:00:00-04:00", "value": "5.33"}
                ],
            },
        ),
        RawPayload(
            source=SignalSource.TIINGO,
            payload={
                "ticker": "eurusd",
                "quote": {
                    "timestamp": "2026-04-16T16:00:00-04:00",
                    "bid": "1.0832", "ask": "1.0834", "mid": "1.0833",
                    "volume": 2_345_000,
                },
            },
        ),
    ]
    orch.run(prior, question="baseline")


def main() -> None:
    cfg = EngineConfig(
        postgres_dsn="postgres://unused",     # in-memory repo, DSN ignored
        genai_provider="anthropic",
        genai_api_key="",                     # empty → EchoClient fallback
        signal_half_life_hours=Decimal("48"),
        rejection_log_path=Path("./logs/rejections.jsonl"),
        audit_log_path=Path("./logs/audit.jsonl"),
    )
    configure_logging(cfg)
    repo = InMemoryStateRepository()
    orch = build_default_orchestrator(cfg, repo, EchoClient())

    _seed_prior_state(orch, repo)

    # Fed hikes 25bps. Simultaneous FX move + vol tick-up + NFP surprise.
    hike = [
        RawPayload(
            source=SignalSource.FRED,
            payload={
                "series_id": "DFF",
                "unit": "pct",
                "observations": [
                    {"date": "2026-04-17T14:00:00-04:00", "value": "5.58"}
                ],
            },
        ),
        RawPayload(
            source=SignalSource.FRED,
            payload={
                "series_id": "DGS10",
                "unit": "pct",
                "observations": [
                    {"date": "2026-04-17T14:05:00-04:00", "value": "4.42"}
                ],
            },
        ),
        RawPayload(
            source=SignalSource.CME,
            payload={
                "contract": "ZQZ6",
                "observed_at": "2026-04-17T14:10:00-04:00",
                "settlement_price": "94.60",
                "expiry": "2026-12-31",
            },
        ),
        RawPayload(
            source=SignalSource.TIINGO,
            payload={
                "ticker": "eurusd",
                "quote": {
                    "timestamp": "2026-04-17T14:15:00-04:00",
                    "bid": "1.0755", "ask": "1.0757", "mid": "1.0756",
                    "volume": 4_120_000,
                },
            },
        ),
        RawPayload(
            source=SignalSource.CBOE,
            payload={
                "index": "VIX",
                "level": "18.40",
                "observed_at": "2026-04-17T14:15:00-04:00",
            },
        ),
        RawPayload(
            source=SignalSource.CFTC,
            payload={
                "market": "EURO FX",
                "report_date": "2026-04-15",
                "noncomm_long": 180_000,
                "noncomm_short": 120_000,
                "open_interest": 800_000,
            },
        ),
        RawPayload(
            source=SignalSource.CALENDAR,
            payload={
                "series": "NFP",
                "actual": "220000",
                "consensus": "175000",
                "release_time": "2026-04-17T08:30:00-04:00",
            },
        ),
    ]

    result = orch.run(
        hike,
        question=(
            "Given a 25bps Fed hike and the observed cross-asset response, "
            "what is the highest-conviction EUR/USD view for the next session?"
        ),
    )
    report(result)


def report(result) -> None:
    summary = {
        "stopped_at": result.stopped_at,
        "state": {
            "state_id": str(result.state.state_id),
            "version": result.state.version,
            "checksum": result.state.checksum(),
            "n_signals": len(result.state.signals),
            "n_events": len(result.state.events),
            "n_relationships": len(result.state.relationships),
        },
        "delta": result.delta.model_dump(mode="json"),
        "validation": {
            "passed": result.validation.passed,
            "n_conflicts": len(result.validation.conflicts),
            "n_warnings": len(result.validation.warnings),
            "conflicts": [c.model_dump(mode="json")
                          for c in result.validation.conflicts],
        },
        "disproof": (
            None if result.disproof is None else {
                "primary_survived": result.disproof.primary_survived,
                "primary_claim": result.disproof.primary.claim,
                "primary_rank": result.disproof.primary.rank,
                "rationale": result.disproof.rationale,
            }
        ),
        "inference": (
            None if result.inference is None else {
                "provider": result.inference.provider,
                "model": result.inference.model,
                "cost_usd": str(result.inference.cost_usd),
                "input_tokens": result.inference.input_tokens,
                "output_tokens": result.inference.output_tokens,
                "answer": result.inference.answer,
            }
        ),
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
