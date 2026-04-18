"""Minimal integration template.

Copy this into your own project, replace the payloads with real data
pulled from your vendor clients, and swap the backend when you're
ready.

  $ PYTHONPATH=. python examples/minimal_integration.py
"""
from __future__ import annotations

from decimal import Decimal

from forex_engine import (
    EchoClient,
    EngineConfig,
    InMemoryStateRepository,
    RawPayload,
    SignalSource,
    build_default_orchestrator,
    configure_logging,
)


def get_vendor_payloads() -> list[RawPayload]:
    """Replace the dicts below with whatever your FRED/Tiingo/... client returns."""
    return [
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
    ]


def main() -> None:
    # 1. Build the config. For real use, prefer load_from_env().
    cfg = EngineConfig(
        postgres_dsn="postgres://unused",
        genai_api_key="",                 # empty key ⇒ EchoClient fallback
        signal_half_life_hours=Decimal("24"),
    )
    configure_logging(cfg)

    # 2. Pick a repository. Use PostgresStateRepository in production.
    repo = InMemoryStateRepository()

    # 3. Pick a Gen AI backend. Swap with build_client(cfg) when you have a key.
    genai = EchoClient()

    # 4. Build the orchestrator — A, B, C, D, E wired with defaults.
    orch = build_default_orchestrator(cfg, repo, genai)

    # 5. Feed it payloads and ask a question.
    result = orch.run(
        get_vendor_payloads(),
        question="What is the near-term directional bias for EUR/USD?",
    )

    # 6. Consume the result.
    print(f"stopped_at     = {result.stopped_at}")
    print(f"state version  = {result.state.version}")
    print(f"state checksum = {result.state.checksum()}")
    print(f"validation     = passed={result.validation.passed}, "
          f"conflicts={len(result.validation.conflicts)}")
    if result.disproof:
        print(f"primary claim  = {result.disproof.primary.claim}")
        print(f"primary rank   = {result.disproof.primary.rank}")
        print(f"survived       = {result.disproof.primary_survived}")
    if result.inference:
        print(f"answer         = {result.inference.answer}")
        print(f"cost USD       = {result.inference.cost_usd}")


if __name__ == "__main__":
    main()
