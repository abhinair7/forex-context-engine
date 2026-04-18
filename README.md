# Forex Context Engine

A production-grade 5-node pipeline that turns raw institutional forex data
(FRED, Tiingo, CME, CFTC, ECB, BoJ, CBOE, Calendar, …) into validated,
versioned context and hands it to a pluggable Gen AI backend (Claude /
OpenAI / your own).

```
Raw payloads ─► Node A ─► Node B ─► Node D ─► Node C ─► Node E ─► Gen AI
              Extract  Evolve  Persist   Validate  Disprove
```

- **Pydantic v2** strict schemas, `extra="forbid"`, immutable states
- **EST** timestamps end-to-end (fixed UTC-5, no DST)
- **PostgreSQL** persistence with ACID versioning + append-only audit trail
- **Pluggable Gen AI** via `GenAIClient` Protocol (Claude Extended Thinking, OpenAI, or Echo for tests)
- **Adversarial disproof**: every primary hypothesis is ranked against three orthogonal alternatives before any Gen AI call
- **Cost tracking**: every inference logs tokens and USD cost
- Every file under 500 lines, every function has one responsibility

---

## Install

```bash
git clone https://github.com/abhinair7/forex-context-engine.git
cd forex-context-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`pydantic` is the only hard dependency. `psycopg`, `anthropic`, and
`openai` are only needed if you actually use those backends.

Run the worked example (no API keys, no Postgres — in-memory repo and
Echo backend):

```bash
PYTHONPATH=. python examples/fed_rate_hike.py
```

You should see a JSON report with `stopped_at: "ok"`, a disproof
ranking, and an inference block.

---

## Using it in your own code

### 1. Minimal — in-memory, Echo backend (no keys, no DB)

```python
from decimal import Decimal
from forex_engine import (
    EchoClient, EngineConfig, InMemoryStateRepository,
    RawPayload, SignalSource, build_default_orchestrator,
    configure_logging,
)

cfg = EngineConfig(
    postgres_dsn="postgres://unused",   # ignored by InMemoryStateRepository
    genai_api_key="",                   # empty ⇒ EchoClient fallback
    signal_half_life_hours=Decimal("24"),
)
configure_logging(cfg)

repo  = InMemoryStateRepository()
orch  = build_default_orchestrator(cfg, repo, EchoClient())

payloads = [
    RawPayload(
        source=SignalSource.FRED,
        payload={
            "series_id": "DFF", "unit": "pct",
            "observations": [
                {"date": "2026-04-17T14:00:00-04:00", "value": "5.58"}
            ],
        },
    ),
    # ...add Tiingo, CME, CFTC, CBOE, Calendar payloads as needed
]

result = orch.run(
    payloads,
    question="Highest-conviction EUR/USD view for next session?",
)

print(result.stopped_at)                      # "ok" | "validation" | "disproof"
print(result.state.version, result.state.checksum())
print(result.inference.answer)                # None if a gate closed
print(result.inference.cost_usd)              # Decimal, always populated
```

### 2. Swap to Claude with Extended Thinking

```python
from forex_engine import build_client, EngineConfig

cfg = EngineConfig(
    postgres_dsn="postgres://user:pw@host/db",
    genai_provider="anthropic",
    genai_model="claude-3-5-sonnet-20241022",
    genai_api_key="sk-ant-...",
    genai_thinking_budget_tokens=8000,    # clamped to [5000, 10000]
)
genai = build_client(cfg)                 # ← one line swap
```

### 3. Swap to GPT-4o

```python
cfg = EngineConfig(
    postgres_dsn="postgres://...",
    genai_provider="openai",
    genai_model="gpt-4o",
    genai_api_key="sk-...",
)
genai = build_client(cfg)
```

`build_client` is the only place provider names live. Nodes A–E never
import `genai_backend`.

### 4. Use real PostgreSQL

```bash
psql "$FOREX_PG_DSN" -f sql/001_init.sql
```

```python
from forex_engine import PostgresStateRepository, build_default_orchestrator

repo = PostgresStateRepository(cfg.postgres_dsn)
orch = build_default_orchestrator(cfg, repo, genai)
```

State immutability is enforced in three places:
- `ContextState` is `frozen=True` (Pydantic)
- `InMemoryStateRepository.save` rejects duplicate `state_id`
- PostgreSQL: `UNIQUE(version)`, `CHECK` on genesis/parent, recommend
  `REVOKE UPDATE, DELETE` on the app role

### 5. Environment-driven config

If you'd rather drive it from env vars:

```python
from forex_engine import load_from_env, configure_logging

cfg = load_from_env()
configure_logging(cfg)
```

Required: `FOREX_PG_DSN`.
Optional: `FOREX_GENAI_PROVIDER`, `FOREX_GENAI_MODEL`,
`FOREX_GENAI_API_KEY`, `FOREX_THINKING_BUDGET`,
`FOREX_SIGNAL_HALF_LIFE_HOURS`, `FOREX_RATE_TOLERANCE_BPS`,
`FOREX_REJECTION_LOG`, `FOREX_AUDIT_LOG`.

---

## Data source payload shapes

Node A parses the exact shape each vendor ships. Mismatched payloads go
to the rejection log with a structured reason — never into a state.

| Source    | `SignalSource`    | Example payload fields |
|-----------|-------------------|-----------------------|
| FRED      | `FRED`            | `series_id`, `unit`, `observations[{date, value}]` |
| Tiingo    | `TIINGO`          | `ticker`, `quote{timestamp, bid, ask, mid, volume}` |
| CME       | `CME`             | `contract`, `observed_at`, `settlement_price`, `expiry` |
| CFTC      | `CFTC`            | `market`, `report_date`, `noncomm_long/short`, `open_interest` |
| ECB / BoJ | `ECB` / `BOJ`     | `observed_at`, `policy_rate_pct`, `forward_guidance{tone_score}` |
| CBOE      | `CBOE`            | `index` ∈ {VIX, MOVE, EVZ}, `level`, `observed_at` |
| Calendar  | `CALENDAR`        | `series` ∈ {NFP, CPI_YOY, PMI, UNEMPLOYMENT, RETAIL_SALES}, `actual`, `consensus`, `release_time` |

See [`examples/fed_rate_hike.py`](examples/fed_rate_hike.py) for
complete working payloads.

---

## What the pipeline returns

`orch.run(...)` returns a `PipelineResult`:

```python
PipelineResult(
    state:       ContextState,        # version, signals, events, relationships, checksum()
    delta:       StateDelta,          # added/removed/decayed ids
    validation:  ValidationResult,    # passed, conflicts, warnings
    disproof:    DisproofResult|None, # primary_hypothesis + 3 alternatives + rationale
    inference:   InferenceResponse|None,  # answer, tokens, cost_usd
    stopped_at:  str,                 # "ok" | "validation" | "disproof"
)
```

If a gate closed (`stopped_at != "ok"`), `inference` is `None` and
`disproof` may also be `None`. No Gen AI call is made — no tokens burned.

---

## Architecture

```
forex_engine/
├── models.py              Pydantic v2 schemas (Signal, Event, ContextState, …)
├── time_utils.py          EST policy (single source of truth)
├── config.py              EngineConfig + load_from_env
├── exceptions.py          Typed exception hierarchy
├── logging_setup.py       JSONL audit + rejection sinks
├── node_a_extraction.py   Vendor parsers → Signal[]
├── node_b_evolution.py    Delta detection, decay, relationships, events
├── node_c_validation.py   Rate-conflict / temporal / spread-sanity checks
├── node_d_persistence.py  StateRepository Protocol (in-memory + Postgres)
├── node_e_disproof.py     Primary + 3 alternatives, scored rubric
├── genai_backend.py       GenAIClient Protocol (Anthropic, OpenAI, Echo)
└── orchestrator.py        A → B → D → C → E → Gen AI
sql/001_init.sql           Postgres schema (context_state + audit table)
examples/fed_rate_hike.py  Runnable end-to-end demo
```

---

## Extending

### Add a new data source
1. Add a member to `SignalSource` in `models.py`.
2. Add a parser class in `node_a_extraction.py` exposing `parse(payload) -> list[Signal]`.
3. Register it in `Extractor.__init__` and the dispatch dict in `Extractor.extract`.

### Add a new validation rule
1. Add a `_check_*` method on `Validator` in `node_c_validation.py`.
2. Call it from `validate()`; return either `Conflict`s (with `Severity`) or warning strings.

### Add a new Gen AI provider
1. Add a class with a `provider`, `model`, and `infer(request) -> InferenceResponse` method.
2. Add a price entry in `_PRICE_TABLE` in `genai_backend.py`.
3. Add a branch in `build_client`.

No other file needs to change.

---

## Logging

Two JSONL streams, both created under `cfg.audit_log_path.parent`:

- `logs/audit.jsonl` — one line per decision point (node entered, N signals parsed, conflict detected, hypothesis ranked, inference cost).
- `logs/rejections.jsonl` — one line per rejected payload with the vendor, reason, and the raw input.

Both are single-level JSON. Tail with `jq` or ship to Splunk/Datadog without a parser.

---

## Quality bar

- All state-bearing models are `frozen=True`.
- Every financial value is `int` (bps) or `Decimal` — never `float`.
- All timestamps are EST, rejected otherwise.
- Every node is unit-testable with dependency injection; no module does network I/O at import time.
- Every rejection is logged with a structured reason; a quant desk can reconstruct *why* any payload was dropped.

---

## License / use

Private repo. Add your own `LICENSE` before distributing.
