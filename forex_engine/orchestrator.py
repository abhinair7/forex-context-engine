"""Orchestrator — the LangGraph-style flow A → B → D → C → E → Gen AI.

Design notes
------------
- Every node is injected. The orchestrator itself has no knowledge of
  which parser, which repo, or which Gen AI backend is in use.
- The flow returns a ``PipelineResult`` carrying every intermediate
  artifact so callers can inspect audit trails without re-deriving them.
- If Node C fails (critical conflict) or Node E rejects the primary
  hypothesis, ``inference`` is ``None`` and ``stopped_at`` says which
  gate closed. The caller decides whether to page an analyst.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from .config import EngineConfig
from .exceptions import DisproofError
from .genai_backend import GenAIClient
from .logging_setup import audit
from .models import (
    ContextState,
    DisproofResult,
    InferenceRequest,
    InferenceResponse,
    Signal,
    SignalSource,
    StateDelta,
    ValidationResult,
)
from .node_a_extraction import Extractor
from .node_b_evolution import Evolver
from .node_c_validation import Validator
from .node_d_persistence import StateRepository
from .node_e_disproof import DisproofEngine


@dataclass(frozen=True)
class RawPayload:
    """A single vendor payload awaiting extraction."""
    source: SignalSource
    payload: dict


@dataclass(frozen=True)
class PipelineResult:
    state: ContextState
    delta: StateDelta
    validation: ValidationResult
    disproof: DisproofResult | None
    inference: InferenceResponse | None
    stopped_at: str  # "ok" | "validation" | "disproof" | "genai_skipped"


class Orchestrator:
    def __init__(
        self,
        cfg: EngineConfig,
        extractor: Extractor,
        evolver: Evolver,
        validator: Validator,
        disproof: DisproofEngine,
        repo: StateRepository,
        genai: GenAIClient,
    ) -> None:
        self._cfg = cfg
        self._extractor = extractor
        self._evolver = evolver
        self._validator = validator
        self._disproof = disproof
        self._repo = repo
        self._genai = genai

    # ---- public -----------------------------------------------------
    def run(
        self,
        raw_payloads: Iterable[RawPayload],
        question: str,
    ) -> PipelineResult:
        # A — Extract
        signals = self._extract_all(raw_payloads)
        audit("orchestrator.extracted", n=len(signals))

        # B — Evolve
        parent = self._repo.latest()
        evolution = self._evolver.evolve(parent, signals)
        state, delta = evolution.state, evolution.delta

        # C — Validate (pre-persist so we don't save garbage if critical)
        validation = self._validator.validate(state)

        # D — Persist only if not critically broken.
        # (The audit trail still gets written either way via logs; the
        # DB invariant is "no critical-conflict states survive".)
        if validation.passed:
            self._repo.save(state, delta, validation)
        else:
            audit(
                "orchestrator.validation_failed.skip_persist",
                state_id=str(state.state_id),
                n_conflicts=len(validation.conflicts),
            )
            return PipelineResult(
                state=state, delta=delta, validation=validation,
                disproof=None, inference=None,
                stopped_at="validation",
            )

        # E — Disproof
        try:
            disproof = self._disproof.challenge(state)
        except DisproofError as exc:
            audit(
                "orchestrator.disproof_error",
                state_id=str(state.state_id), reason=exc.message,
            )
            return PipelineResult(
                state=state, delta=delta, validation=validation,
                disproof=None, inference=None,
                stopped_at="disproof",
            )

        if not disproof.primary_survived:
            audit(
                "orchestrator.primary_not_survived.skip_genai",
                state_id=str(state.state_id),
            )
            return PipelineResult(
                state=state, delta=delta, validation=validation,
                disproof=disproof, inference=None,
                stopped_at="disproof",
            )

        # Gen AI
        request = self._build_request(state, disproof, question)
        response = self._genai.infer(request)
        audit(
            "orchestrator.genai_complete",
            state_id=str(state.state_id),
            provider=response.provider,
            model=response.model,
            cost_usd=str(response.cost_usd),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        return PipelineResult(
            state=state, delta=delta, validation=validation,
            disproof=disproof, inference=response,
            stopped_at="ok",
        )

    # ---- helpers ----------------------------------------------------
    def _extract_all(
        self, raw: Iterable[RawPayload]
    ) -> tuple[Signal, ...]:
        out: list[Signal] = []
        for item in raw:
            out.extend(self._extractor.extract(item.source, item.payload))
        return tuple(out)

    def _build_request(
        self,
        state: ContextState,
        disproof: DisproofResult,
        question: str,
    ) -> InferenceRequest:
        """Curate the payload — only what passed validation + the surviving
        primary hypothesis + its cited evidence."""
        signal_ids: set[UUID] = set(disproof.primary.supporting_signal_ids)
        sigs = [
            s.model_dump(mode="json")
            for s in state.signals if s.signal_id in signal_ids
        ]
        payload = {
            "state_id": str(state.state_id),
            "version": state.version,
            "primary_hypothesis": disproof.primary.claim,
            "primary_rank": disproof.primary.rank,
            "signals": sigs,
            "events": [e.model_dump(mode="json") for e in state.events],
            "relationships": [
                r.model_dump(mode="json") for r in state.relationships
            ],
        }
        return InferenceRequest(
            state_id=state.state_id,
            context_payload=payload,
            question=question,
            thinking_budget_tokens=self._cfg.genai_thinking_budget_tokens,
        )


def build_default_orchestrator(
    cfg: EngineConfig,
    repo: StateRepository,
    genai: GenAIClient,
) -> Orchestrator:
    """Convenience constructor with stock Node A–E instances."""
    return Orchestrator(
        cfg=cfg,
        extractor=Extractor(),
        evolver=Evolver(cfg),
        validator=Validator(cfg),
        disproof=DisproofEngine(),
        repo=repo,
        genai=genai,
    )
