"""Node E — Disproof.

The disproof layer is a Popper-style adversarial filter: we do not ship
a conclusion until we have explicitly considered the three strongest
*alternative* explanations and confirmed that our primary hypothesis
still dominates on a scored rubric.

Scoring rubric (each in [0, 1], weighted and summed):
  - evidence_breadth:   how many signals does the hypothesis cite?
  - evidence_recency:   how close to ``now`` is the cited evidence?
  - orthogonality:      does it cite signals from multiple sources?
  - counter_evidence:   inverse — does any signal contradict it?
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable, Iterable

from .exceptions import DisproofError
from .logging_setup import audit
from .models import (
    ContextState,
    DisproofResult,
    Event,
    Hypothesis,
    Signal,
    SignalKind,
)
from .time_utils import now_est


# Weights for the scoring rubric. Re-tuning these is a supervised exercise;
# keeping them here (not in config) makes it clear they're product-level
# parameters and must be re-validated if they change.
_W_BREADTH = Decimal("0.25")
_W_RECENCY = Decimal("0.30")
_W_ORTHO = Decimal("0.25")
_W_COUNTER = Decimal("0.20")


@dataclass(frozen=True)
class _Scored:
    hypothesis: Hypothesis
    raw: Decimal


class DisproofEngine:
    """Node E entry point."""

    def __init__(self, clock: Callable[[], datetime] = now_est) -> None:
        self._clock = clock

    def challenge(self, state: ContextState) -> DisproofResult:
        primary_claim = self._derive_primary_claim(state)
        primary_signals = self._signals_supporting_primary(state)
        if not primary_signals:
            raise DisproofError(
                "no signals available to form a primary hypothesis",
                state_id=str(state.state_id),
            )

        candidates: list[Hypothesis] = [
            self._build(
                is_primary=True,
                claim=primary_claim,
                supporting=primary_signals,
                counter=self._counter_evidence(state, primary_signals),
            ),
            *self._alternatives(state),
        ]
        scored = self._score_all(state, candidates)
        ranked = self._rank(scored)
        primary_survived = ranked[0].hypothesis.is_primary

        # Re-stamp ranks from scoring order, then re-split by flag.
        ranked_with_ranks = [
            self._with_rank(s.hypothesis, i + 1)
            for i, s in enumerate(ranked)
        ]
        primary_out = next(h for h in ranked_with_ranks if h.is_primary)
        alts_out = tuple(h for h in ranked_with_ranks if not h.is_primary)

        rationale = self._rationale(scored, ranked)
        result = DisproofResult(
            state_id=state.state_id,
            primary=primary_out,
            alternatives=alts_out,
            primary_survived=primary_survived,
            rationale=rationale,
            evaluated_at=self._clock(),
        )
        audit(
            "node_e.challenged",
            state_id=str(state.state_id),
            primary_survived=primary_survived,
            top_score=str(ranked[0].raw),
        )
        return result

    # ---- primary hypothesis ----------------------------------------
    @staticmethod
    def _derive_primary_claim(state: ContextState) -> str:
        """Turn the latest policy-rate event (if any) into a hypothesis."""
        rate_events: list[Event] = [
            e for e in state.events if "_RATE_" in e.event_type
        ]
        if rate_events:
            ev = max(rate_events, key=lambda e: e.occurred_at)
            direction = "strengthen" if "HIKE" in ev.event_type else "weaken"
            return (
                f"{ev.entity} carry-trade flows will {direction} the "
                f"currency over the next session "
                f"({ev.event_type}, {ev.magnitude_bps}bps)."
            )
        return (
            "Current positioning and yield differentials dominate near-term "
            "FX direction; no fresh policy catalyst."
        )

    @staticmethod
    def _signals_supporting_primary(
        state: ContextState,
    ) -> tuple[Signal, ...]:
        kinds = {
            SignalKind.POLICY_RATE,
            SignalKind.TREASURY_YIELD,
            SignalKind.IMPLIED_RATE_PATH,
        }
        return tuple(s for s in state.signals if s.kind in kinds)

    # ---- alternatives ----------------------------------------------
    def _alternatives(self, state: ContextState) -> Iterable[Hypothesis]:
        """The spec requires exactly three alternative hypotheses.

        Each one leans on a different *signal channel* so they are
        genuinely orthogonal — not three framings of the same view.
        """
        yield self._build(
            is_primary=False,
            claim=(
                "Positioning is already one-sided; a short-squeeze unwind "
                "dominates the rate-differential story."
            ),
            supporting=tuple(
                s for s in state.signals
                if s.kind == SignalKind.COT_POSITIONING
            ),
            counter=self._counter_evidence(state, ()),
        )
        yield self._build(
            is_primary=False,
            claim=(
                "Realized and implied vol is elevated; risk-off flows to "
                "funding currencies (JPY, CHF, USD) outweigh carry."
            ),
            supporting=tuple(
                s for s in state.signals
                if s.kind == SignalKind.VOLATILITY
            ),
            counter=self._counter_evidence(state, ()),
        )
        yield self._build(
            is_primary=False,
            claim=(
                "Upcoming economic release (NFP/CPI) dominates regime; "
                "pre-release markets are noise, post-release will reprice."
            ),
            supporting=tuple(
                s for s in state.signals
                if s.kind == SignalKind.ECON_RELEASE
            ),
            counter=self._counter_evidence(state, ()),
        )

    # ---- scoring ---------------------------------------------------
    def _score_all(
        self, state: ContextState, hs: list[Hypothesis]
    ) -> list[_Scored]:
        now = self._clock()
        max_age_h = self._max_age_hours(state, now)
        all_sources_count = max(
            1, len({s.source for s in state.signals})
        )
        return [
            _Scored(
                hypothesis=h,
                raw=self._score(
                    h, state, now, max_age_h, all_sources_count
                ),
            )
            for h in hs
        ]

    @staticmethod
    def _max_age_hours(state: ContextState, now: datetime) -> Decimal:
        if not state.signals:
            return Decimal("1")
        max_seconds = max(
            (now - s.observed_at).total_seconds() for s in state.signals
        )
        return Decimal(str(max(1.0, max_seconds / 3600)))

    @staticmethod
    def _score(
        h: Hypothesis,
        state: ContextState,
        now: datetime,
        max_age_h: Decimal,
        all_sources_count: int,
    ) -> Decimal:
        sig_by_id = {s.signal_id: s for s in state.signals}
        supporting = [
            sig_by_id[sid] for sid in h.supporting_signal_ids
            if sid in sig_by_id
        ]
        # breadth
        breadth = Decimal(
            str(min(1.0, len(supporting) / 5.0))
        )
        # recency
        if supporting:
            avg_age_s = sum(
                (now - s.observed_at).total_seconds() for s in supporting
            ) / len(supporting)
            avg_age_h = Decimal(str(max(0.0, avg_age_s / 3600)))
            recency = max(Decimal("0"), Decimal("1") - (avg_age_h / max_age_h))
        else:
            recency = Decimal("0")
        # orthogonality
        ortho = (
            Decimal(len({s.source for s in supporting}))
            / Decimal(all_sources_count)
            if supporting else Decimal("0")
        )
        # counter evidence
        counter_pen = Decimal(
            str(min(1.0, len(h.counter_evidence) / 3.0))
        )
        counter_term = Decimal("1") - counter_pen

        raw = (
            _W_BREADTH * breadth
            + _W_RECENCY * recency
            + _W_ORTHO * ortho
            + _W_COUNTER * counter_term
        )
        return raw.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    # ---- ranking / helpers -----------------------------------------
    @staticmethod
    def _rank(scored: list[_Scored]) -> list[_Scored]:
        return sorted(scored, key=lambda s: s.raw, reverse=True)

    @staticmethod
    def _with_rank(h: Hypothesis, rank: int) -> Hypothesis:
        return Hypothesis(
            hypothesis_id=h.hypothesis_id,
            is_primary=h.is_primary,
            claim=h.claim,
            supporting_signal_ids=h.supporting_signal_ids,
            confidence_score=h.confidence_score,
            counter_evidence=h.counter_evidence,
            rank=rank,
        )

    def _build(
        self,
        *,
        is_primary: bool,
        claim: str,
        supporting: tuple[Signal, ...],
        counter: tuple[str, ...],
    ) -> Hypothesis:
        # Confidence is a coarse prior; the rubric score is the truth.
        base = Decimal("0.6") if is_primary else Decimal("0.4")
        if not supporting:
            base = Decimal("0.1")
        return Hypothesis(
            is_primary=is_primary,
            claim=claim,
            supporting_signal_ids=tuple(s.signal_id for s in supporting),
            confidence_score=base,
            counter_evidence=counter,
            rank=1,                              # placeholder, overwritten
        )

    @staticmethod
    def _counter_evidence(
        state: ContextState, supporting: tuple[Signal, ...]
    ) -> tuple[str, ...]:
        """Treat vol spikes as counter-evidence to any directional thesis."""
        supporting_ids = {s.signal_id for s in supporting}
        out: list[str] = []
        for s in state.signals:
            if s.signal_id in supporting_ids:
                continue
            if s.kind == SignalKind.VOLATILITY and s.value_decimal is not None:
                if s.value_decimal > Decimal("25"):
                    out.append(
                        f"{s.entity} at {s.value_decimal} indicates stress "
                        f"regime, counter to carry thesis"
                    )
        return tuple(out)

    @staticmethod
    def _rationale(scored: list[_Scored], ranked: list[_Scored]) -> str:
        lines = ["Ranking (rubric: breadth+recency+orthogonality-counter):"]
        for i, s in enumerate(ranked, 1):
            tag = "PRIMARY" if s.hypothesis.is_primary else "alt"
            lines.append(f"  {i}. [{tag}] score={s.raw}  {s.hypothesis.claim}")
        return "\n".join(lines)
