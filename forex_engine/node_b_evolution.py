"""Node B — Evolution.

Responsibilities:

1. **Delta detection** — what changed vs. the parent state.
2. **Temporal decay** — drop signals whose weight has fallen below the
   floor. Half-life is configured in ``EngineConfig``.
3. **Relationship building** — derive directed edges between entities
   when two signals co-move in ways documented in the rules table.
4. **Event synthesis** — turn a series of policy-rate signals into a
   ``FED_RATE_HIKE`` / ``CUT`` event when the bps change is non-zero.

The class is a pure function from (parent_state, new_signals) →
(next_state, delta). No I/O, no clock reads except via the injected
``clock`` callable (tests pass a fixed clock).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable, Iterable

from .config import EngineConfig
from .logging_setup import audit
from .models import (
    ContextState,
    Event,
    Relationship,
    Signal,
    SignalKind,
    StateDelta,
)
from .time_utils import now_est


# Weight below which a signal is considered decayed and is dropped.
_DECAY_FLOOR = Decimal("0.05")


@dataclass(frozen=True)
class EvolutionResult:
    state: ContextState
    delta: StateDelta


class Evolver:
    """Node B entry point."""

    def __init__(
        self,
        cfg: EngineConfig,
        clock: Callable[[], datetime] = now_est,
    ) -> None:
        self._cfg = cfg
        self._clock = clock

    # ---- public -----------------------------------------------------
    def evolve(
        self,
        parent: ContextState | None,
        incoming: Iterable[Signal],
    ) -> EvolutionResult:
        now = self._clock()
        incoming = tuple(incoming)

        kept, decayed_ids = self._apply_decay(parent, now)
        merged = self._merge_signals(kept, incoming)

        new_events = tuple(self._synthesize_events(parent, merged))
        new_rels = tuple(self._build_relationships(merged))

        version = 1 if parent is None else parent.version + 1
        next_state = ContextState(
            version=version,
            parent_state_id=parent.state_id if parent else None,
            created_at=now,
            signals=merged,
            events=new_events,
            relationships=new_rels,
            metadata={
                "half_life_hours": str(self._cfg.signal_half_life_hours),
            },
        )

        delta = StateDelta(
            added_signal_ids=tuple(s.signal_id for s in incoming),
            removed_signal_ids=decayed_ids,
            added_event_ids=tuple(e.event_id for e in new_events),
            added_relationship_ids=tuple(r.relationship_id for r in new_rels),
            decayed_signal_ids=decayed_ids,
            notes=(
                f"parent_version={parent.version if parent else 0}",
                f"n_kept={len(kept)}",
                f"n_incoming={len(incoming)}",
            ),
        )
        audit(
            "node_b.evolved",
            state_id=str(next_state.state_id),
            version=version,
            added=len(incoming),
            decayed=len(decayed_ids),
            events=len(new_events),
            relationships=len(new_rels),
        )
        return EvolutionResult(state=next_state, delta=delta)

    # ---- decay ------------------------------------------------------
    def _weight(self, observed_at: datetime, now: datetime) -> Decimal:
        """0.5 ** (age_hours / half_life). Age clamped at 0."""
        age_h = Decimal(str(max(0.0, (now - observed_at).total_seconds() / 3600)))
        half = self._cfg.signal_half_life_hours
        # Decimal lacks **Decimal, so use float math and re-wrap to Decimal.
        weight = Decimal(str(0.5 ** float(age_h / half)))
        return weight

    def _apply_decay(
        self, parent: ContextState | None, now: datetime
    ) -> tuple[tuple[Signal, ...], tuple]:
        if parent is None:
            return (), ()
        kept: list[Signal] = []
        decayed: list = []
        for sig in parent.signals:
            if self._weight(sig.observed_at, now) >= _DECAY_FLOOR:
                kept.append(sig)
            else:
                decayed.append(sig.signal_id)
        return tuple(kept), tuple(decayed)

    # ---- merge ------------------------------------------------------
    @staticmethod
    def _merge_signals(
        kept: tuple[Signal, ...], incoming: tuple[Signal, ...]
    ) -> tuple[Signal, ...]:
        """Incoming replaces kept on (source, kind, entity) collisions."""
        by_key: dict[tuple, Signal] = {
            (s.source, s.kind, s.entity): s for s in kept
        }
        for s in incoming:
            by_key[(s.source, s.kind, s.entity)] = s
        # Stable order by observed_at then signal_id for deterministic checksums.
        return tuple(
            sorted(by_key.values(), key=lambda s: (s.observed_at, str(s.signal_id)))
        )

    # ---- events -----------------------------------------------------
    @staticmethod
    def _synthesize_events(
        parent: ContextState | None, merged: tuple[Signal, ...]
    ) -> Iterable[Event]:
        """Detect rate-change events by entity."""
        prev_by_entity: dict[str, Signal] = {}
        if parent is not None:
            for s in parent.signals:
                if s.kind == SignalKind.POLICY_RATE:
                    prev_by_entity[s.entity] = s

        for s in merged:
            if s.kind != SignalKind.POLICY_RATE:
                continue
            prev = prev_by_entity.get(s.entity)
            if prev is None or prev.value_bps == s.value_bps:
                continue
            delta_bps = int(s.value_bps - prev.value_bps)
            direction = "HIKE" if delta_bps > 0 else "CUT"
            yield Event(
                event_type=f"{s.entity}_RATE_{direction}",
                entity=s.entity,
                magnitude_bps=abs(delta_bps),
                occurred_at=s.observed_at,
                source_signal_ids=(prev.signal_id, s.signal_id),
                description=(
                    f"{s.entity} policy rate moved "
                    f"{prev.value_bps}bps → {s.value_bps}bps"
                ),
            )

    # ---- relationships ---------------------------------------------
    @staticmethod
    def _build_relationships(
        merged: tuple[Signal, ...]
    ) -> Iterable[Relationship]:
        """Static rules mapping co-present signals into directed edges.

        The edge *strength* is a qualitative prior from the macro-finance
        literature, not an empirical correlation — it's meant as a
        starting point the validation / disproof nodes can challenge.
        """
        rules: list[tuple[SignalKind, SignalKind, str, Decimal]] = [
            (SignalKind.POLICY_RATE, SignalKind.FX_SPOT, "INFLUENCES", Decimal("0.7")),
            (SignalKind.TREASURY_YIELD, SignalKind.FX_SPOT, "INFLUENCES", Decimal("0.5")),
            (SignalKind.ECON_RELEASE, SignalKind.VOLATILITY, "TRIGGERS", Decimal("0.6")),
            (SignalKind.IMPLIED_RATE_PATH, SignalKind.POLICY_RATE, "PREDICTS", Decimal("0.4")),
            (SignalKind.COT_POSITIONING, SignalKind.FX_SPOT, "LEADS", Decimal("0.3")),
            (SignalKind.FORWARD_GUIDANCE, SignalKind.IMPLIED_RATE_PATH, "SHAPES", Decimal("0.5")),
        ]
        by_kind: dict[SignalKind, list[Signal]] = {}
        for s in merged:
            by_kind.setdefault(s.kind, []).append(s)

        for src_kind, tgt_kind, rtype, strength in rules:
            for src in by_kind.get(src_kind, []):
                for tgt in by_kind.get(tgt_kind, []):
                    if src.entity == tgt.entity:
                        continue
                    yield Relationship(
                        source_entity=src.entity,
                        target_entity=tgt.entity,
                        relationship_type=rtype,
                        strength=strength,
                        observed_at=max(src.observed_at, tgt.observed_at),
                        evidence_signal_ids=(src.signal_id, tgt.signal_id),
                    )
