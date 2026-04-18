"""Node C — Validation.

Three checks, each a small method so any one can be toggled, replaced,
or audited in isolation:

1. **Rate conflicts** — two sources claiming different values for the
   same policy rate / treasury yield, outside tolerance.
2. **Temporal consistency** — no signal observed in the future relative
   to the state's ``created_at``; no event whose source signals predate
   the event itself.
3. **Spread sanity** — FX spreads must be non-negative and implausibly
   large spreads are flagged (liquidity shock or bad data).

A state ``passes`` iff there are no ``CRITICAL`` conflicts. ``HIGH``
conflicts are returned for the orchestrator to decide policy.
"""
from __future__ import annotations

from collections import defaultdict

from .config import EngineConfig
from .logging_setup import audit
from .models import (
    Conflict,
    ContextState,
    Severity,
    Signal,
    SignalKind,
    ValidationResult,
)


# Above this spread (in bps) we assume something broke (dislocation or bad feed).
_SPREAD_SANITY_CEILING_BPS = 200


class Validator:
    def __init__(self, cfg: EngineConfig) -> None:
        self._cfg = cfg

    def validate(self, state: ContextState) -> ValidationResult:
        conflicts: list[Conflict] = []
        warnings: list[str] = []

        conflicts.extend(self._check_rate_conflicts(state))
        warnings.extend(self._check_temporal_consistency(state))
        conflicts.extend(self._check_spread_sanity(state))

        passed = not any(c.severity == Severity.CRITICAL for c in conflicts)
        result = ValidationResult(
            state_id=state.state_id,
            passed=passed,
            conflicts=tuple(conflicts),
            warnings=tuple(warnings),
        )
        audit(
            "node_c.validated",
            state_id=str(state.state_id),
            passed=passed,
            n_conflicts=len(conflicts),
            n_warnings=len(warnings),
        )
        return result

    # ---- rate conflicts -----------------------------------------------------
    def _check_rate_conflicts(
        self, state: ContextState
    ) -> list[Conflict]:
        """Group rate-type signals by (kind, entity); flag disagreements."""
        rate_kinds = {
            SignalKind.POLICY_RATE,
            SignalKind.TREASURY_YIELD,
            SignalKind.REAL_YIELD,
        }
        grouped: dict[tuple, list[Signal]] = defaultdict(list)
        for s in state.signals:
            if s.kind in rate_kinds:
                grouped[(s.kind, s.entity)].append(s)

        conflicts: list[Conflict] = []
        for (kind, entity), sigs in grouped.items():
            if len(sigs) < 2:
                continue
            lo = min(s.value_bps for s in sigs)
            hi = max(s.value_bps for s in sigs)
            if hi - lo > self._cfg.rate_conflict_tolerance_bps:
                conflicts.append(
                    Conflict(
                        conflict_type="rate_disagreement",
                        description=(
                            f"{kind.value} on {entity}: sources disagree "
                            f"by {hi - lo}bps (tolerance="
                            f"{self._cfg.rate_conflict_tolerance_bps}bps)"
                        ),
                        signal_ids=tuple(s.signal_id for s in sigs),
                        severity=Severity.CRITICAL if hi - lo > 10 else Severity.HIGH,
                    )
                )
        return conflicts

    # ---- temporal consistency ----------------------------------------------
    @staticmethod
    def _check_temporal_consistency(
        state: ContextState,
    ) -> list[str]:
        """Warnings, not conflicts — these can't falsify the inference
        but the quant desk wants to see them in the audit log."""
        warns: list[str] = []
        for s in state.signals:
            if s.observed_at > state.created_at:
                warns.append(
                    f"signal {s.signal_id} observed_at > state.created_at"
                )
        sig_by_id = {s.signal_id: s for s in state.signals}
        for ev in state.events:
            for sid in ev.source_signal_ids:
                if sid in sig_by_id:
                    s = sig_by_id[sid]
                    if s.observed_at > ev.occurred_at:
                        warns.append(
                            f"event {ev.event_id} cites signal {sid} "
                            f"observed after event time"
                        )
        return warns

    # ---- spread sanity ------------------------------------------------------
    @staticmethod
    def _check_spread_sanity(
        state: ContextState,
    ) -> list[Conflict]:
        conflicts: list[Conflict] = []
        for s in state.signals:
            if s.kind != SignalKind.FX_SPREAD:
                continue
            if s.value_bps is None or s.value_bps < 0:
                conflicts.append(
                    Conflict(
                        conflict_type="negative_spread",
                        description=f"{s.entity} spread is negative/null",
                        signal_ids=(s.signal_id,),
                        severity=Severity.CRITICAL,
                    )
                )
            elif s.value_bps > _SPREAD_SANITY_CEILING_BPS:
                conflicts.append(
                    Conflict(
                        conflict_type="implausible_spread",
                        description=(
                            f"{s.entity} spread {s.value_bps}bps exceeds "
                            f"sanity ceiling {_SPREAD_SANITY_CEILING_BPS}bps"
                        ),
                        signal_ids=(s.signal_id,),
                        severity=Severity.HIGH,
                    )
                )
        return conflicts
