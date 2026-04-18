"""Node D — Persistence.

``StateRepository`` is a Protocol. Two concrete implementations ship
with the engine:

- ``PostgresStateRepository`` — ACID, versioned, with an append-only
  audit-trail table.
- ``InMemoryStateRepository`` — deterministic, dependency-free, used by
  unit tests and the worked example.

Swapping DBs is a one-line change at the orchestrator's construction
site; no other node sees the backing store.
"""
from __future__ import annotations

from typing import Protocol
from uuid import UUID

from .exceptions import PersistenceError
from .logging_setup import audit
from .models import ContextState, StateDelta, ValidationResult


class StateRepository(Protocol):
    def save(
        self,
        state: ContextState,
        delta: StateDelta,
        validation: ValidationResult | None = None,
    ) -> None: ...

    def load(self, state_id: UUID) -> ContextState: ...

    def latest(self) -> ContextState | None: ...


# --------------------------------------------------------------------------- #
# In-memory                                                                   #
# --------------------------------------------------------------------------- #
class InMemoryStateRepository:
    """Ordered, immutable log of states keyed by state_id."""

    def __init__(self) -> None:
        self._by_id: dict[UUID, ContextState] = {}
        self._order: list[UUID] = []
        self._audit: list[dict] = []

    def save(
        self,
        state: ContextState,
        delta: StateDelta,
        validation: ValidationResult | None = None,
    ) -> None:
        if state.state_id in self._by_id:
            raise PersistenceError(
                "state_id already persisted — states are immutable",
                state_id=str(state.state_id),
            )
        if state.parent_state_id and state.parent_state_id not in self._by_id:
            raise PersistenceError(
                "parent state not found",
                parent_state_id=str(state.parent_state_id),
            )
        self._by_id[state.state_id] = state
        self._order.append(state.state_id)
        self._audit.append({
            "state_id": str(state.state_id),
            "parent_state_id": str(state.parent_state_id)
                if state.parent_state_id else None,
            "version": state.version,
            "checksum": state.checksum(),
            "delta": delta.model_dump(mode="json"),
            "validation_passed": validation.passed if validation else None,
            "created_at": state.created_at.isoformat(),
        })
        audit(
            "node_d.saved.memory",
            state_id=str(state.state_id),
            version=state.version,
            checksum=state.checksum(),
        )

    def load(self, state_id: UUID) -> ContextState:
        try:
            return self._by_id[state_id]
        except KeyError as exc:
            raise PersistenceError(
                "state not found", state_id=str(state_id)
            ) from exc

    def latest(self) -> ContextState | None:
        if not self._order:
            return None
        return self._by_id[self._order[-1]]

    # Test convenience — not on the Protocol.
    def audit_trail(self) -> list[dict]:
        return list(self._audit)


# --------------------------------------------------------------------------- #
# PostgreSQL                                                                  #
# --------------------------------------------------------------------------- #
class PostgresStateRepository:
    """psycopg3-backed repository.

    ``psycopg`` is imported lazily so the rest of the engine has no hard
    dependency on it — the in-memory repo works without psycopg installed.
    The schema lives in ``sql/001_init.sql``.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg                               # noqa: F401
        except ImportError as exc:                        # pragma: no cover
            raise PersistenceError(
                "psycopg is required for PostgresStateRepository — "
                "install with `pip install psycopg[binary]`"
            ) from exc
        self._dsn = dsn

    # Separate method so tests can monkeypatch the connect call.
    def _connect(self):
        import psycopg
        return psycopg.connect(self._dsn, autocommit=False)

    def save(
        self,
        state: ContextState,
        delta: StateDelta,
        validation: ValidationResult | None = None,
    ) -> None:
        payload = state.model_dump(mode="json")
        checksum = state.checksum()
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO context_state
                        (state_id, version, parent_state_id, created_at,
                         checksum, payload)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(state.state_id),
                            state.version,
                            str(state.parent_state_id)
                                if state.parent_state_id else None,
                            state.created_at,
                            checksum,
                            self._to_jsonb(payload),
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO context_state_audit
                        (state_id, checksum, delta, validation_passed,
                         recorded_at)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            str(state.state_id),
                            checksum,
                            self._to_jsonb(delta.model_dump(mode="json")),
                            validation.passed if validation else None,
                            state.created_at,
                        ),
                    )
                conn.commit()
        except Exception as exc:                         # pragma: no cover
            raise PersistenceError(
                "failed to persist state",
                state_id=str(state.state_id),
                cause=repr(exc),
            ) from exc
        audit(
            "node_d.saved.postgres",
            state_id=str(state.state_id),
            version=state.version,
            checksum=checksum,
        )

    def load(self, state_id: UUID) -> ContextState:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT payload FROM context_state WHERE state_id = %s",
                        (str(state_id),),
                    )
                    row = cur.fetchone()
        except Exception as exc:                         # pragma: no cover
            raise PersistenceError(
                "failed to load state",
                state_id=str(state_id),
                cause=repr(exc),
            ) from exc
        if row is None:
            raise PersistenceError("state not found", state_id=str(state_id))
        return ContextState.model_validate(row[0])

    def latest(self) -> ContextState | None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT payload FROM context_state
                        ORDER BY version DESC, created_at DESC LIMIT 1
                        """
                    )
                    row = cur.fetchone()
        except Exception as exc:                         # pragma: no cover
            raise PersistenceError(
                "failed to load latest state", cause=repr(exc)
            ) from exc
        return ContextState.model_validate(row[0]) if row else None

    @staticmethod
    def _to_jsonb(obj):                                   # pragma: no cover
        # psycopg3 adapts dicts to jsonb natively; this wrapper exists so
        # tests using fakes can override serialization behaviour cheaply.
        import json as _json
        return _json.dumps(obj, default=str)
