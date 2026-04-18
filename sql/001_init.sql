-- Forex Context Engine — initial schema.
--
-- Design:
--  * One row per ContextState version. States are immutable.
--  * Integrity enforced at the DB boundary (UNIQUE on version,
--    CHECK on version >= 1, FK to parent).
--  * Audit trail is an append-only companion table. One row per save.
--
-- Run:
--   psql "$FOREX_PG_DSN" -f sql/001_init.sql

BEGIN;

CREATE TABLE IF NOT EXISTS context_state (
    state_id          UUID PRIMARY KEY,
    version           INTEGER NOT NULL CHECK (version >= 1),
    parent_state_id   UUID NULL
        REFERENCES context_state(state_id)
        ON DELETE RESTRICT,
    created_at        TIMESTAMPTZ NOT NULL,
    checksum          TEXT NOT NULL,
    payload           JSONB NOT NULL,
    CONSTRAINT context_state_version_unique UNIQUE (version),
    CONSTRAINT context_state_genesis_parent CHECK (
        (version = 1 AND parent_state_id IS NULL)
        OR (version > 1 AND parent_state_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS context_state_created_at_idx
    ON context_state (created_at DESC);

CREATE TABLE IF NOT EXISTS context_state_audit (
    audit_id           BIGSERIAL PRIMARY KEY,
    state_id           UUID NOT NULL
        REFERENCES context_state(state_id)
        ON DELETE RESTRICT,
    checksum           TEXT NOT NULL,
    delta              JSONB NOT NULL,
    validation_passed  BOOLEAN NULL,
    recorded_at        TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS context_state_audit_state_id_idx
    ON context_state_audit (state_id);

-- No UPDATE / DELETE on either table — enforce at role level:
-- REVOKE UPDATE, DELETE ON context_state, context_state_audit FROM forex_app;

COMMIT;
