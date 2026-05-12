-- =============================================================================
-- 007_zone_lifecycle.sql
--
-- Formalises the zone state machine that was previously implicit. Adds a
-- ``status`` column with the five persisted states + matching timestamps,
-- and a ``flipped_direction`` column for setup #4 (SnD Flip) groundwork.
--
-- States (persisted; FRESH / TRADEABLE remain in-memory only)
--   CONFIRMED  Strong Point validated; eligible for retest.
--   ACTIVE     ≥1 setup created on this zone (PENDING insert).
--   CONSUMED   Price entered the zone bounds (any high/low touch).
--              Permanent in the original direction. Per design decision Q1,
--              CONSUMED on first touch — fill-agnostic.
--   VIOLATED   Body close past the wrong-side zone bound. Can be reached
--              from CONFIRMED, ACTIVE, or CONSUMED (Q3 — CONSUMED zones can
--              still violate and later flip).
--   FLIPPED    From VIOLATED, body close past the nearest opposite-side
--              swing (BoS). Terminal. ``flipped_direction`` carries the
--              new direction (opposite of the original).
--
-- Allowed transitions:
--   CONFIRMED → ACTIVE / CONSUMED / VIOLATED
--   ACTIVE    → CONSUMED / VIOLATED
--   CONSUMED  → VIOLATED
--   VIOLATED  → FLIPPED
--   FLIPPED   → (terminal)
--
-- Backfill of existing rows is left manual rather than auto-run on
-- migration apply (we don't want to silently rewrite history). A sample
-- snippet that walks zones → setups → terminal status is at the bottom
-- of this file in a comment block.
--
-- The pre-existing ``invalidated_at`` column (migration 001) is unused
-- in code today and is kept for back-compat; new code reads/writes
-- ``violated_at`` / ``flipped_at`` instead.
--
-- IDEMPOTENT — uses ADD COLUMN IF NOT EXISTS + named constraints.
-- =============================================================================

ALTER TABLE zones ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'CONFIRMED';

-- The status CHECK constraint is added separately so we can drop+recreate
-- if we later expand the enum (a CHECK inside ADD COLUMN is harder to
-- migrate). DROP IF EXISTS keeps re-runs idempotent.
ALTER TABLE zones DROP CONSTRAINT IF EXISTS zones_status_check;
ALTER TABLE zones ADD CONSTRAINT zones_status_check
  CHECK (status IN ('CONFIRMED', 'ACTIVE', 'CONSUMED', 'VIOLATED', 'FLIPPED'));

ALTER TABLE zones ADD COLUMN IF NOT EXISTS consumed_at      TIMESTAMPTZ;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS violated_at      TIMESTAMPTZ;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS flipped_at       TIMESTAMPTZ;
ALTER TABLE zones ADD COLUMN IF NOT EXISTS flipped_direction TEXT;

ALTER TABLE zones DROP CONSTRAINT IF EXISTS zones_flipped_direction_check;
ALTER TABLE zones ADD CONSTRAINT zones_flipped_direction_check
  CHECK (flipped_direction IS NULL OR flipped_direction IN ('BUY', 'SELL'));

-- Status ↔ timestamp consistency. Each status implies which timestamp
-- columns must / must not be set. Keeping this as a CHECK means a bug
-- in the application layer (e.g. writing FLIPPED without flipped_at)
-- fails loudly at the DB rather than silently desyncing analytics.
ALTER TABLE zones DROP CONSTRAINT IF EXISTS zones_status_timestamps_consistent;
ALTER TABLE zones ADD CONSTRAINT zones_status_timestamps_consistent CHECK (
  (status IN ('CONFIRMED', 'ACTIVE')
     AND consumed_at IS NULL
     AND violated_at IS NULL
     AND flipped_at  IS NULL
     AND flipped_direction IS NULL)
  OR (status = 'CONSUMED'
     AND consumed_at IS NOT NULL
     AND violated_at IS NULL
     AND flipped_at  IS NULL
     AND flipped_direction IS NULL)
  OR (status = 'VIOLATED'
     AND violated_at IS NOT NULL
     AND flipped_at  IS NULL
     AND flipped_direction IS NULL)
  OR (status = 'FLIPPED'
     AND violated_at IS NOT NULL
     AND flipped_at  IS NOT NULL
     AND flipped_direction IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_zones_status
  ON zones (status);

-- Partial index for the FLIP scanner: it only needs to look at zones in
-- VIOLATED state to decide whether a subsequent BoS confirms the flip.
CREATE INDEX IF NOT EXISTS idx_zones_violated_pending_flip
  ON zones (id) WHERE status = 'VIOLATED';

-- =============================================================================
-- OPTIONAL BACKFILL (manual — copy into SQL Editor if desired).
-- Skipped automatically so this migration is safe to re-run without
-- rewriting history.
--
--   UPDATE zones z
--     SET status = 'CONSUMED', consumed_at = COALESCE(s.closed_at, NOW())
--     FROM setups s
--    WHERE s.zone_id = z.id
--      AND s.status IN ('TP1_HIT', 'CLOSED', 'STOPPED_OUT')
--      AND z.status = 'CONFIRMED';
--
--   UPDATE zones z
--     SET status = 'ACTIVE'
--     FROM setups s
--    WHERE s.zone_id = z.id
--      AND s.status = 'ACTIVE'
--      AND z.status = 'CONFIRMED';
--
-- =============================================================================
