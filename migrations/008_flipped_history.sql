-- =============================================================================
-- 008_flipped_history.sql
--
-- Relaxes the zones_status_timestamps_consistent CHECK so the
-- ``flipped_direction`` column can carry over when a zone transitions
-- FLIPPED → ACTIVE → CONSUMED.
--
-- Why: PR #38 makes FLIPPED zones tradeable in flipped_direction. Once
-- a setup is created on a flipped zone, the zone moves FLIPPED → ACTIVE
-- and later → CONSUMED. With migration 007's strict CHECK,
-- ``flipped_direction`` had to be NULL in ACTIVE/CONSUMED — which lost
-- the "this zone was once flipped" indicator on the row. Reconstructing
-- that from setup history requires a join; far cleaner to keep it on
-- the row itself.
--
-- The post-relaxation invariants:
--   * status='FLIPPED'  ⇒ flipped_direction IS NOT NULL  (unchanged)
--   * All timestamp NULL/NOT NULL rules per status unchanged.
--   * flipped_direction may be NOT NULL on any status (new).
--
-- IDEMPOTENT — drops the old constraint if present, recreates the
-- relaxed one. Safe to re-run.
-- =============================================================================

-- Drop the strict variant from migration 007.
ALTER TABLE zones DROP CONSTRAINT IF EXISTS zones_status_timestamps_consistent;

-- Re-add the same per-status timestamp rules, but without constraining
-- flipped_direction's NULL/NOT NULL based on status. The narrower
-- "FLIPPED implies flipped_direction IS NOT NULL" rule is a separate
-- CHECK below.
ALTER TABLE zones ADD CONSTRAINT zones_status_timestamps_consistent CHECK (
  (status IN ('CONFIRMED', 'ACTIVE')
     AND consumed_at IS NULL
     AND violated_at IS NULL
     AND flipped_at  IS NULL)
  OR (status = 'CONSUMED'
     AND consumed_at IS NOT NULL
     AND violated_at IS NULL
     AND flipped_at  IS NULL)
  OR (status = 'VIOLATED'
     AND violated_at IS NOT NULL
     AND flipped_at  IS NULL)
  OR (status = 'FLIPPED'
     AND violated_at IS NOT NULL
     AND flipped_at  IS NOT NULL)
);

-- Separate narrower rule: status='FLIPPED' must have flipped_direction set.
-- ACTIVE/CONSUMED zones that were once flipped can keep flipped_direction
-- populated (analytics trail) — that's the whole point of this migration.
ALTER TABLE zones DROP CONSTRAINT IF EXISTS zones_flipped_direction_required_when_flipped;
ALTER TABLE zones ADD CONSTRAINT zones_flipped_direction_required_when_flipped CHECK (
  status != 'FLIPPED' OR flipped_direction IS NOT NULL
);
