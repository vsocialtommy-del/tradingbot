-- =============================================================================
-- 010_relax_active_timestamps.sql
--
-- Production regression after PR #38 / migration 008: the
-- ``zones_status_timestamps_consistent`` CHECK required ACTIVE rows
-- to have ``violated_at`` and ``flipped_at`` NULL. That conflicts with
-- the SnD Flip trade path (FLIPPED → ACTIVE) because those timestamps
-- ARE set on the row — they represent the zone's history. PR #38's
-- supabase_logger worked around this by emitting explicit NULLs to
-- clear the timestamps on the ACTIVE transition, but that destroys
-- the audit trail (you can no longer tell from the zone row alone
-- whether an ACTIVE / CONSUMED zone went through a flip).
--
-- Real fix: relax the CHECK to preserve the timestamps as history
-- markers. The companion code change (supabase_logger.update_zone_status)
-- stops the explicit-NULL clearing; once both ship, an ACTIVE /
-- CONSUMED zone that came via the flip path retains its full
-- timeline: violated_at, flipped_at, flipped_direction all populated.
--
-- New invariants
-- --------------
--   CONFIRMED  all three lifecycle timestamps NULL (fresh state)
--   ACTIVE     no timestamp NULL/NOT NULL requirements
--   CONSUMED   consumed_at NOT NULL (still required); others unrestricted
--   VIOLATED   violated_at NOT NULL; flipped_at NULL
--   FLIPPED    violated_at NOT NULL AND flipped_at NOT NULL
--   Separate narrower CHECK (unchanged): status='FLIPPED' implies
--                                        flipped_direction IS NOT NULL.
--
-- IDEMPOTENT — drops the old constraint if present, recreates the
-- relaxed one. Safe to re-run.
-- =============================================================================

ALTER TABLE zones DROP CONSTRAINT IF EXISTS zones_status_timestamps_consistent;

ALTER TABLE zones ADD CONSTRAINT zones_status_timestamps_consistent CHECK (
  (status = 'CONFIRMED'
     AND consumed_at IS NULL
     AND violated_at IS NULL
     AND flipped_at  IS NULL)
  -- ACTIVE has no timestamp restrictions — preserves history across
  -- FLIPPED → ACTIVE transitions for analytics.
  OR (status = 'ACTIVE')
  OR (status = 'CONSUMED'
     AND consumed_at IS NOT NULL)
  OR (status = 'VIOLATED'
     AND violated_at IS NOT NULL
     AND flipped_at  IS NULL)
  OR (status = 'FLIPPED'
     AND violated_at IS NOT NULL
     AND flipped_at  IS NOT NULL)
);

-- The narrower "status=FLIPPED ⇒ flipped_direction IS NOT NULL" rule
-- was added in migration 008 and stays unchanged. No need to touch
-- ``zones_flipped_direction_required_when_flipped`` here.
