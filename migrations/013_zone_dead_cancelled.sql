-- =============================================================================
-- 013_zone_dead_cancelled.sql
--
-- Adds ``ZONE_DEAD_CANCELLED`` to the ``trades_close_reason_check``
-- whitelist for PR #48 (zone visibility gate).
--
-- PR #48 introduces a per-tick visibility check: a zone is dead once
-- 2 or more candles have bodied through it since formation. The
-- pipeline placement gate refuses to place new setups on dead zones,
-- and the per-tick ``_run_zone_death_pass`` cancels any WAITING
-- layers on already-placed setups whose zones have died.
--
-- ZONE_DEAD_CANCELLED distinguishes that path from the existing
-- cancel reasons:
--
--   * CASCADE_CANCELLED (PR #43 / migration 011) — WAITING cancelled
--     because a previous layer's TP fired on the parent setup.
--   * ZONE_EXIT_CANCELLED (PR #47 / migration 012) — WAITING cancelled
--     because the M5 close body-confirmed zone exit in the profit
--     direction.
--   * ZONE_DEAD_CANCELLED (this migration) — WAITING cancelled
--     because 2+ candles have bodied through the zone, regardless of
--     direction. The zone is obscured / no longer respected by price
--     action.
--
-- Operators can now distinguish "killed because TP cascade" /
-- "killed because confirmed exit" / "killed because zone obscured"
-- in the trade ledger.
--
-- IDEMPOTENT — drops + recreates the CHECK.
-- =============================================================================

ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_close_reason_check;
ALTER TABLE trades ADD CONSTRAINT trades_close_reason_check CHECK (
  close_reason IS NULL OR close_reason IN (
    'TP1', 'TP2', 'TP3',
    'SL_HIT', 'BE_HIT',
    'MANUAL_CLOSE', 'NEWS_CLOSE',
    'CASCADE_CANCELLED',
    'ZONE_EXIT',
    'ZONE_EXIT_CANCELLED',
    -- PR #48: WAITING layer cancelled because 2+ candles have bodied
    -- through the parent zone since formation (visibility rule).
    'ZONE_DEAD_CANCELLED'
  )
);
