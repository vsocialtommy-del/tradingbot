-- ============================================================================
-- 006_pattern_type_snd_codes.sql
--
-- Relax the CHECK constraint on ``zones.pattern_type`` to accept the four
-- Supply & Demand pattern codes alongside the legacy W/M/N values.
--
-- Why
-- ---
-- PR #31 pivoted the strategy from W/M pattern detection to institutional
-- S&D (RBR / DBD / DBR / RBD). v1 mapped the new codes to W (BUY) / M
-- (SELL) at the persistence boundary to satisfy the old CHECK constraint,
-- but that loses the distinction between:
--   * RBR (rally-base-rally → continuation BUY)
--   * DBR (drop-base-rally  → reversal BUY)
--   * DBD (drop-base-drop   → continuation DBD)
--   * RBD (rally-base-drop  → reversal SELL)
--
-- Tommy's demo-trading analytics need this distinction to compare
-- continuation vs reversal pattern performance. This migration drops
-- the old constraint and adds a new one that accepts the full set.
--
-- Backward compat
-- ---------------
-- The W / M / N legacy codes remain valid — any rows already written
-- with those values are untouched. Going forward the bot writes
-- the real S&D code (RBR / DBD / DBR / RBD).
--
-- Idempotent: ``DROP CONSTRAINT IF EXISTS`` + ``ADD CONSTRAINT`` with a
-- versioned name. Safe to re-run.
-- ============================================================================

ALTER TABLE zones
  DROP CONSTRAINT IF EXISTS zones_pattern_type_check;

ALTER TABLE zones
  ADD CONSTRAINT zones_pattern_type_check
  CHECK (pattern_type IN (
    -- Legacy W/M era (preserved for back-compat on existing rows).
    'W', 'M', 'N',
    -- S&D codes (PR #31 onward).
    'RBR', 'DBD', 'DBR', 'RBD'
  ));
