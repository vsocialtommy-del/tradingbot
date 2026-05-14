-- =============================================================================
-- 012_zone_exit_close_reasons.sql
--
-- Adds ``ZONE_EXIT`` and ``ZONE_EXIT_CANCELLED`` to the
-- ``trades_close_reason_check`` whitelist for PR #47 (zone-exit BE
-- trigger).
--
-- Rationale: the new zone-exit manager fires on M5 close when the
-- candle's body closes past the trade's L1 entry in the profit
-- direction (BUY: close > L1; SELL: close < L1). On fire it:
--
--   * Closes the shallowest still-FILLED layer with
--     ``close_reason='ZONE_EXIT'``.
--   * Moves SL to entry on every remaining FILLED layer (no close
--     happens here; status stays FILLED, just sl_price patched +
--     broker modify_order).
--   * Cancels every still-WAITING layer with
--     ``close_reason='ZONE_EXIT_CANCELLED'``.
--
-- Why two distinct reasons (vs reusing CASCADE_CANCELLED from
-- migration 011): the TP cascade and the zone-exit trigger are
-- different signals. Operators reading the trade ledger should be
-- able to distinguish "this WAITING layer was cancelled because TP1
-- on a prior layer fired" (CASCADE_CANCELLED) from "this WAITING
-- layer was cancelled because price closed out of the zone before
-- the layer ever filled" (ZONE_EXIT_CANCELLED) — different lifecycle
-- events, different downstream analytics.
--
-- IDEMPOTENT — drops + recreates the CHECK.
-- =============================================================================

ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_close_reason_check;
ALTER TABLE trades ADD CONSTRAINT trades_close_reason_check CHECK (
  close_reason IS NULL OR close_reason IN (
    'TP1', 'TP2', 'TP3',
    'SL_HIT', 'BE_HIT',
    'MANUAL_CLOSE', 'NEWS_CLOSE',
    -- PR #43: WAITING layer cancelled because a previous layer's TP
    -- fired (cascade close on the parent setup).
    'CASCADE_CANCELLED',
    -- PR #47: body close out of the zone (BUY: close > L1 / zone.top;
    -- SELL: close < L1 / zone.bottom) on an M5 bar after entry.
    -- ZONE_EXIT = the shallowest FILLED layer closed at the trigger;
    -- ZONE_EXIT_CANCELLED = a WAITING layer cancelled because the
    -- zone has been confirmed-and-exited (no point waiting for a
    -- deeper retest that may never come).
    'ZONE_EXIT',
    'ZONE_EXIT_CANCELLED'
  )
);
