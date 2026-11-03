-- =============================================================================
-- 015_xautism_layer2_stop.sql
--
-- XAUTISM entries (step 6):
--
--   * Layer 2 is now a broker BUY STOP / SELL STOP at the zone edge, so
--     trades.order_type needs 'STOP'.
--   * New close reason 'M5_CLOSE_EXIT': an M5 candle closed beyond the
--     far edge of the zone, so the bot closed every open layer and
--     cancelled a pending layer 2.
--
-- Run this in the Supabase SQL editor BEFORE starting the bot on this
-- version, otherwise layer 2 rows fail to insert.
--
-- IDEMPOTENT: drops + recreates both CHECKs.
-- =============================================================================

ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_order_type_check;
ALTER TABLE trades ADD CONSTRAINT trades_order_type_check
  CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP'));

ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_close_reason_check;
ALTER TABLE trades ADD CONSTRAINT trades_close_reason_check CHECK (
  close_reason IS NULL OR close_reason IN (
    'TP1', 'TP2', 'TP3',
    'SL_HIT', 'BE_HIT',
    'MANUAL_CLOSE', 'NEWS_CLOSE',
    'CASCADE_CANCELLED',
    'ZONE_EXIT',
    'ZONE_EXIT_CANCELLED',
    'ZONE_DEAD_CANCELLED',
    -- 015: M5 candle closed beyond the far edge of the zone.
    'M5_CLOSE_EXIT'
  )
);
