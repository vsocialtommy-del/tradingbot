-- =============================================================================
-- 009_per_layer_tps.sql
--
-- Per-layer take-profit schema (PR #41). Each of the three layers now
-- has its own TP price, hit independently and closed by ``tp_manager``
-- with cascading SL protection. The old "partial close at TP1 + move
-- all SLs to BE" model didn't work at 0.01 lots (broker rejects
-- partial close).
--
-- New columns
--   setups.planned_tp2_price  NUMERIC(10,2)  NULL
--   setups.planned_tp3_price  NUMERIC(10,2)  NULL
--
-- Both nullable: at setup creation we compute as many TPs as the
-- lookback yields. TP1 is required; TP2 / TP3 are best-effort. NULL
-- values are recomputed at the previous layer's TP hit time via
-- ``tp_target.find_nearest_local_peak`` against the then-current bars.
--
-- Trade close_reason
--   Add 'TP2', 'TP3'. The existing CHECK is dropped + recreated.
--   Existing rows are valid under the new CHECK (TP1/SL_HIT/BE_HIT/
--   MANUAL_CLOSE/NEWS_CLOSE all retained).
--
-- IDEMPOTENT — ADD COLUMN IF NOT EXISTS + DROP/ADD CONSTRAINT.
-- =============================================================================

ALTER TABLE setups ADD COLUMN IF NOT EXISTS planned_tp2_price NUMERIC(10,2);
ALTER TABLE setups ADD COLUMN IF NOT EXISTS planned_tp3_price NUMERIC(10,2);

ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_close_reason_check;
ALTER TABLE trades ADD CONSTRAINT trades_close_reason_check CHECK (
  close_reason IS NULL OR close_reason IN (
    'TP1', 'TP2', 'TP3',
    'SL_HIT', 'BE_HIT',
    'MANUAL_CLOSE', 'NEWS_CLOSE'
  )
);
