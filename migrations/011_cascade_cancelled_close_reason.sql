-- =============================================================================
-- 011_cascade_cancelled_close_reason.sql
--
-- Adds ``CASCADE_CANCELLED`` to the ``trades_close_reason_check`` whitelist.
--
-- Production bug after PR #41: when a setup's TP1 fired, ``tp_manager``
-- updated ``sl_price`` on every remaining layer (FILLED + WAITING) to
-- the closed layer's entry price. For BUY setups this puts the new SL
-- ABOVE Layer 2 / Layer 3 entries — fine for FILLED positions (broker
-- accepts it as a trailing-stop-style move) but **invalid** for a
-- WAITING layer when ``entry_trigger`` later fires it as a market
-- order. MT5 returns retcode 10016 'Invalid stops' (BUY market
-- requires SL < entry; SELL mirror).
--
-- Fix: WAITING layers get **cancelled** when a previous layer's TP
-- fires, rather than having their SL patched. This new ``close_reason``
-- distinguishes that path from MANUAL_CLOSE (operator action) /
-- SL_HIT / TP* / NEWS_CLOSE for clean analytics.
--
-- IDEMPOTENT — drops + recreates the CHECK.
-- =============================================================================

ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_close_reason_check;
ALTER TABLE trades ADD CONSTRAINT trades_close_reason_check CHECK (
  close_reason IS NULL OR close_reason IN (
    'TP1', 'TP2', 'TP3',
    'SL_HIT', 'BE_HIT',
    'MANUAL_CLOSE', 'NEWS_CLOSE',
    -- PR #43: a WAITING layer cancelled because a previous layer's
    -- TP fired (cascade close on the parent setup).
    'CASCADE_CANCELLED'
  )
);
