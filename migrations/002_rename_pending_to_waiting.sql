-- =============================================================================
-- 002_rename_pending_to_waiting.sql
--
-- Strategy change (May 2026): Layers 2 and 3 are no longer placed at the
-- broker as pending limit orders. They're bot-managed market orders fired
-- when the live tick reaches the trigger price. The trade-status value
-- "PENDING" (semantically: "broker has the limit order, awaiting fill")
-- becomes "WAITING" (semantically: "bot is watching for the trigger
-- price"). Different concept; deserves a different name.
--
-- This migration is `trades` only. ``setups.status`` keeps PENDING with
-- its original meaning ("setup created, awaiting Layer 1 fill") — that's
-- a different state machine.
--
-- IDEMPOTENT: data migration uses WHERE status = 'PENDING' (no-op on
-- re-run). Constraint replacement uses DROP IF EXISTS.
--
-- HOW TO RUN:
--   Open Supabase Dashboard → SQL Editor → paste this file → Run.
-- =============================================================================

-- Step 1: migrate any existing PENDING rows to WAITING.
UPDATE trades SET status = 'WAITING' WHERE status = 'PENDING';

-- Step 2: replace the CHECK constraint.
-- Postgres auto-names column-level constraints as <table>_<column>_check.
-- DROP IF EXISTS makes this safe to re-run.
ALTER TABLE trades DROP CONSTRAINT IF EXISTS trades_status_check;
ALTER TABLE trades ADD CONSTRAINT trades_status_check
    CHECK (status IN (
        'WAITING', 'FILLED', 'PARTIALLY_CLOSED', 'CLOSED', 'CANCELLED'
    ));
