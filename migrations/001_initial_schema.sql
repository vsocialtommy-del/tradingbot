-- =============================================================================
-- 001_initial_schema.sql
--
-- Initial schema for the XAUUSD Supply & Demand trading bot.
-- Tables: zones, setups, trades, daily_pnl, news_events, bot_logs, bot_config.
-- See docs/strategy-spec-v4-final.md Section 9.2 for the full reference.
--
-- HOW TO RUN:
--   Open Supabase Dashboard → SQL Editor → paste this file → Run.
--   No manual ordering or intervention needed — the file is self-contained.
--
-- IDEMPOTENT:
--   All CREATE statements use IF NOT EXISTS, the trigger function uses
--   CREATE OR REPLACE, triggers use DROP IF EXISTS + CREATE, and the
--   bot_config seed uses ON CONFLICT DO NOTHING. Re-running this file is
--   safe — it will not overwrite existing data or duplicate objects.
--
-- ROW LEVEL SECURITY:
--   Per Phase A decision (b), RLS is enabled on every table with NO
--   policies. That means only the service_role key (used by the bot) can
--   read/write. The dashboard's anon-role policies will land in a
--   follow-up migration during Phase E.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Extensions
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- supplies gen_random_uuid()


-- -----------------------------------------------------------------------------
-- Shared updated_at trigger
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- 1. zones
--
-- A structural finding from the W/M pattern detector, refined to candle
-- bodies. Exists independently of any trade decision — a zone may be
-- found, evaluated, and skipped without ever producing a setup.
-- last_evaluation_result captures the most recent skip reason for that
-- "found 100 zones, only traded 40, why?" backtest analysis.
-- =============================================================================
CREATE TABLE IF NOT EXISTS zones (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  symbol TEXT NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
  zone_type TEXT NOT NULL CHECK (zone_type IN ('STRONG_POINT', 'IMBALANCE')),
  pattern_type TEXT NOT NULL CHECK (pattern_type IN ('W', 'M', 'N')),
  top NUMERIC(10,2) NOT NULL,
  bottom NUMERIC(10,2) NOT NULL,
  approach_count INT NOT NULL DEFAULT 0,
  qualified_imbalance_at TIMESTAMPTZ,
  formed_at TIMESTAMPTZ NOT NULL,
  invalidated_at TIMESTAMPTZ,
  last_evaluation_result JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT zones_top_above_bottom CHECK (top > bottom)
);

CREATE INDEX IF NOT EXISTS idx_zones_active
  ON zones (symbol, invalidated_at)
  WHERE invalidated_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_zones_formed_at
  ON zones (formed_at DESC);

DROP TRIGGER IF EXISTS trg_zones_updated_at ON zones;
CREATE TRIGGER trg_zones_updated_at
  BEFORE UPDATE ON zones
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =============================================================================
-- 2. setups
--
-- A tradeable instance of a zone — created when the bot decides this
-- zone is worth scaling into. Captures planned prices for all three
-- layers + SL + TP1, plus a status lifecycle that drives execution.
-- =============================================================================
CREATE TABLE IF NOT EXISTS setups (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  zone_id UUID NOT NULL REFERENCES zones(id) ON DELETE RESTRICT,
  direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
  entry_mode TEXT NOT NULL CHECK (entry_mode IN (
    'STRONG_POINT_FIRST_TOUCH',
    'IMBALANCE_FIRST_TOUCH'
  )),
  planned_layer1_price NUMERIC(10,2) NOT NULL,
  planned_layer2_price NUMERIC(10,2) NOT NULL,
  planned_layer3_price NUMERIC(10,2) NOT NULL,
  planned_sl_price NUMERIC(10,2) NOT NULL,
  planned_tp1_price NUMERIC(10,2) NOT NULL,
  status TEXT NOT NULL CHECK (status IN (
    'PENDING', 'ACTIVE', 'TP1_HIT', 'CLOSED', 'SKIPPED', 'STOPPED_OUT'
  )),
  skip_reason TEXT,
  activated_at TIMESTAMPTZ,
  closed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_setups_status     ON setups (status);
CREATE INDEX IF NOT EXISTS idx_setups_zone_id    ON setups (zone_id);
CREATE INDEX IF NOT EXISTS idx_setups_created_at ON setups (created_at DESC);

DROP TRIGGER IF EXISTS trg_setups_updated_at ON setups;
CREATE TRIGGER trg_setups_updated_at
  BEFORE UPDATE ON setups
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =============================================================================
-- 3. trades
--
-- One row per filled (or pending) layer. A setup typically produces 1-3
-- trade rows; partial fills mean fewer than 3. mt5_ticket is the broker's
-- position ID (unique across the broker account) and is used to
-- reconcile bot state with MT5's reality.
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  setup_id UUID NOT NULL REFERENCES setups(id) ON DELETE RESTRICT,
  layer_number INT NOT NULL CHECK (layer_number BETWEEN 1 AND 3),
  direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
  order_type TEXT NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT')),
  mt5_ticket BIGINT,
  entry_price NUMERIC(10,2),
  exit_price NUMERIC(10,2),
  lot_size NUMERIC(8,2) NOT NULL,
  sl_price NUMERIC(10,2) NOT NULL,
  tp_price NUMERIC(10,2),
  status TEXT NOT NULL CHECK (status IN (
    'PENDING', 'FILLED', 'PARTIALLY_CLOSED', 'CLOSED', 'CANCELLED'
  )),
  pnl NUMERIC(12,2),
  commission NUMERIC(12,2) NOT NULL DEFAULT 0,
  swap NUMERIC(12,2) NOT NULL DEFAULT 0,
  close_reason TEXT CHECK (
    close_reason IS NULL OR close_reason IN (
      'TP1', 'SL_HIT', 'BE_HIT', 'MANUAL_CLOSE', 'NEWS_CLOSE'
    )
  ),
  filled_at TIMESTAMPTZ,
  closed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_setup_id  ON trades (setup_id);
CREATE INDEX IF NOT EXISTS idx_trades_status    ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_filled_at ON trades (filled_at DESC);
-- Partial unique index — null tickets allowed pre-placement, but no two
-- placed orders may share a broker ticket.
CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_mt5_ticket
  ON trades (mt5_ticket)
  WHERE mt5_ticket IS NOT NULL;

DROP TRIGGER IF EXISTS trg_trades_updated_at ON trades;
CREATE TRIGGER trg_trades_updated_at
  BEFORE UPDATE ON trades
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =============================================================================
-- 4. daily_pnl
--
-- One row per broker trading day (rollover at 17:00 EST per spec
-- Section 7). trading_date stores the date the broker day STARTED; e.g.
-- '2026-05-06' covers Tue 17:00 EST through Wed 17:00 EST.
-- =============================================================================
CREATE TABLE IF NOT EXISTS daily_pnl (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  trading_date DATE NOT NULL UNIQUE,
  starting_balance NUMERIC(12,2) NOT NULL,
  ending_balance NUMERIC(12,2),
  realized_pnl NUMERIC(12,2) NOT NULL DEFAULT 0,
  trade_count INT NOT NULL DEFAULT 0,
  winning_trades INT NOT NULL DEFAULT 0,
  losing_trades INT NOT NULL DEFAULT 0,
  halted_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl (trading_date DESC);

DROP TRIGGER IF EXISTS trg_daily_pnl_updated_at ON daily_pnl;
CREATE TRIGGER trg_daily_pnl_updated_at
  BEFORE UPDATE ON daily_pnl
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- =============================================================================
-- 5. news_events
--
-- High-impact event calendar — populated by Vercel cron, read by bot
-- (spec Section 8). Unique constraint prevents the cron from inserting
-- duplicates when re-fetching the same window.
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_time TIMESTAMPTZ NOT NULL,
  currency TEXT NOT NULL,
  title TEXT NOT NULL,
  impact_level TEXT NOT NULL CHECK (impact_level IN ('HIGH', 'MEDIUM', 'LOW')),
  forecast TEXT,
  actual TEXT,
  fetched_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT news_events_event_time_title_unique UNIQUE (event_time, title)
);

CREATE INDEX IF NOT EXISTS idx_news_events_time ON news_events (event_time);
-- Partial index for the bot's hot-path: "any high-impact USD event near
-- now?" — avoids a scan over MEDIUM/LOW rows the bot ignores anyway.
CREATE INDEX IF NOT EXISTS idx_news_events_high_usd
  ON news_events (event_time)
  WHERE impact_level = 'HIGH' AND currency = 'USD';


-- =============================================================================
-- 6. bot_logs
--
-- Structured event log. Immutable — no updated_at, no UPDATE expected.
-- FKs are nullable + ON DELETE SET NULL because retaining a log row
-- after the related setup/trade is deleted is acceptable.
-- =============================================================================
CREATE TABLE IF NOT EXISTS bot_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  level TEXT NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARN', 'ERROR')),
  message TEXT NOT NULL,
  context JSONB NOT NULL DEFAULT '{}'::jsonb,
  setup_id UUID REFERENCES setups(id) ON DELETE SET NULL,
  trade_id UUID REFERENCES trades(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bot_logs_created_at
  ON bot_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_logs_level_created
  ON bot_logs (level, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_bot_logs_setup_id
  ON bot_logs (setup_id) WHERE setup_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_bot_logs_trade_id
  ON bot_logs (trade_id) WHERE trade_id IS NOT NULL;


-- =============================================================================
-- 7. bot_config
--
-- Key/value config table. The dashboard writes (kill switch, parameter
-- tuning), the bot polls every ~30 s.
--
-- DELIBERATE EXCEPTION TO THE GLOBAL UUID-PK RULE:
--   This table uses `key TEXT PRIMARY KEY` instead of UUID + UNIQUE(key).
--   Reason: every read and write is by key, never by UUID. Adding a UUID
--   column would create two indexes for no benefit and slow the kill-
--   switch hot path. This is the textbook key/value-table exception.
-- =============================================================================
CREATE TABLE IF NOT EXISTS bot_config (
  key TEXT PRIMARY KEY,
  value JSONB NOT NULL,
  description TEXT,
  updated_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DROP TRIGGER IF EXISTS trg_bot_config_updated_at ON bot_config;
CREATE TRIGGER trg_bot_config_updated_at
  BEFORE UPDATE ON bot_config
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Seed defaults. ON CONFLICT DO NOTHING means the operator's manual
-- changes survive a re-run of this migration.
INSERT INTO bot_config (key, value, description) VALUES
  ('kill_switch', 'false'::jsonb,
   'Master bot pause. true = halt all new entries; in-flight trades continue to be managed.'),
  ('pause_until', 'null'::jsonb,
   'ISO-8601 timestamp; bot ignores new setups until this time. Used by the dashboard "Pause for X minutes" button.'),
  ('pattern_tolerance_pct', '0.1'::jsonb,
   'Max % difference between two swing lows/highs to qualify as a W/M (spec Section 13).'),
  ('recent_swing_lookback', '20'::jsonb,
   '1H candles to look back for the structural low/high used in SL placement (spec Section 5.1).'),
  ('imbalance_approach_distance', '7.5'::jsonb,
   'Points within zone top/bottom that count as a "failed approach" for imbalance qualification (spec Section 3.2).'),
  ('imbalance_approach_threshold', '2'::jsonb,
   'Failed approaches required to promote a Strong Point to an Imbalance Zone (spec Section 3.2).'),
  ('sl_buffer_points', '17.5'::jsonb,
   'Anti-stop-hunt buffer added to the structural swing for SL placement (spec Section 5.1).'),
  ('tp1_distance_dollars', '4'::jsonb,
   'Distance beyond zone edge for TP1 (spec Section 6.1, default $4, optimised in $2-10 range).'),
  ('zone_min_size_points', '5'::jsonb,
   'Skip setups if refined zone is narrower than this (spec Section 7).'),
  ('zone_max_size_points', '80'::jsonb,
   'Skip setups if refined zone is wider than this (spec Section 7).'),
  ('max_simultaneous_setups', '3'::jsonb,
   'Concurrent live-setup cap (spec Section 7).'),
  ('daily_loss_limit_pct', '10'::jsonb,
   'Daily halt threshold as % of starting balance (spec Section 7).'),
  ('lot_size', '0.01'::jsonb,
   'Fixed lot size per layer in v1 (spec Section 4.4); v1.1 switches to risk-based sizing.'),
  ('news_blackout_minutes_before', '30'::jsonb,
   'Minutes before a high-impact USD event when bot stops opening setups (spec Section 8.3).'),
  ('news_blackout_minutes_after', '15'::jsonb,
   'Minutes after event before bot resumes (spec Section 8.3).')
ON CONFLICT (key) DO NOTHING;


-- =============================================================================
-- Row Level Security
--
-- Enable on every table. With NO policies attached, only the
-- service_role key (which bypasses RLS) can read/write. The bot uses
-- service_role; the dashboard's anon-role policies will be added in a
-- Phase E migration.
-- =============================================================================
ALTER TABLE zones        ENABLE ROW LEVEL SECURITY;
ALTER TABLE setups       ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades       ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_pnl    ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_events  ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_logs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_config   ENABLE ROW LEVEL SECURITY;


-- =============================================================================
-- Realtime (deferred)
--
-- Adding tables to the supabase_realtime publication enables the
-- dashboard's live-trade / kill-switch subscriptions. Held back to the
-- Phase E migration so realtime config lands alongside the dashboard
-- code that consumes it. Reference for that follow-up:
--   ALTER PUBLICATION supabase_realtime ADD TABLE trades, setups, zones, bot_config;
-- =============================================================================
