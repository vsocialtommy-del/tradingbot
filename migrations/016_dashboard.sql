-- 016: dashboard (Nov 2026)
--
-- Run once in EACH bot's Supabase project (M5 bot and M1 bot). Safe to
-- run again: everything is IF NOT EXISTS / ON CONFLICT DO NOTHING.
--
--   bot_status      one row per bot, written by the bot every ~10 s
--                   (balance, today, open trade, closest zones...) and by
--                   the laptop supervisor (is the process running).
--   bot_commands    buttons pressed on the dashboard (update / restart /
--                   start / stop), picked up by the laptop supervisor.
--   closed_trades   every closed position from MT5's own history, for the
--                   monthly calendar. trading_day = the bot's trading day
--                   (resets 17:00 New York, 22:00 UK most of the year).
--   dashboard_auth  failed PIN attempts / 15 minute lock (used in the M5
--                   bot's project; harmless in the M1 one).
--   bot_config      new keys "risk" and "max_losses" (null = use the bot's
--                   own default / .env setting).
--
-- No anon policies: only the dashboard's server (service role key) and the
-- bots read or write these tables.

CREATE TABLE IF NOT EXISTS bot_status (
  bot TEXT PRIMARY KEY,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  data JSONB NOT NULL DEFAULT '{}'::jsonb,
  supervisor_seen_at TIMESTAMPTZ,
  process_running BOOLEAN
);
-- What the laptop supervisor last had to say ("already running in
-- another window", "stopped by itself, restarting in 20s").
ALTER TABLE bot_status ADD COLUMN IF NOT EXISTS supervisor_note TEXT;

CREATE TABLE IF NOT EXISTS bot_commands (
  id BIGSERIAL PRIMARY KEY,
  bot TEXT NOT NULL,
  command TEXT NOT NULL CHECK (command IN ('update', 'restart', 'start', 'stop')),
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'running', 'done', 'failed')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  output TEXT
);
CREATE INDEX IF NOT EXISTS idx_bot_commands_pending
  ON bot_commands (bot, created_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_bot_commands_created
  ON bot_commands (created_at DESC);

CREATE TABLE IF NOT EXISTS closed_trades (
  position_id BIGINT PRIMARY KEY,
  bot TEXT NOT NULL,
  setup_key TEXT NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
  volume NUMERIC(10,2) NOT NULL,
  open_time TIMESTAMPTZ NOT NULL,
  close_time TIMESTAMPTZ NOT NULL,
  open_price NUMERIC(12,2),
  close_price NUMERIC(12,2),
  profit NUMERIC(12,2) NOT NULL,
  comment TEXT,
  trading_day DATE NOT NULL,
  synced_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_closed_trades_day
  ON closed_trades (bot, trading_day);

CREATE TABLE IF NOT EXISTS dashboard_auth (
  id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  failed_attempts INT NOT NULL DEFAULT 0,
  locked_until TIMESTAMPTZ
);
INSERT INTO dashboard_auth (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

INSERT INTO bot_config (key, value, description) VALUES
  ('risk', 'null'::jsonb,
   'Dashboard risk per trade: "1%", "0.5%" or "100" (fixed). null = the bot''s own default / .env setting.'),
  ('max_losses', 'null'::jsonb,
   'Dashboard losing trades a day before the bot stops. null = the bot''s own default (2).')
ON CONFLICT (key) DO NOTHING;

ALTER TABLE bot_status     ENABLE ROW LEVEL SECURITY;
ALTER TABLE bot_commands   ENABLE ROW LEVEL SECURITY;
ALTER TABLE closed_trades  ENABLE ROW LEVEL SECURITY;
ALTER TABLE dashboard_auth ENABLE ROW LEVEL SECURITY;
