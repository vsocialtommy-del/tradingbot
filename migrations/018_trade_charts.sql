-- 018: trade photo journal (26 Nov 2026)
--
-- Run once in EACH bot's Supabase project (M5 bot and M1 bot). Safe to
-- run again.
--
--   trade_charts   one row per closed trade that has a picture: where the
--                  picture is in storage, plus the trade's numbers (zone,
--                  M1/M5, buy/sell, H1 trend, entry/SL/TP, every layer,
--                  result, how far price went for / against) as JSON for
--                  analysis later.
--   trade-charts   private storage bucket holding the PNG pictures. Only
--                  the bot and the dashboard's server (service role key)
--                  can read or write it; the dashboard shows each picture
--                  through a link that expires after an hour.

CREATE TABLE IF NOT EXISTS trade_charts (
  bot TEXT NOT NULL,
  account BIGINT NOT NULL DEFAULT 0,
  setup_key TEXT NOT NULL,
  path TEXT NOT NULL,
  opened_at TIMESTAMPTZ,
  closed_at TIMESTAMPTZ,
  data JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (bot, account, setup_key)
);
CREATE INDEX IF NOT EXISTS idx_trade_charts_opened ON trade_charts (bot, account, opened_at DESC);
ALTER TABLE trade_charts ENABLE ROW LEVEL SECURITY;   -- no policies: service role only

INSERT INTO storage.buckets (id, name, public)
VALUES ('trade-charts', 'trade-charts', false)
ON CONFLICT (id) DO NOTHING;
