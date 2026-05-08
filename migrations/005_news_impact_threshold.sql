-- =============================================================================
-- 005_news_impact_threshold.sql
--
-- Adds the ``news_impact_threshold`` config knob read by
-- ``bot/filters/news_filter.py``. The bot blocks new entries when an event
-- of this severity (or higher) is within the existing
-- ``news_blackout_minutes_before`` / ``_after`` window.
--
-- Severity ordering (Finnhub convention, mirrored in the schema's
-- impact_level CHECK constraint): ``HIGH`` > ``MEDIUM`` > ``LOW``.
--
-- Default ``HIGH`` matches spec Section 8.3 — only top-tier events
-- (NFP, FOMC, CPI, retail sales, GDP, Fed/Powell speeches) trigger the
-- blackout. Operators tightening the bot during volatile macro periods
-- can flip this to ``MEDIUM`` from the dashboard without a code deploy.
--
-- HOW TO RUN:
--   Open Supabase Dashboard → SQL Editor → paste this file → Run.
--
-- IDEMPOTENT: ON CONFLICT DO NOTHING preserves operator-edited values.
-- =============================================================================

INSERT INTO bot_config (key, value, description) VALUES
  ('news_impact_threshold', '"HIGH"'::jsonb,
   'Minimum impact_level that triggers the news blackout. "HIGH" (default) blocks only top-tier events; "MEDIUM" tightens during volatile macro periods (spec Section 8.3).')
ON CONFLICT (key) DO NOTHING;
