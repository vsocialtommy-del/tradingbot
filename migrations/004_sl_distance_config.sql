-- =============================================================================
-- 004_sl_distance_config.sql
--
-- Adds the ``min_sl_distance_points`` and ``max_sl_distance_points`` config
-- knobs read by ``bot/exits/sl_manager.py`` (spec Section 5.1). These are
-- sanity-check bounds applied to ``calculate_initial_sl`` results so that:
--
--   * SL too close to entry (< min_sl_distance_points) — slippage / spread
--     would routinely stop us out before the trade breathes. Default 5
--     points (XAUUSD spread + a small margin).
--   * SL too far from entry (> max_sl_distance_points) — risk per trade
--     becomes uncomfortable even at fixed 0.01 lots, and signals that
--     swing detection picked up a level that's not actually structural.
--     Default 200 points (~$2 on XAUUSD; well above any reasonable
--     structural SL but well below "something is wrong" widths).
--
-- The validator emits ``is_too_close`` / ``is_too_far`` flags so the
-- caller can decide whether to skip the setup, clip the SL, or proceed
-- with a warning. v1 default behaviour: skip-with-log.
--
-- HOW TO RUN:
--   Open Supabase Dashboard → SQL Editor → paste this file → Run.
--
-- IDEMPOTENT: ON CONFLICT DO NOTHING preserves operator-edited values.
-- =============================================================================

INSERT INTO bot_config (key, value, description) VALUES
  ('min_sl_distance_points', '5'::jsonb,
   'Minimum SL distance from entry price (price units). Below this, slippage/spread routinely stops the trade out before it breathes (spec Section 5.1).'),
  ('max_sl_distance_points', '200'::jsonb,
   'Maximum SL distance from entry price (price units). Above this, the structural reference is likely spurious; backtest will tune the real bound (spec Section 5.1).')
ON CONFLICT (key) DO NOTHING;
