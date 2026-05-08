-- =============================================================================
-- 003_tp1_method_config.sql
--
-- Strategy refinement (May 2026): TP1 is now computed as the BoS level — the
-- swing high/low from BEFORE the zone formed that the impulse broke through —
-- instead of a fixed $4 beyond the zone edge. The BoS level is what
-- traders watch for retracement, and per Tommy's screenshot analysis it's
-- the natural exit point. The fixed-distance method is preserved as a
-- backtest-revert escape hatch.
--
-- This migration adds the ``tp1_method`` config knob:
--
--     'BOS_LEVEL'      (default v1) — TP1 = setup.zone.bos_event.broken_level
--     'FIXED_DISTANCE' (legacy)     — TP1 = zone_edge ± tp1_distance_dollars
--
-- The existing ``tp1_distance_dollars`` row is intentionally retained so the
-- FIXED_DISTANCE path stays usable for backtesting and rollback.
--
-- HOW TO RUN:
--   Open Supabase Dashboard → SQL Editor → paste this file → Run.
--
-- IDEMPOTENT: ON CONFLICT DO NOTHING preserves any operator-edited value.
-- =============================================================================

INSERT INTO bot_config (key, value, description) VALUES
  ('tp1_method', '"BOS_LEVEL"'::jsonb,
   'TP1 calculation method. "BOS_LEVEL" (v1 default) uses the broken structural level the impulse cleared; "FIXED_DISTANCE" (legacy) uses zone_edge ± tp1_distance_dollars (spec Section 6.1).')
ON CONFLICT (key) DO NOTHING;
