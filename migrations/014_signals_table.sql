-- =============================================================================
-- 014_signals_table.sql
--
-- PR #51: signals dashboard.
--
-- New ``signals`` table holds a small live snapshot of "the next zone the
-- bot is waiting to retest" — one row per direction (BUY / SELL). The
-- Python bot writes this snapshot on every M5 close; a Vercel-hosted
-- Next.js dashboard reads it (via the Supabase anon role + RLS policy
-- defined below) and renders it for the operator's phone / browser.
--
-- The table is small by design: at most 2 active rows (one BUY signal,
-- one SELL signal). The writer marks the previous same-direction row
-- ``is_active = FALSE`` before inserting the new one, so historical
-- rows accumulate but the "current state" query stays trivially fast
-- via ``WHERE is_active = TRUE``.
--
-- IDEMPOTENT — uses IF NOT EXISTS / CREATE OR REPLACE / DROP IF EXISTS
-- so re-applying is safe.
--
-- RLS:
--   The base policy of the schema is RLS-enabled-no-policies (only the
--   service_role can read/write — see 001_initial_schema.sql comment).
--   The signals table needs to be readable by the anon role so the
--   dashboard can fetch it client-side / via its server-side API
--   route using the anon key. We grant SELECT only — writes still
--   require service_role. Signals contain only public market analysis
--   (zone bounds, target prices); there's no PII or account data here.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- Table
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS signals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  direction TEXT NOT NULL CHECK (direction IN ('BUY', 'SELL')),
  -- FK to zones is nullable: if the underlying zone row is later
  -- deleted (manual cleanup), we don't want to cascade-delete the
  -- historical signal record. ON DELETE SET NULL preserves the
  -- snapshot's other columns for the audit trail.
  zone_id UUID REFERENCES zones(id) ON DELETE SET NULL,
  zone_top NUMERIC(10,2) NOT NULL,
  zone_bottom NUMERIC(10,2) NOT NULL,
  -- Layer 1 entry price (= zone.top for BUY, zone.bottom for SELL).
  -- Snapshotted here for convenience so the dashboard doesn't need
  -- the direction-conditional rendering.
  entry_price NUMERIC(10,2) NOT NULL,
  sl_price NUMERIC(10,2) NOT NULL,
  tp1_price NUMERIC(10,2),
  tp2_price NUMERIC(10,2),
  tp3_price NUMERIC(10,2),
  pattern_type TEXT,
  -- 'CONFIRMED' for a brand-new zone, 'FLIPPED' for a previously-
  -- violated zone now tradeable in the flipped direction (PR #38).
  zone_status TEXT,
  -- Bid (BUY) or ask (SELL) at the moment the signal was computed.
  -- The dashboard uses it for the "current price" footer and the
  -- "$X away" display, so we don't bake formatting into the bot.
  current_price NUMERIC(10,2),
  -- Signed distance from current_price to entry_price. Positive
  -- means price has further to travel to trigger the entry (the
  -- expected case for a pending retest). Negative means the entry
  -- has already been reached / passed.
  distance_dollars NUMERIC(10,2),
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT signals_top_above_bottom CHECK (zone_top > zone_bottom)
);

-- Reuse the shared trigger from 001_initial_schema.sql.
DROP TRIGGER IF EXISTS trg_signals_updated_at ON signals;
CREATE TRIGGER trg_signals_updated_at
  BEFORE UPDATE ON signals
  FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- -----------------------------------------------------------------------------
-- Indexes
-- -----------------------------------------------------------------------------
-- "Give me the live signals" — the dashboard's primary query path.
CREATE INDEX IF NOT EXISTS idx_signals_active_direction
  ON signals (is_active, direction)
  WHERE is_active = TRUE;

-- Time-ordering for the historical view (not used by v1 dashboard,
-- but cheap and useful for ad-hoc inspection).
CREATE INDEX IF NOT EXISTS idx_signals_updated_at
  ON signals (updated_at DESC);


-- -----------------------------------------------------------------------------
-- Row Level Security
--
-- The base schema (001) sets RLS-enabled-no-policies on every table.
-- For signals we need public read so the dashboard's anon key works.
-- Writes stay locked: only the service_role (used by the bot) can
-- INSERT / UPDATE / DELETE.
-- -----------------------------------------------------------------------------
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;

-- DROP first so re-running this migration after policy edits is clean.
DROP POLICY IF EXISTS signals_public_read ON signals;
CREATE POLICY signals_public_read
  ON signals
  FOR SELECT
  TO anon, authenticated
  USING (TRUE);
