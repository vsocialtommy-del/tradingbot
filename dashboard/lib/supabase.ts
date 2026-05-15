import { createClient } from "@supabase/supabase-js";

// Read-only Supabase client used by the API route. The anon key is
// safe to ship to the browser, but we keep all queries server-side
// in the API route so we can shape the response (and add server-side
// caching headers later if needed).
const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

if (!url || !anonKey) {
  // Throwing at module load gives a clear "you forgot to set the env
  // vars" error in `vercel deploy` build logs.
  throw new Error(
    "Missing NEXT_PUBLIC_SUPABASE_URL or NEXT_PUBLIC_SUPABASE_ANON_KEY",
  );
}

export const supabase = createClient(url, anonKey, {
  auth: { persistSession: false },
});

// Row shape mirrors `signals` in migration 014. Numeric columns come
// back as strings from PostgREST (NUMERIC → text); the API route
// converts them.
export interface SignalRow {
  id: string;
  direction: "BUY" | "SELL";
  zone_id: string | null;
  zone_top: string;
  zone_bottom: string;
  entry_price: string;
  sl_price: string;
  tp1_price: string | null;
  tp2_price: string | null;
  tp3_price: string | null;
  pattern_type: string | null;
  zone_status: string | null;
  current_price: string | null;
  distance_dollars: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

// Numeric-coerced view used by the front-end. `null` means the bot
// didn't compute that level (e.g. no local peak found for TP3).
export interface Signal {
  direction: "BUY" | "SELL";
  zoneTop: number;
  zoneBottom: number;
  entryPrice: number;
  slPrice: number;
  tp1Price: number | null;
  tp2Price: number | null;
  tp3Price: number | null;
  patternType: string | null;
  zoneStatus: string | null;
  currentPrice: number | null;
  distanceDollars: number | null;
  updatedAt: string;
}

const num = (v: string | null): number | null =>
  v === null ? null : Number(v);

const numNotNull = (v: string): number => Number(v);

export function rowToSignal(row: SignalRow): Signal {
  return {
    direction: row.direction,
    zoneTop: numNotNull(row.zone_top),
    zoneBottom: numNotNull(row.zone_bottom),
    entryPrice: numNotNull(row.entry_price),
    slPrice: numNotNull(row.sl_price),
    tp1Price: num(row.tp1_price),
    tp2Price: num(row.tp2_price),
    tp3Price: num(row.tp3_price),
    patternType: row.pattern_type,
    zoneStatus: row.zone_status,
    currentPrice: num(row.current_price),
    distanceDollars: num(row.distance_dollars),
    updatedAt: row.updated_at,
  };
}
