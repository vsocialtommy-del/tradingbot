# Trading Signals Dashboard

A small Next.js (App Router, TypeScript, Tailwind) page that renders
the bot's next-pending BUY and SELL signals. Polls `/api/signals`
every 10 seconds.

The bot side (PR #51) writes to the Supabase `signals` table on every
M5 close. This dashboard reads it. See:

- `migrations/014_signals_table.sql` — table + RLS policy
- `bot/signals/next_signal_writer.py` — writer module
- `bot/main.py` `_run_next_signal_pass` — caller

## Local development

```sh
cd dashboard
npm install
cp .env.example .env.local
# Fill in NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY
# from your Supabase project (Settings → API).
npm run dev
```

The dev server runs on http://localhost:3000.

## Deployment (Vercel)

1. In Vercel, click **Add New Project** and import this repo.
2. Set the **Root Directory** to `dashboard`.
3. **Environment Variables** — add both for Production (and Preview /
   Development if you want previews to read the same DB):
   - `NEXT_PUBLIC_SUPABASE_URL` — your Supabase project URL.
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY` — your Supabase **anon** (public)
     key.
4. Click **Deploy**.

Vercel auto-assigns a domain (e.g.
`tradingbot-signals.vercel.app`). Custom domains can be added in
Project Settings → Domains.

## Why the anon key?

Migration 014 enables Row Level Security on `signals` and adds a
SELECT-only policy for the `anon` role. That means:

- The anon key can READ signal rows. Safe to ship in `NEXT_PUBLIC_*`.
- INSERT / UPDATE / DELETE still require the bot's `service_role`
  key, which is never exposed to the dashboard.

## Architecture

```
┌──────────┐  M5 close   ┌────────────┐  GET /api/signals   ┌──────────┐
│   bot    │ ──────────► │  signals   │ ◄────── poll 10s ── │ browser  │
│ (Python) │   upsert    │  (Supabase)│  anon SELECT only   │  (Next)  │
└──────────┘             └────────────┘                     └──────────┘
```

- **Page** (`app/page.tsx`): client component, polls `/api/signals` on
  a 10 s interval, renders two cards (BUY + SELL) plus a footer.
- **API route** (`app/api/signals/route.ts`): server-side, queries
  Supabase for `is_active = true` rows, normalises numeric strings to
  numbers, returns JSON. Cache disabled (`force-dynamic`) so the
  Vercel CDN never serves stale data.
- **Lib** (`lib/supabase.ts`): typed client + row→signal mapper.

## What you'll see

- **🟢 BUY** card — closest BUY zone the bot is waiting to retest:
  zone range, pattern (RBR/DBR/etc.), status (CONFIRMED / FLIPPED),
  Entry / SL / TP1 / TP2 / TP3, distance from current price.
- **🔴 SELL** card — same shape, mirrored.
- **Footer** — current XAUUSD price + "X seconds ago" freshness label.
- **No active signal** — placeholder when the bot has no qualifying
  zone in that direction (e.g. price is currently inside both the
  nearest BUY and the nearest SELL zone, or no zones are within
  $50 of price).
