# Dashboard (placeholder)

The Next.js monitoring dashboard lives here. Built in **Phase E** per
spec Section 9.3.

Pages planned:
- Live trades view (Supabase realtime)
- Performance summary (win rate, P&L, drawdown)
- Trade history table
- Zone history visualisation
- Bot status (running / halted / error)
- Kill switch (writes to `bot_config` → bot polls every 30 s)
- Config editor (parameter tuning without redeploy)

Plus a Vercel cron route that fetches Finnhub every 15 min and writes to
Supabase `news_events` (spec Section 8.4).

Nothing committed here yet — folder reserved.
