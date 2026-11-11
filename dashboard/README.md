# XAU/TISM BOT dashboard

Phone dashboard for both bots (Next.js on Vercel). Black and gold, PIN
login, add it to your home screen like an app.

- Both bots side by side: on/off switch, status, balance, today's P/L,
  W/L today, closest buy and sell zone. Tap a bot for everything else:
  open trade, zone details, risk and max-losses settings, Update /
  Restart / Stop / Start, last commands.
- A monthly calendar per bot (trading day = 22:00 to 22:00 UK). Tap to
  enlarge, tap a day for its trades.

## How it fits together

```
phone  ->  this site (Vercel)  ->  each bot's Supabase  <-  bot + laptop supervisor
```

- The bots write `bot_status` (every 10 s) and `closed_trades` (every
  60 s) to their own Supabase project (SQL 016).
- The on/off switch and settings write `bot_config` (`kill_switch`,
  `risk`, `max_losses`); the bot reads them every 30 s.
- Buttons write `bot_commands`; the laptop supervisor (`start_bots.bat`)
  carries them out within a few seconds.
- The service_role keys stay on the server. Every page and API route
  needs the PIN cookie (30 days). 5 wrong PINs lock it for 15 minutes,
  counted in the M5 project's `dashboard_auth` table.

## Vercel settings

Root Directory: `dashboard`. Environment variables: see `.env.example`
(`DASHBOARD_PIN`, `SESSION_SECRET`, `M5_SUPABASE_URL`,
`M5_SUPABASE_SERVICE_ROLE_KEY`, `M1_SUPABASE_URL`,
`M1_SUPABASE_SERVICE_ROLE_KEY`). The old `NEXT_PUBLIC_SUPABASE_*`
variables aren't used any more and can be deleted.

## Local

```sh
cd dashboard
npm install
cp .env.example .env.local   # fill it in
npm run dev
```
