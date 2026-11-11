# XAUTISM bot: what it trades and how to run it on the demo

## The locked-in rules (Nov 2026)

| Part | Rule |
|---|---|
| Zones | M5 vertex zones: the M (sell) / W (buy) top, 3-8 candles, marked wick to wick. Leg in and leg out each move at least one zone height; the leg out closes beyond the zone by half its height. DBR / RBD only, minimum 3 points. |
| Fresh | Untouched (wicks count) since the BOS, never closed through. |
| BOS / TP1 | The foot: the low (sell) / high (buy) the whole move into the zone started from, on the body close. Pullbacks that bounce less than 25 % of the zone height are skipped (zigzag). The drop / rally out must close through it. TP1 = that level, 0.3 inside for spread. |
| Direction | M15 trend, or against it only when the M5 zone sits in a fresh H1 zone. |
| Entry | 0.01 at the zone edge on first touch. Layer 2 (the rest) is a broker stop at the same edge, placed once price is 0.10 inside the zone, filling on the way back out. Same SL on both (TP: see the 75/25 runner row). Layer 2 is cancelled if TP or SL hits first. |
| TP1 exit (75/25 runner) | At TP1, 75 % of the whole position closes (layer 1 by the bot, the layer 2 main part at its broker TP) and 25 % (the runner part of layer 2, a second stop at the same price with no TP) keeps going. Its SL moves to breakeven, then trails 4.00 ($4 = 400 MT5 points) behind the best price, never back. If layer 2 never filled, the 0.01 becomes the runner instead of closing for a tiny win. Nothing changes before TP1. 12-month backtest: +£7,446 vs +£4,976 all-out at TP1, max drawdown -£568 vs -£651, no losing month. Off switch: `runner_enabled`. |
| SL | 30 % of the zone past the far edge (below for buys, above for sells), pushed beyond a swing just past it. The candle close exit is **off** on both bots (10 Nov 2026): only the 30% SL, TP1 and the runner trail close a trade. Code kept (switch: `m5_close_exit`). |
| Risk | £100 per trade (both layers together), -£300 a day, stop after 2 losing trades, one trade at a time, no entry above a 0.50 spread. |
| News | USD red folder events (Forex Factory calendar, downloaded by the bot every hour): no new trades from 15 min before to 15 min after; any open trade is closed and a pending layer 2 cancelled 15 min before. Zones keep being found. Not in the backtest. |
| Never twice | A zone is traded once. Nothing else blocks a fresh zone (same as the backtest). |
| Off (code kept) | Stacked two-zone layering, flip trading, bot-managed TPs / break even / trailing / zone exit, the classic detector, the old overlap dedup, the old Supabase/Vercel news table, the old % daily halt. |

Backtest, 8 Jul - 21 Oct 2026 (M5, M1 from Sep, £100 risk): 229 trades,
62.4 % win rate, +£1,788, profit factor 1.39, max drawdown -£365.

## Starting it on the Vantage demo (Windows PC)

1. In `C:\Users\tomeo\Documents\tradingbot`: `git pull` then
   `pip install -r requirements.txt`.
2. **Supabase**: SQL editor → paste `migrations/015_xautism_layer2_stop.sql` → Run.
   Without it layer 2 can't be saved.
3. **MT5**: log in to the new Vantage demo, turn Algo Trading on, make
   sure `XAUUSD` is in Market Watch.
4. **`.env`**: set `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER` (and
   `MT5_PATH` if MT5 isn't in the default place) to the demo's details.
5. Run `python -m bot.main`. The log shows zones being found and checked
   on every M5 close.

On startup the bot closes off any open setup in Supabase with nothing
live at the broker (e.g. left over from an old account), so it can't
block the one-trade limit.

First days, watch for: lots of `spread` skips (raise `risk.max_spread`).

## Re-running the backtest

```
python scripts/backtest_xautism.py M5_part1.csv,M5_part2.csv M1.csv --start 2026-07-08 --out trades.csv
```

## M1 bot (second bot, same code)

Two bots share this code. Each runs in its own window, on its own MT5
account and its own Supabase project, so they never see each other's trades.

| | M5 bot | M1 bot |
|---|---|---|
| Start | `python -m bot.main` (or `--bot m5`), or double-click `start_m5_bot.bat` | `python -m bot.main --bot m1`, or double-click `start_m1_bot.bat` |
| Settings file | `.env` | `.env.m1` (copy `.env.m1.example`) |
| Zones | M5, 3 pt minimum | M1, 1 pt minimum |
| Direction | M15 trend, fresh H1 zone exception | M5 trend, fresh M15 zone exception |
| Close exit | **None.** Only the 30% SL, TP1 and the runner trail close a trade | **None.** Only the 30% SL, TP1 and the runner trail close a trade |
| Risk | £100 a trade, 2 losses a day, -£300 cap | Switchable: `RISK=10%` (default), `RISK=5%`, `RISK=1%` or `RISK=100` (£100 a trade). Losses a day: `MAX_LOSSES=2` (default), raise it for testing. The daily cap is one trade more than the losses allowed (2 losses at 10% = 30%, 2 at £100 = £300). Set both in `.env.m1`, or override for one run with `python -m bot.main --bot m1 --risk 5% --max-losses 10`. The startup log shows what's on. |
| After TP1 | 75% closes, 25% trails 4 pts | 50% closes, 50% trails 2 pts |
| Supabase | M5 bot project | M1 bot project (`nvkwnpesedzwnxewuczn`) |

Everything else (30% SL, TP at the BOS foot, 0.01 + layer 2 stop, one
trade at a time, news filter, spread filter) is the same.

**First-time M1 setup**

1. Open a new MetaQuotes demo account for the M1 bot (start it at £1,000
   to match the backtest).
2. Install MetaTrader 5 a second time into its own folder (for example
   `C:\Program Files\MetaTrader 5 M1`) and log that copy into the M1 demo.
   Tick Algo Trading in it.
3. Supabase, "M1 bot" project, SQL Editor: run `m1_bot_supabase_setup.sql`
   (all migrations 001-015 in one file).
4. Copy `.env.m1.example` to `.env.m1` and fill in the M1 account login,
   password, server, the second MT5's `terminal64.exe` path and the M1
   project's service_role key.
5. Start it in a new Command Prompt: `python -m bot.main --bot m1`. The
   first log lines say `=== M1 bot ===` and list the settings.

## Running both bots with the dashboard supervisor (Nov 2026)

Double-click **`start_bots.bat`**. It does a `git pull`, then opens the
supervisor with **both bots stopped**. Press **Start** on each bot in the
phone dashboard to run it. Keep the supervisor window open: it's what
carries out the dashboard's buttons (and runs the bots).

- The bots have no windows of their own. To watch one, double-click
  `watch_m5.bat` or `watch_m1.bat` (shows `logs/m5.log` / `logs/m1.log`
  live). Anything a bot prints when it crashes is in
  `logs/m5.console.log` / `logs/m1.console.log`.
- **Close the supervisor window (or Ctrl+C in it) and both bots stop.**
  Open trades and pending orders stay in MT5 as they are, same as Ctrl+C
  in a bot window.
- A bot that crashes is started again after 10 s, then 20 s, 40 s... up
  to 5 minutes between tries.
- Dashboard buttons: **Update** = git pull (and pip install if
  `requirements.txt` changed) then restart both bots. **Restart** /
  **Stop** / **Start** = just that bot. A bot stopped from the dashboard
  stays stopped until you press Start (or double-click `start_bots.bat`
  again). Buttons pressed while the laptop was off are ignored after 10
  minutes.
- Each bot only runs once: if it's already running in an old window, the
  new copy refuses to start and the supervisor waits until you close the
  old window, then takes over within a minute. So close the old
  `start_m5_bot.bat` / `start_m1_bot.bat` windows first. Those two still
  work on their own if you ever want a bot in its own window.
- MT5 still needs to be open and logged in for each bot, as before.
