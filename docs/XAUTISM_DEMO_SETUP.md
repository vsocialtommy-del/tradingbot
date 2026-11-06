# XAUTISM bot: what it trades and how to run it on the demo

## The locked-in rules (Nov 2026)

| Part | Rule |
|---|---|
| Zones | M5 vertex zones: the M (sell) / W (buy) top, 3-8 candles, marked wick to wick. Leg in and leg out each move at least one zone height; the leg out closes beyond the zone by half its height. DBR / RBD only, minimum 3 points. |
| Fresh | Untouched (wicks count) since the BOS, never closed through. |
| BOS / TP1 | The foot: the low (sell) / high (buy) the whole move into the zone started from, on the body close. Pullbacks that bounce less than 25 % of the zone height are skipped (zigzag). The drop / rally out must close through it. TP1 = that level, 0.3 inside for spread. |
| Direction | M15 trend, or against it only when the M5 zone sits in a fresh H1 zone. |
| Entry | 0.01 at the zone edge on first touch. Layer 2 (the rest) is a broker stop at the same edge, placed once price is 0.10 inside the zone, filling on the way back out. Same SL and TP on both. Layer 2 is cancelled if TP or SL hits first. |
| SL | 30 % of the zone past the far edge (below for buys, above for sells), pushed beyond a swing just past it. Extra exit: an M5 candle closing beyond the far edge closes everything. |
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
