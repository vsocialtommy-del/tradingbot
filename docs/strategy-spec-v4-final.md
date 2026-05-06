# XAUUSD Supply & Demand Trading Bot — Strategy Specification v4.0 (FINAL)

**Author:** Tommy
**Date:** 6 May 2026
**Status:** Final draft — ready for build phase
**Supersedes:** v3.0

---

## 1. Executive Summary

A semi-automated trading bot that trades XAUUSD (Gold) using a Supply & Demand strategy taught by Tommy's private Telegram group. The bot detects two specific setup types — Strong Point and Imbalance Zone — places three layered market orders when price re-enters the zone, manages exits through a TP1 + Break-Even system, and hands off the runner trade to Tommy for manual management.

Risk: 0.33% of balance per layer, 1% per setup, max 2-3 simultaneous setups, daily halt at -10%.

Broker: Vantage Markets (MT5).

Goal: Build a live, profitable bot for Tommy's own account first. Once 60+ days of profitable live trading is established, enable as a vSocial signal provider.

---

## 2. Strategy Overview

### 2.1 Core methodology

The strategy is rooted in institutional Supply & Demand concepts (Strong Point, Imbalance Zone, BoS, CHoCH) as taught by Tommy's Telegram group. It uses line-chart pattern recognition (W/M/N) to identify zones, candle-body refinement to define exact zone boundaries, and layered market orders to scale into the zone.

### 2.2 Asset and timeframes

| Parameter | Value |
|---|---|
| Asset (v1) | XAUUSD only |
| Bias / zone timeframe | 1H |
| Entry timeframe | M5 / M1 |
| Sessions | 24/5 (will narrow via backtest) |
| News filter | Avoid high-impact USD news (±30 min) |

### 2.3 Setup types traded in v1

| Setup | Trade in v1? | Notes |
|---|---|---|
| Strong Point | ✅ Yes | Foundation setup — every other type builds on it |
| Imbalance Zone | ✅ Yes | Purely mechanical, easy to codify |
| CHoCH | ❌ Defer to v2 | Requires trend detection logic |
| SnD Flip | ❌ Defer to v2 | Requires multi-zone tracking |
| High Risk | ❌ Never | Explicitly judgement-based |
| Standalone N pattern | ❌ Defer to v2 | Often overlaps with CHoCH |

### 2.4 Trade frequency target

5–20 trades per day across all open setups.

---

## 3. Setup Definitions

### 3.1 Strong Point (foundation setup)

**Definition:**
A zone formed by a high-quality reversal pattern with clear structure validation.

**Identification rules (BUY example):**

- Pattern detection (1H line chart): Detect W pattern — two swing lows within 0.1% tolerance, with a higher peak between them
- Quality validation:
  - The move out of the zone must have broken a previous structure level (BoS up)
  - The base before the impulsive move = small consolidation candles
  - The impulsive move = strong-bodied candles, not just wicks
- Zone marking: Box around the W pattern's reversal area
- Zone refinement (candles): Adjust to candle bodies only, exclude wicks
- Final zone: has defined `top` and `bottom` price levels

Educational reference: "Strong Point for Sell" / "Strong Point for Buy" images

Key principle from education: "No need to check the overall trend, focus only on the zone."

### 3.2 Imbalance Zone (mechanical setup)

**Definition:**
A Strong Point zone that has been approached multiple times but never tapped — building "pent-up energy" for a powerful reaction on first touch.

**Identification rules (BUY example):**

- Identify a Strong Point demand zone using rules in 3.1
- Track zone approaches: count instances where price came within 5-10 points of zone top but did not enter
- Imbalance qualification: zone has had 2+ failed approaches without tapping
- Entry trigger: first actual touch of the zone after being qualified as Imbalance

Higher probability characteristic: Imbalance Zones often produce direct push to TP2 with minimal pullback.

Educational reference: "Imbalance Zone for Buy" / "Imbalance Zone for Sell" images

---

## 4. Entry Execution

### 4.1 Layered market orders

When price first touches the refined zone, place three equal-sized orders:

| Position | Trigger | Order type | Detection |
|---|---|---|---|
| Layer 1 | Tick crosses zone top (BUY) / bottom (SELL) | Market order, fires instantly | Real-time tick |
| Layer 2 | Zone midpoint = (top + bottom) / 2 | Pending limit order | M5 wick touches |
| Layer 3 | Zone bottom (BUY) / top (SELL) | Pending limit order | M5 wick touches |

Entry mode: First touch (no body close confirmation required for v1)

### 4.2 Entry detection precision

- **Layer 1:** Real-time tick monitoring. The instant a tick prints at or below zone top (for BUY) or at or above zone bottom (for SELL), Layer 1 fires as a market order.
- **Layers 2 & 3:** Set as pending limit orders at zone midpoint and zone bottom (BUY) / zone top (SELL). Filled when M5 candle wick touches the limit price.

### 4.3 Gap protection rule

If the entry tick gaps through the entire zone in a single price update (e.g., price goes from above zone top to below zone bottom instantly):

- Skip the setup entirely
- Do NOT fire Layer 1
- Do NOT set pending Layers 2/3
- Log: "Setup skipped — gap through zone detected"

This protects against entries during news spikes or low-liquidity gaps.

### 4.4 Lot sizing (v1: Fixed mode)

- All layers use fixed 0.01 lot size regardless of balance
- This is for bot validation purposes — focus is on proving entry/exit logic works correctly, NOT realistic return generation
- Risk per setup at 0.01 lots × 3 layers ≈ $0.60-$1.20 per losing setup (depending on SL distance)
- Risk-based sizing (0.33% per layer) deferred to v1.1 once bot is validated

**Important implications:**

- Daily halt at -10% of balance is functionally inactive at low balances + small lots — that's intentional during validation
- Backtest results will show small absolute returns; what matters is win rate, profit factor, and drawdown %, not $ amounts
- vSocial signal provision is NOT viable in v1 (copiers' proportional fills would be sub-minimum) — defer to v1.1

### 4.5 Partial fill handling

If price reverses before all three layers fill:

- Cancel remaining pending Layer 2/3 orders
- Manage filled positions normally (do not chase missed layers)
- TP1 calculation uses average entry of filled positions only

---

## 5. Stop Loss

### 5.1 Initial SL placement (real SL, set at trade open)

- For BUY: Below the recent lower low + 15-20 point buffer (anti stop-hunt)
- For SELL: Above the recent higher high + 15-20 point buffer
- "Recent" definition: Lowest swing low / highest swing high within the last 20 candles on 1H (configurable parameter).

**Behaviour:** All three positions share the same initial SL. If hit before TP1, all three positions close together. Total loss = exactly 1% of account balance.

### 5.2 SL lifecycle

```
Stage 1: Trade opens         → SL at "real" level (below structure + buffer)
Stage 2: TP1 hits            → SL moves to break-even on remaining 50%
Stage 3: Runner phase        → Tommy manually manages or SL stays at BE until hit
```

❌ The bot does NOT trail SL automatically. Tommy manages the runner.

---

## 6. Take Profit

### 6.1 TP1 (automated)

**Trigger:** Price moves $4-5 beyond the zone edge in trade direction
- For BUY: zone top + $4-5
- For SELL: zone bottom - $4-5

Default value: $4 (configurable parameter, will be tuned in backtest)

**Action at TP1:**
- Close 50% of total position (half of each filled layer)
- Move SL on remaining 50% to break-even (= average entry price of filled layers)

### 6.2 TP2 / Runner (manual)

After TP1 + BE move:
- Bot does not set TP2
- Bot does not trail SL automatically
- Runner trade holds until:
  - Tommy manually closes (via dashboard or phone)
  - BE SL hits (price comes back to average entry)

**Tommy's manual triggers (informational only — not bot logic):**
- New BoS in favour → consider tightening SL
- Approach to next opposing zone → consider taking profit
- News imminent → close runner

---

## 7. Risk Management

| Rule | Value |
|---|---|
| Lot sizing (v1) | Fixed 0.01 per layer, all balances |
| Risk per layer (v1.1+) | 0.33% of account balance |
| Total risk per setup (v1.1+) | 1% of account balance |
| Max simultaneous setups | 2-3 |
| Daily loss limit | -10% of starting daily balance → halt for the day |
| Daily reset time | 17:00 EST (broker rollover) |
| Weekend behaviour | Close losing trades Friday 16:00 EST, hold winners |
| Zone size filter (default v1) | Skip if zone < 5 points OR > 80 points |

**Daily halt logic:**
- At day start (17:01 EST), record `starting_balance`
- After each closed trade, check if `current_balance < starting_balance * 0.90`
- If true: cancel all pending orders, do NOT open new setups, allow open positions to run their course
- Resume next trading day

---

## 8. News Filter

### 8.1 Data source

- **Primary:** Finnhub Economic Calendar API (free tier — 60 calls/min, well within our needs)
- **Backup:** Manual override via dashboard config if Finnhub is down
- Sign up: finnhub.io → free API key

### 8.2 Architecture (uses traditional stack)

```
Vercel cron (every 15 min)
    ↓
Fetches from Finnhub API
    ↓
Filters for high-impact USD events
    ↓
Writes to Supabase news_events table
    ↓
Bot polls Supabase every 1 min
    ↓
Bot acts on imminent news
```

**Why this design:** The bot itself never calls Finnhub directly. Vercel handles fetching, Supabase stores the data, bot just reads. This means:
- No API keys on the VPS
- News data visible in dashboard (Supabase realtime)
- Easy to swap providers (only change Vercel route)
- Bot keeps working even if Finnhub is briefly down (last fetched data still in Supabase)

### 8.3 Filter logic

**High-impact events to track:** NFP, FOMC, CPI, retail sales, GDP, Fed/Powell speeches, ECB rate decisions, plus any USD event marked "high impact" by Finnhub.

**Bot behaviour:**
- 30 minutes before event: close all open Gold positions
- 30 minutes before event: do not open new setups
- 15 minutes after event: resume normal operation

### 8.4 Required setup

**Vercel:**
- Cron route: `app/api/cron/news/route.ts` — fetches Finnhub, writes to Supabase
- Cron schedule in `vercel.json`: `*/15 * * * *` (every 15 mins)
- Environment variable: `FINNHUB_API_KEY`

**Supabase:**
- `news_events` table with columns: `event_time`, `currency`, `title`, `impact_level`, `forecast`, `actual`, `fetched_at`
- Index on `event_time` for fast queries
- Unique constraint on `(event_time, title)` to prevent duplicates

**Bot:**
- `news_filter.py` polls Supabase every 1 minute
- `is_news_imminent()` returns True if any high-impact USD event is within 30 mins

### 8.5 Manual override

The dashboard (Section 9.3) includes a manual news override:
- "Pause bot for X minutes" button (for unscheduled events like geopolitical surprises)
- Writes to `bot_config` table → bot picks up within 30 sec

---

## 9. Tech Stack

**Design principle:** Use Tommy's existing traditional stack (GitHub + Supabase + Vercel + Next.js) for everything possible. Only introduce one new piece — a VPS — for the bot's 24/5 runtime, since serverless platforms cannot host long-running processes.

### 9.1 Bot runtime (the only new piece)

| Component | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for trading |
| Broker API | `MetaTrader5` Python library | Connects to MT5 desktop |
| Backtest framework | `vectorbt` or `backtrader` | Decide during build |
| Historical data | Dukascopy tick data (free) | One-time download |
| Hosting | Hetzner VPS, London region (~$5/mo) | Required for 24/5 runtime |
| MT5 desktop | Runs on the VPS | Connects bot to Vantage broker |

**Why VPS:** Vercel/Cloudflare Workers cannot host trading bots. They are serverless (max 30-60 sec runtime) and stateless (cold starts). A bot needs a persistent, stateful, 24/5 process. VPS is industry standard for this and is the only non-traditional piece in the stack.

### 9.2 Database (traditional stack ✅)

| Component | Choice |
|---|---|
| Database | Supabase (Tommy's existing stack) |
| Tables | `trades`, `setups`, `zones`, `daily_pnl`, `news_events`, `bot_logs`, `bot_config` |
| Real-time | Supabase realtime subscriptions for dashboard |

**Note:** A new `bot_config` table allows the dashboard (Vercel) to send commands to the bot (VPS) without direct connection — bot polls Supabase for config changes.

### 9.3 Dashboard & control plane (traditional stack ✅)

| Component | Choice |
|---|---|
| Frontend framework | Next.js (Tommy's existing stack) |
| Hosting | Vercel (Tommy's existing stack) |
| Real-time updates | Supabase realtime subscriptions |
| API routes | Vercel API endpoints for control commands |
| Mobile alerts | Telegram bot (triggered from Vercel API or VPS bot directly) |

**Pages:**
- Live trades view (real-time from Supabase)
- Performance summary (win rate, P&L, drawdown)
- Trade history table
- Zone history (visualize where bot found setups)
- Bot status (running/halted/error)
- Kill switch (writes to `bot_config` table → bot polls and pauses)
- Config editor (adjust parameters without redeploying bot)

### 9.4 Code repository & deployment (traditional stack ✅)

| Component | Choice |
|---|---|
| Source control | GitHub (Tommy's existing stack) |
| Bot deployment | GitHub Actions → SSH to VPS → pull + restart |
| Dashboard deployment | Vercel auto-deploys from GitHub (existing pattern) |

### 9.5 News data

| Component | Choice |
|---|---|
| Calendar API | Finnhub free tier (finnhub.io) |
| Update frequency | Every 15 minutes |
| Where it runs | Vercel cron job → writes to Supabase `news_events` → bot reads from Supabase |

**Why Vercel handles news:** Fetching news doesn't need 24/5 uptime — a Vercel cron job every 15 mins is perfect. Bot just reads the latest from Supabase. See Section 8 for full integration details.

---

### 9.6 Architecture diagram

```
┌─────────────────────────────────────────────────────────────┐
│  TRADITIONAL STACK (Tommy's existing skills/tools)          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GitHub ─────────────► Vercel (Next.js dashboard)           │
│   │                         │                                │
│   │                         ├── Live trades view             │
│   │                         ├── Performance metrics          │
│   │                         ├── Kill switch ──┐              │
│   │                         └── Config editor ┤              │
│   │                                            │              │
│   │                                            ▼              │
│   │                                    ┌──────────────┐      │
│   │                                    │   Supabase   │      │
│   │                                    │              │      │
│   │                                    │ - trades     │      │
│   │                                    │ - setups     │      │
│   │                                    │ - bot_config │      │
│   │                                    │ - bot_logs   │      │
│   │                                    │ - news       │      │
│   │                                    └──────┬───────┘      │
│   │                                           ▲              │
│   │                                           │              │
│   │  Vercel cron ────────► news scraper ─────┘              │
│   │  (every 15 min)                                          │
└───┼──────────────────────────────────────────┼──────────────┘
    │                                          │
    │ deploys to                               │ reads/writes
    ▼                                          │
┌─────────────────────────────────────────────────────────────┐
│  VPS (Hetzner, ~$5/mo) — the ONLY non-traditional piece     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Python bot (24/5)                                         │
│    ├── Reads config from Supabase                          │
│    ├── Reads news events from Supabase                     │
│    ├── Detects setups (W/M/N, Strong Point, Imbalance)     │
│    ├── Writes all trades/logs to Supabase                  │
│    │                                                         │
│    └── Communicates with ──┐                                │
│                            ▼                                 │
│                    MT5 desktop app (on VPS)                 │
│                            │                                 │
└────────────────────────────┼────────────────────────────────┘
                             │ MT5 native protocol
                             ▼
                    ┌────────────────┐
                    │ Vantage broker │
                    │ (your account) │
                    └────────────────┘
```

**Communication flow:**
- Bot → Database: writes trades, logs, status (constantly)
- Dashboard → Database: reads trades for display, writes config changes
- Database → Bot: bot polls config table every 30 seconds for changes (kill switch, parameter updates)
- Bot → Broker: via MT5 desktop (running on the same VPS)

This way, Tommy mostly works in his familiar GitHub/Vercel/Supabase environment. The VPS is essentially a "black box" that runs the bot — Tommy SSHs in only for initial setup and rare debugging.

---

## 10. Project Structure

```
gold-bot/
├── bot/
│   ├── main.py
│   ├── strategy/
│   │   ├── pattern_detection.py    # W/M/N detection on line chart
│   │   ├── structure.py            # HH/HL/LL/LH tracking, BoS detection
│   │   ├── zone_marking.py         # Box → refined zone
│   │   ├── strong_point.py         # Strong Point validation
│   │   └── imbalance.py            # Imbalance Zone tracking
│   ├── execution/
│   │   ├── mt5_connector.py
│   │   ├── order_manager.py
│   │   └── position_tracker.py
│   ├── risk/
│   │   ├── position_sizing.py      # 0.33% per layer calc
│   │   ├── daily_halt.py
│   │   └── exposure_check.py       # Max 2-3 setups
│   ├── exits/
│   │   ├── tp1_manager.py          # TP1 detection, 50% close, SL→BE
│   │   └── sl_manager.py
│   ├── filters/
│   │   ├── news_filter.py
│   │   └── zone_size_filter.py
│   ├── data/
│   │   ├── ohlc_provider.py
│   │   └── tick_handler.py
│   ├── logging/
│   │   ├── supabase_logger.py
│   │   └── telegram_alerts.py
│   └── config.py
├── backtest/
│   ├── run_backtest.py
│   ├── data_loader.py
│   └── results_analyzer.py
├── dashboard/                       # Next.js
├── tests/
├── .env
├── requirements.txt
└── README.md
```

---

## 11. Backtest Plan

### 11.1 Data requirements

- 3-5 years XAUUSD M1 data (Dukascopy)
- Realistic spread modelling (Vantage Gold spread: 20-40 points typical)
- Slippage modelling (1-3 points per fill)

### 11.2 Pass criteria

- Profitable on out-of-sample data (train: 2020-2023, test: 2024-2025)
- Max drawdown <20%
- Profit factor >1.3
- At least 200 trades in sample
- Win rate >55%

### 11.3 Parameters to optimize

- Zone size filter ranges (currently 5-80, optimize)
- TP1 distance (currently $4, optimize $2-10 range)
- SL buffer (currently 15-20 points, optimize 10-30 range)
- "Recent low" lookback (currently 20 candles, optimize 10-50)
- Pattern tolerance (currently 0.1%, optimize 0.05-0.2%)

**Avoid overfitting:** Walk-forward analysis, no more than 5 simultaneous parameter optimizations.

---

## 12. Build Phases

| Phase | Duration | Output |
|---|---|---|
| A. Foundation | Week 1 | Project scaffolding, Supabase schema, MT5 connection test |
| B. Core strategy | Week 2-3 | W/M/N detection, structure tracking, zone refinement |
| C. Execution layer | Week 3-4 | Order manager, position sizing, risk checks |
| D. Backtest | Week 4-5 | Historical run, parameter tuning |
| E. Dashboard | Week 5-6 | Monitoring UI, alerts |
| F. Demo trading | Month 2-3 | 30+ days demo |
| G. Live micro | Month 3-4 | $200-500 live, 30-60 days |
| H. Scale + vSocial | Month 4+ | Increase size, set up signal account |

Realistic time to vSocial signal: 4-6 months.

---

## 13. Open Items (to resolve via Telegram export, but not blocking build)

These can be defaulted in v1 and refined later:

- Pattern tolerance % — defaulted to 0.1%, refine in backtest
- "Recent" lookback for SL — defaulted to 20 candles, refine in backtest
- Approach distance for Imbalance qualification — defaulted to 5-10 points, refine in backtest
- Exact SL buffer on Gold — defaulted to 15-20 points (per education), refine in backtest

---

## 14. Definition of Done — Spec Phase

- [x] Strategy fully described
- [x] Setup types decided (Strong Point + Imbalance only)
- [x] Entry rules locked (first touch, 3 layers)
- [x] SL rules locked (real SL → BE after TP1)
- [x] TP1 rule locked ($4-5 beyond zone)
- [x] Runner management decided (manual)
- [x] Risk parameters locked (0.33% per layer)
- [x] Tech stack decided
- [x] Project structure planned

This spec is now complete enough to start building. Open items above will be resolved via backtest, not more upfront discussion.

---

## 15. Operator Commitment (Tommy)

The bot is not a "set and forget" system. Tommy commits to the following operational duties:

### 15.1 Backtest phase

- Active review of backtest results before any live deployment
- Iterative parameter tuning based on walk-forward analysis
- No live deployment until backtest pass criteria (Section 11.2) are met

### 15.2 Demo phase

- Daily review of bot's trade decisions
- Compare bot behaviour vs Tommy's manual judgement on same setups
- Flag and document divergences for spec refinement
- Minimum 30 days demo before live

### 15.3 Live phase (when active)

- Active monitoring of all open positions
- Manual intervention available at all times:
  - Closing trades early if market conditions warrant
  - Tightening SL on runner trades after BoS
  - Pausing bot during unusual market conditions
- Daily P&L review
- Weekly performance review (win rate, drawdown, profit factor vs backtest expectations)

### 15.4 Kill switch

- Bot must have an easily accessible "halt all trading" button (dashboard + Telegram command)
- Tommy can stop the bot from any device at any time
- All open positions remain controllable via MT5 mobile app independently

### 15.5 Performance review triggers

The bot is paused for review if any of these occur:
- 3 consecutive losing setups
- 5 losing setups within 24 hours
- 10% account drawdown from peak (regardless of daily limit)
- Bot behaviour diverges materially from Tommy's manual analysis on 3+ setups in a row
- Any unexplained order rejection or execution error

---

## 16. What's NOT in v1 (defer to v1.1+)

- Risk-based position sizing (0.33% per layer) — v1 uses fixed 0.01 lots
- vSocial signal provision — requires risk-based sizing first
- CHoCH detection and entry
- SnD Flip detection
- Standalone N-pattern entries
- High Risk signals (judgement-based)
- FX pairs (only XAUUSD in v1)
- Indices (defer)
- Automated trail SL
- Multiple broker support (Vantage only)

---

End of Specification v4.0 — Final
