# XAUUSD Supply & Demand Trading Bot

A semi-automated bot that trades Gold (XAUUSD) on Vantage Markets via MT5
using a Supply & Demand strategy (Strong Point + Imbalance Zone setups).

**Source of truth:** [`docs/strategy-spec-v4-final.md`](docs/strategy-spec-v4-final.md).
**Build checklist:** [`docs/tommy-todo-list.md`](docs/tommy-todo-list.md).

Current phase: **Phase A — Foundation** (project scaffolding only — no
strategy logic yet; modules are docstring-only placeholders).

---

## Prerequisites

- Python 3.11 (pinned in `.python-version`)
- MetaTrader 5 desktop app installed and logged into a Vantage demo or live account
- A Supabase project (URL + service role key)
- A Finnhub free API key (only needed once we build the news cron in Phase E)

> **Platform note:** the `MetaTrader5` Python library is officially Windows
> only. Local dev works best on Windows. The production VPS (Hetzner,
> Phase G) runs MT5 + Python under Wine. Installing `MetaTrader5` on plain
> Linux without a terminal will fail — that's expected.

---

## Setup (run on your laptop)

```bash
# 1. Clone
git clone https://github.com/vsocialtommy-del/tradingbot.git
cd tradingbot

# 2. Create and activate a virtualenv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy env template and fill in real values
cp .env.example .env
# then edit .env with your MT5 / Supabase / Finnhub credentials
```

---

## Project structure

Mirrors Section 10 of the spec, flattened to repo root:

```
tradingbot/
├── bot/                  # Python bot — runs 24/5 on the VPS
│   ├── main.py
│   ├── config.py
│   ├── strategy/         # Setup detection (Sections 3, 5)
│   ├── execution/        # MT5 connector + order management (Section 4)
│   ├── risk/             # Position sizing, daily halt, exposure (Section 7)
│   ├── exits/            # TP1 + SL management (Sections 5, 6)
│   ├── filters/          # News + zone-size filters (Sections 7, 8)
│   ├── data/             # OHLC + tick providers
│   └── logging/          # Supabase + Telegram outputs
├── backtest/             # Phase D — historical run + parameter tuning
├── dashboard/            # Phase E — Next.js dashboard (placeholder)
├── tests/                # pytest suite
├── docs/                 # Strategy spec + Tommy's todo list
├── .env.example
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```

---

## Running tests

```bash
pytest
```

(No tests yet — they land alongside each module starting in Phase B.)

---

## Lint / format

```bash
ruff check .
black .
```

---

## Phase status

| Phase | Status |
|---|---|
| A. Foundation | 🔨 in progress (this scaffold) |
| B. Core strategy | ⏳ next |
| C. Execution layer | ⏳ |
| D. Backtest | ⏳ |
| E. Dashboard | ⏳ |
| F. Demo trading | ⏳ |
| G. Live micro | ⏳ |
| H. Scale + vSocial | ⏳ |

---

## MT5 chart zone visualization (PR #49)

The bot can render its currently-tracked zones as colored rectangles
on your MT5 chart. The Python side writes a CSV file each M5 close; a
companion MQL5 EA (`mql5/ZoneOverlay.mq5`) polls that file every 5
seconds and reconciles the chart rectangles.

### One-time setup

**On the VPS:**

1. **Find your MT5 data folder.** In MT5: `File → Open Data Folder`.
   Inside is `MQL5/Files/` — note the full path.
2. **Set the env var** (in your `.env` or systemd unit file):
   ```
   MT5_FILES_DIR=/absolute/path/to/MQL5/Files
   ```
   On Wine the path is typically under
   `~/.wine/drive_c/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/<instance>/MQL5/Files`.
3. Restart the bot. Look for `zone snapshot: enabled, writing to …`
   in the log to confirm. If the env var is unset, the writer is a
   no-op and the bot runs normally — visualization is just disabled.

**In MetaTrader 5 (one-time):**

4. Open **MetaEditor** (toolbar button or `F4`).
5. `File → Open` → navigate to this repo's `mql5/ZoneOverlay.mq5`.
6. Compile with `F7`. Produces `ZoneOverlay.ex5` next to the source.
7. Back in the MT5 terminal: open the Navigator panel (`Ctrl+N`) →
   **Expert Advisors** → drag `ZoneOverlay` onto your XAUUSD M5
   chart.
8. In the popup: tick **Allow Algo Trading** → OK.

Rectangles appear within ~5 seconds and refresh on every bot M5
close.

### Color scheme

| Zone state | Color |
|---|---|
| CONFIRMED BUY (fresh demand, not yet traded) | Sea green |
| CONFIRMED SELL (fresh supply, not yet traded) | Fire brick (dark red) |
| ACTIVE BUY (open setup) | Lime green (brighter) |
| ACTIVE SELL (open setup) | Crimson (brighter) |
| FLIPPED (any direction, PR #38 trade) | Dark orange |

CONSUMED / VIOLATED zones are not drawn (they're dead).

### Filters

The Python writer applies four filters before emitting to CSV (tunable
in `bot/visualization/zone_snapshot.py:ZoneSnapshotConfig`):

| Filter | Default | Effect |
|---|---|---|
| Distance | ±$50 from current price | Hides zones far from market |
| Age | 7 days | Hides week-old zones |
| Status | CONFIRMED / ACTIVE / FLIPPED | Hides dead zones |
| Visibility (PR #48) | ≥2 bodies through → drop | Hides obscured zones |

### Troubleshooting

* **No rectangles appearing**: check the EA's "Experts" tab at the
  bottom of MT5 for diagnostic prints. Most likely the file path is
  wrong — verify `tradingbot_zones.csv` exists in the directory you
  set as `MT5_FILES_DIR`.
* **Old rectangles linger after EA removed**: the EA has
  `InpCleanupOnExit=true` by default — they should be cleaned up.
  If not, manually remove via `Right-click chart → Objects List`.
* **Disable temporarily**: unset `MT5_FILES_DIR` and restart the bot;
  no CSV writes, EA stops getting updates.

Visualization failures NEVER affect trading — every CSV write is
wrapped in try/except in `bot/main.py`.
