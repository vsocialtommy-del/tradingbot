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
