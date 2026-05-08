# Backtest framework

The `bot.backtest` package replays the live strategy pipeline over
historical OHLC data, simulates broker behaviour, and produces a
metrics summary plus interactive Plotly charts.

## What's in the box

| Module | Purpose |
|---|---|
| `bot.backtest.data_loader` | Load Dukascopy / generic CSV → tz-aware DataFrame |
| `bot.backtest.simulator` | `BacktestBroker`, OHLC walk tick generation |
| `bot.backtest.engine` | `BacktestEngine.run(df) → BacktestResult` |
| `bot.backtest.metrics` | Trade / equity / setup metrics |
| `bot.backtest.reporter` | Plotly charts + self-contained HTML report |

The strategy modules used (`bot.strategy.*`) are the **same modules**
the live bot uses. A zone the backtest detects is the zone the live
bot would detect — that's what makes backtest-driven tuning meaningful.

## Running the demo notebook

The repo ships with `backtest_demo.ipynb` — an end-to-end walkthrough
that loads data, runs the engine, renders the charts, and writes an
HTML report.

### Option A — Google Colab

1. Open <https://colab.research.google.com>
2. **File → Open notebook → GitHub** tab → paste the repo URL
3. Select `backtest_demo.ipynb`
4. **Runtime → Run all**

The notebook auto-detects Colab via `IN_COLAB`, clones the repo, and
installs deps. The first time, allow ~1 minute for the clone + install.

### Option B — Local (Jupyter / VS Code)

```bash
# from the repo root
pip install -r requirements.txt
pip install jupyter   # not in requirements.txt — only needed for the notebook
jupyter notebook backtest_demo.ipynb
```

Or open the notebook directly in VS Code (Python extension installed).

The Colab-specific cells (`google.colab` import, `files.upload()`)
silently no-op on local — the synthetic data fallback is used unless
you set `CSV_PATH` manually in Section 2.

## Getting real data

The demo's synthetic generator is for smoke-testing the pipeline only.
For meaningful backtest results, use real Dukascopy data:

1. <https://www.dukascopy.com/swiss/english/marketwatch/historical/>
2. **Symbol**: XAU/USD
3. **Timeframe**: 5 Min
4. **Date range**: 1+ year recommended
5. **Format**: CSV, **UTC** timezone
6. Click **Download**
7. In Colab: run the upload cell. Local: set `CSV_PATH` in Section 2.

`bot.backtest.data_loader.load_dukascopy_csv` handles Dukascopy's
`dd.mm.yyyy hh:mm:ss.fff` format and ISO 8601, auto-converts to UTC,
deduplicates timestamps, and validates OHLC consistency.

## Expected runtime

Approximate wall-clock on a Colab CPU runtime:

| Bars | Time |
|---|---|
| 1,000 (~3 days M5; synthetic) | < 1 s |
| 50,000 (~6 months M5) | 10-30 s |
| 200,000 (~2 years M5) | 1-3 min |

The bottleneck is the strategy pipeline (`run_strategy_pipeline`), run
once per M5 bar. `BacktestConfig.progress_log_every_bars` controls
progress logging cadence (default 1,000).

## Interpreting the metrics

### Strong run

* **Profit factor** > 1.5
* **Win rate** > 40% combined with **avg R-multiple** > 0.5R
* **Max drawdown** < 15%
* **Sharpe (ann)** > 1.0
* **TP1 hit rate** ≥ 50% (matches the spec target)

### Suspect run

* **PF < 1.0 across multiple data ranges** → strategy logic flaw, not just bad luck
* **Big winning streaks followed by big losses** → regime sensitivity; backtest may be overfit
* **Low setup count** (< 1 / week) → filters too strict
* **High setup count with poor PF** → filters too loose

## Caveats

* **No news filter** — Finnhub historical data isn't available in the
  pipeline. Real performance may differ on NFP, CPI, FOMC days.
  Quantify by sub-setting your CSV to exclude high-impact event windows
  and re-running.
* **Tick approximation** — the OHLC walk method (4 ticks per M5 bar)
  is the standard fallback when real ticks aren't available, but it's
  approximate. Same-bar SL+TP conflict resolves SL-wins (pessimistic)
  by default; flip via `BacktestConfig.sl_tp_conflict_sl_wins=False`
  for sensitivity testing.
* **Fixed lot size** in v1 (`BacktestConfig.fixed_lot_size = 0.01`).
  v1.1 will integrate `bot.risk.position_sizing` for risk-based sizing.
* **Pessimism layer** — slippage moves every fill in the adverse
  direction. This biases results worse than reality; treat the
  numbers as a conservative lower bound.

## Programmatic use

For non-notebook workflows:

```python
import pandas as pd
from bot.backtest import (
    BacktestConfig, BacktestEngine,
    load_dukascopy_csv,
    generate_html_report,
)

df = load_dukascopy_csv("xauusd_m5.csv")
result = BacktestEngine(BacktestConfig()).run(df)
print(f"Net PnL: ${result.metrics.trades.net_pnl:+.2f}")
generate_html_report(result, "report.html", df=df)
```

The `BacktestResult` dataclass exposes `metrics`, `closed_positions`,
`equity_curve`, `setups_detected`, `setups_taken`, `skip_reasons`, and
`config`. See `bot.backtest.metrics` and `bot.backtest.engine` for the
full schema.

## Tuning checklist

When iterating on parameters:

1. **Change one knob at a time.** Multi-parameter changes muddle which
   change moved which metric.
2. **Archive each report** — rename `backtest_report.html` to include
   the parameter set (`report_layer2_5_layer3_10.html`).
3. **Re-run on multiple data periods** before drawing conclusions.
   A single 1-year window can be regime-specific.
4. **Look at skip reasons** — `exposure_cap`, `sl_too_close`,
   `sl_too_far` reveal whether the bot's filters are doing what you
   intended.
