"""Performance metrics for the backtest engine.

Pure functions over a list of closed trades + an equity time series.
No I/O, no plotting (the reporter module — PR #2 — will handle visuals).

A "trade" here is one closed :class:`bot.backtest.simulator.BacktestPosition`
— i.e. one *layer* of a setup, not the setup as a whole. We aggregate
per-layer P&L because that's the granularity at which money actually
changes hands. Per-setup metrics (e.g. "TP1 hit rate") are derived
separately from the engine's setup tracking — see :func:`compute_setup_metrics`.

Calculations
------------

* **win_rate** = winners / non-zero trades. Break-even closes (PnL == 0
  to a hairline tolerance) are excluded from both numerator and
  denominator so a setup that scratched at BE doesn't artificially drag
  the rate down.
* **profit_factor** = gross_profit / gross_loss. ``inf`` when all wins,
  ``0`` when all losses (and there ARE losses).
* **expectancy** = mean P&L per trade in dollars. Combines win rate and
  payoff ratio in one number.
* **R-multiple** for a trade = trade_pnl / risk_at_entry, where
  ``risk_at_entry = |entry - sl| * lots * contract_size`` (in dollars).
  R-multiples normalise across position sizes.
* **max_drawdown_pct** = (trough - peak) / peak from the equity curve.
  Reported as a positive percentage.
* **sharpe** is annualised assuming 252 trading days, computed from
  *daily* equity returns (not per-trade returns).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from bot.backtest.simulator import BacktestPosition, CloseReason


# Tolerance for treating a close as break-even.
BE_TOLERANCE: float = 0.01  # $0.01 — below this we call it a scratch
TRADING_DAYS_PER_YEAR: int = 252


# --------------------------------------------------------------------------- #
# Output dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TradeMetrics:
    total: int = 0
    winners: int = 0
    losers: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    avg_duration_minutes: float = 0.0
    avg_r_multiple: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0


@dataclass(frozen=True)
class EquityMetrics:
    starting_balance: float = 0.0
    ending_balance: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    max_drawdown_pct: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0
    sharpe_ratio: float = 0.0


@dataclass(frozen=True)
class SetupMetrics:
    """Per-setup aggregates (one setup may have multiple layer trades)."""

    detected: int = 0
    taken: int = 0
    skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    tp1_hit_count: int = 0
    sl_stop_count: int = 0
    be_stop_count: int = 0
    tp1_hit_rate: float = 0.0
    sl_stop_rate: float = 0.0


@dataclass(frozen=True)
class BacktestMetrics:
    trades: TradeMetrics
    equity: EquityMetrics
    setups: SetupMetrics
    r_multiples: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def compute_trade_metrics(
    closed_positions: list[BacktestPosition],
) -> TradeMetrics:
    """Aggregate per-layer trade P&L into headline trade stats."""
    if not closed_positions:
        return TradeMetrics()

    pnls = [p.realised_pnl for p in closed_positions]
    winners = [x for x in pnls if x > BE_TOLERANCE]
    losers = [x for x in pnls if x < -BE_TOLERANCE]
    breakevens = len(pnls) - len(winners) - len(losers)

    # win_rate excludes BE — see module docstring.
    decided = len(winners) + len(losers)
    win_rate = (len(winners) / decided) if decided > 0 else 0.0

    durations = [
        (p.exit_time - p.opened_at).total_seconds() / 60.0
        for p in closed_positions
        if p.exit_time is not None
    ]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    r_multiples = [_r_multiple(p) for p in closed_positions]
    finite_r = [r for r in r_multiples if math.isfinite(r)]
    avg_r = sum(finite_r) / len(finite_r) if finite_r else 0.0

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))
    if gross_loss > 0:
        pf = gross_profit / gross_loss
    elif gross_profit > 0:
        pf = float("inf")
    else:
        pf = 0.0

    return TradeMetrics(
        total=len(pnls),
        winners=len(winners),
        losers=len(losers),
        breakevens=breakevens,
        win_rate=win_rate,
        avg_winner=(sum(winners) / len(winners)) if winners else 0.0,
        avg_loser=(sum(losers) / len(losers)) if losers else 0.0,
        largest_winner=max(winners, default=0.0),
        largest_loser=min(losers, default=0.0),
        avg_duration_minutes=avg_duration,
        avg_r_multiple=avg_r,
        expectancy=sum(pnls) / len(pnls),
        profit_factor=pf,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_pnl=sum(pnls),
    )


def compute_equity_metrics(
    equity_curve: pd.Series,
    starting_balance: float,
) -> EquityMetrics:
    """Drawdown, daily returns, Sharpe — from a tz-aware equity ``Series``."""
    if equity_curve.empty:
        return EquityMetrics(starting_balance=starting_balance)

    ending = float(equity_curve.iloc[-1])
    total_ret = (ending - starting_balance) / starting_balance * 100.0

    # Drawdown from running peak.
    peaks = equity_curve.cummax()
    dd_dollars = float((equity_curve - peaks).min())  # negative or zero
    # Pct drawdown of the worst trough relative to its own peak.
    pct_dd = ((equity_curve - peaks) / peaks).fillna(0.0)
    dd_pct = float(pct_dd.min()) * 100.0

    # Daily P&L: sample equity at end-of-day, take diffs.
    daily = equity_curve.resample("1D").last().dropna()
    daily_pnl = daily.diff().dropna()
    if not daily_pnl.empty:
        best_day = float(daily_pnl.max())
        worst_day = float(daily_pnl.min())
    else:
        best_day = 0.0
        worst_day = 0.0

    sharpe = _sharpe(daily)

    return EquityMetrics(
        starting_balance=starting_balance,
        ending_balance=ending,
        total_return_pct=total_ret,
        max_drawdown_dollars=abs(dd_dollars),
        max_drawdown_pct=abs(dd_pct),
        best_day_pnl=best_day,
        worst_day_pnl=worst_day,
        sharpe_ratio=sharpe,
    )


def compute_setup_metrics(
    *,
    detected: int,
    taken: int,
    skip_reasons: dict[str, int],
    closed_positions: list[BacktestPosition],
) -> SetupMetrics:
    """Per-setup outcome breakdown.

    A setup is counted as TP1-hit if AT LEAST ONE of its layer
    positions closed with reason TP1. SL-hit if ALL its layer positions
    closed with SL (i.e. the setup never reached TP1). BE if it had a
    TP1 partial close AND the remaining runner closed at break-even
    (within the BE tolerance).
    """
    by_setup: dict[int, list[BacktestPosition]] = {}
    for pos in closed_positions:
        by_setup.setdefault(pos.setup_id, []).append(pos)

    tp1_hit = 0
    sl_stop = 0
    be_stop = 0
    for sid, positions in by_setup.items():
        reasons = [p.close_reason for p in positions]
        if CloseReason.TP1 in reasons:
            tp1_hit += 1
            # Did the runner come back to BE?
            for p in positions:
                if p.close_reason != CloseReason.TP1 and p.close_reason == CloseReason.SL:
                    if abs(p.realised_pnl) < BE_TOLERANCE:
                        be_stop += 1
                        break
        elif all(r == CloseReason.SL for r in reasons):
            sl_stop += 1

    n = len(by_setup) or 1  # avoid divide-by-zero
    return SetupMetrics(
        detected=detected,
        taken=taken,
        skipped=detected - taken,
        skip_reasons=dict(skip_reasons),
        tp1_hit_count=tp1_hit,
        sl_stop_count=sl_stop,
        be_stop_count=be_stop,
        tp1_hit_rate=tp1_hit / n,
        sl_stop_rate=sl_stop / n,
    )


def compute_metrics(
    *,
    closed_positions: list[BacktestPosition],
    equity_curve: pd.Series,
    starting_balance: float,
    setups_detected: int = 0,
    setups_taken: int = 0,
    skip_reasons: dict[str, int] | None = None,
) -> BacktestMetrics:
    """Top-level entry point used by :class:`BacktestEngine`."""
    return BacktestMetrics(
        trades=compute_trade_metrics(closed_positions),
        equity=compute_equity_metrics(equity_curve, starting_balance),
        setups=compute_setup_metrics(
            detected=setups_detected,
            taken=setups_taken,
            skip_reasons=skip_reasons or {},
            closed_positions=closed_positions,
        ),
        r_multiples=[
            _r_multiple(p) for p in closed_positions
            if math.isfinite(_r_multiple(p))
        ],
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _r_multiple(pos: BacktestPosition) -> float:
    """trade_pnl / risk_at_entry. ``inf`` when SL == entry."""
    risk = abs(pos.entry_price - pos.sl) * pos.lot_size * 100.0  # contract_size
    if risk == 0:
        return float("inf")
    return pos.realised_pnl / risk


def _sharpe(daily_equity: pd.Series) -> float:
    """Annualised Sharpe from daily equity. Risk-free rate = 0 for v1."""
    rets = daily_equity.pct_change().dropna()
    if len(rets) < 2:
        return 0.0
    std = rets.std()
    if std == 0 or math.isnan(std):
        return 0.0
    return float(rets.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))
