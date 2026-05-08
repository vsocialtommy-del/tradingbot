"""Tests for ``bot.backtest.metrics``.

Hand-rolled inputs with arithmetic worked out manually so test failures
read like "expected $34.10, got $X" rather than "metric mismatch."
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from bot.backtest.metrics import (
    BE_TOLERANCE,
    compute_equity_metrics,
    compute_metrics,
    compute_setup_metrics,
    compute_trade_metrics,
)
from bot.backtest.simulator import BacktestPosition, CloseReason


NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_pos(
    *,
    pnl: float,
    setup_id: int = 1,
    layer: int = 1,
    direction: str = "BUY",
    entry: float = 1900.0,
    sl: float = 1880.0,
    lot_size: float = 0.01,
    close_reason: CloseReason = CloseReason.TP1,
    duration_minutes: float = 30.0,
    opened_at: datetime | None = None,
) -> BacktestPosition:
    opened = opened_at or NOW
    return BacktestPosition(
        ticket=100_000 + layer,
        setup_id=setup_id,
        layer=layer,
        direction=direction,  # type: ignore[arg-type]
        entry_price=entry,
        lot_size=lot_size,
        sl=sl,
        tp=1907.0 if direction == "BUY" else 1893.0,
        opened_at=opened,
        status="CLOSED",
        closed_lots=lot_size,
        exit_price=entry + 7 if pnl > 0 else entry - 20,
        exit_time=opened + timedelta(minutes=duration_minutes),
        close_reason=close_reason,
        realised_pnl=pnl,
        commission_paid=0.35,
    )


# --------------------------------------------------------------------------- #
# Trade metrics
# --------------------------------------------------------------------------- #


class TestTradeMetrics:
    def test_empty_returns_zero_metrics(self) -> None:
        m = compute_trade_metrics([])
        assert m.total == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0

    def test_known_winners_and_losers(self) -> None:
        positions = [
            make_pos(pnl=34.10, layer=1),
            make_pos(pnl=-20.50, layer=1, setup_id=2),
            make_pos(pnl=50.00, layer=1, setup_id=3),
            make_pos(pnl=-10.00, layer=1, setup_id=4),
        ]
        m = compute_trade_metrics(positions)
        assert m.total == 4
        assert m.winners == 2
        assert m.losers == 2
        assert m.breakevens == 0
        assert m.win_rate == pytest.approx(0.5)
        # avg_winner = (34.10 + 50) / 2 = 42.05
        assert m.avg_winner == pytest.approx(42.05)
        # avg_loser = (-20.50 + -10) / 2 = -15.25
        assert m.avg_loser == pytest.approx(-15.25)
        assert m.largest_winner == 50.00
        assert m.largest_loser == -20.50
        # gross_profit = 84.10, gross_loss = 30.50, PF = 2.7574
        assert m.profit_factor == pytest.approx(84.10 / 30.50)
        # expectancy = (34.10 - 20.50 + 50 - 10) / 4 = 53.60 / 4 = 13.40
        assert m.expectancy == pytest.approx(13.40)
        assert m.net_pnl == pytest.approx(53.60)

    def test_breakevens_excluded_from_win_rate(self) -> None:
        # 1 winner, 1 loser, 1 BE → win_rate based on 2 decided trades.
        positions = [
            make_pos(pnl=10.00),
            make_pos(pnl=-5.00, setup_id=2),
            make_pos(pnl=0.005, setup_id=3),  # below BE_TOLERANCE
        ]
        m = compute_trade_metrics(positions)
        assert m.breakevens == 1
        # 1 win out of 2 decided = 50%
        assert m.win_rate == pytest.approx(0.5)

    def test_all_winners_profit_factor_inf(self) -> None:
        positions = [
            make_pos(pnl=10.0),
            make_pos(pnl=20.0, setup_id=2),
        ]
        m = compute_trade_metrics(positions)
        assert math.isinf(m.profit_factor)

    def test_all_losers_profit_factor_zero(self) -> None:
        positions = [
            make_pos(pnl=-10.0, close_reason=CloseReason.SL),
            make_pos(pnl=-5.0, setup_id=2, close_reason=CloseReason.SL),
        ]
        m = compute_trade_metrics(positions)
        assert m.profit_factor == 0.0

    def test_avg_duration_minutes(self) -> None:
        positions = [
            make_pos(pnl=10.0, duration_minutes=30),
            make_pos(pnl=20.0, setup_id=2, duration_minutes=60),
        ]
        m = compute_trade_metrics(positions)
        assert m.avg_duration_minutes == pytest.approx(45.0)

    def test_r_multiple_calculation(self) -> None:
        # Entry 1900, SL 1880 → risk = 20 * 0.01 * 100 = $20.
        # Trade pnl 40 → R = 2.0.
        positions = [
            make_pos(pnl=40.0, entry=1900.0, sl=1880.0, lot_size=0.01),
        ]
        m = compute_trade_metrics(positions)
        assert m.avg_r_multiple == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Equity metrics
# --------------------------------------------------------------------------- #


class TestEquityMetrics:
    def test_empty_curve_returns_zero(self) -> None:
        m = compute_equity_metrics(pd.Series(dtype=float), 10_000.0)
        assert m.starting_balance == 10_000.0
        assert m.ending_balance == 0.0

    def test_uptrend_no_drawdown(self) -> None:
        idx = pd.date_range(NOW, periods=5, freq="1D", tz="UTC")
        eq = pd.Series([10_000, 10_100, 10_200, 10_300, 10_400], index=idx)
        m = compute_equity_metrics(eq, 10_000.0)
        assert m.ending_balance == 10_400.0
        assert m.total_return_pct == pytest.approx(4.0)
        assert m.max_drawdown_dollars == 0.0
        assert m.max_drawdown_pct == 0.0

    def test_drawdown_calculation(self) -> None:
        idx = pd.date_range(NOW, periods=5, freq="1D", tz="UTC")
        eq = pd.Series([10_000, 11_000, 9_900, 10_500, 10_200], index=idx)
        m = compute_equity_metrics(eq, 10_000.0)
        # peak 11_000, trough 9_900 → drawdown $1_100, pct 10%.
        assert m.max_drawdown_dollars == pytest.approx(1_100.0)
        assert m.max_drawdown_pct == pytest.approx(10.0, abs=0.01)

    def test_best_and_worst_day(self) -> None:
        idx = pd.date_range(NOW, periods=5, freq="1D", tz="UTC")
        eq = pd.Series([10_000, 10_500, 10_300, 10_400, 10_100], index=idx)
        m = compute_equity_metrics(eq, 10_000.0)
        # Daily diffs: +500, -200, +100, -300.
        assert m.best_day_pnl == pytest.approx(500.0)
        assert m.worst_day_pnl == pytest.approx(-300.0)


# --------------------------------------------------------------------------- #
# Setup metrics
# --------------------------------------------------------------------------- #


class TestSetupMetrics:
    def test_tp1_hit_counted_per_setup(self) -> None:
        # Setup 1: layer 1 hit TP1, layer 2 hit TP1 (counts ONCE).
        # Setup 2: layer 1 hit SL.
        positions = [
            make_pos(pnl=34.10, setup_id=1, layer=1, close_reason=CloseReason.TP1),
            make_pos(pnl=34.10, setup_id=1, layer=2, close_reason=CloseReason.TP1),
            make_pos(pnl=-20.50, setup_id=2, layer=1, close_reason=CloseReason.SL),
        ]
        m = compute_setup_metrics(
            detected=5, taken=2, skip_reasons={"too_narrow": 3},
            closed_positions=positions,
        )
        assert m.detected == 5
        assert m.taken == 2
        assert m.skipped == 3
        assert m.skip_reasons == {"too_narrow": 3}
        assert m.tp1_hit_count == 1
        assert m.sl_stop_count == 1
        # 2 setups total → 50% TP1, 50% SL
        assert m.tp1_hit_rate == pytest.approx(0.5)
        assert m.sl_stop_rate == pytest.approx(0.5)

    def test_no_setups_returns_zero_rates(self) -> None:
        m = compute_setup_metrics(
            detected=0, taken=0, skip_reasons={}, closed_positions=[],
        )
        assert m.tp1_hit_rate == 0.0
        assert m.sl_stop_rate == 0.0


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #


class TestComputeMetricsTopLevel:
    def test_combines_subsections(self) -> None:
        idx = pd.date_range(NOW, periods=3, freq="1D", tz="UTC")
        eq = pd.Series([10_000, 10_050, 10_080], index=idx)
        positions = [
            make_pos(pnl=80.0, setup_id=1, close_reason=CloseReason.TP1),
        ]
        m = compute_metrics(
            closed_positions=positions,
            equity_curve=eq,
            starting_balance=10_000.0,
            setups_detected=1,
            setups_taken=1,
        )
        assert m.trades.total == 1
        assert m.equity.ending_balance == 10_080.0
        assert m.setups.detected == 1
        assert m.setups.taken == 1
        assert len(m.r_multiples) == 1
