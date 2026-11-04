"""XAUTISM backtester: broker-side behaviour on hand-made price paths."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.backtest.xautism import Layer, Trade, XautismBacktest, load_mt5_csv, summarise
from bot.strategy.stacked import StackedPlan

T = pd.Timestamp("2026-10-01 10:00", tz="UTC")


def bt() -> XautismBacktest:
    idx = pd.date_range(T, periods=3, freq="5min")
    m5 = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "spread": 0.3}, index=idx)
    return XautismBacktest(m5)


def buy_trade(*, l2_kind="STOP", l2_price=4000.0, l2_lots=0.18, stacked=None) -> Trade:
    return Trade(
        direction="BUY", zone_bottom=3996.0, zone_top=4000.0, edge=4000.0,
        far_edge=3996.0 if stacked is None else stacked.outer_far_edge,
        sl=3994.8, tp=4010.0, sl_method="PCT_30", opened_at=T, stacked=stacked,
        l1=Layer(0.01, entry=4000.3), l2=Layer(l2_lots, state="UNPLACED"),
        l2_price=l2_price, l2_kind=l2_kind, trend="", planned_risk=98.8,
    )


def feed(b: XautismBacktest, prices: list[float], spread: float = 0.3) -> None:
    for p in prices:
        if b.open is None:
            return
        b._manage(p, spread, T)


class TestLayer2Stop:
    def test_in_and_out_then_tp(self) -> None:
        b = bt(); b.open = buy_trade()
        feed(b, [3999.5, 3998.0, 3999.0, 4000.0, 4005.0, 4010.0])
        tr = b.trades[0]
        assert tr.exit_reason == "TP"
        assert tr.l2.entry == 4000.0 and tr.l2.exit == 4010.0
        assert tr.pnl == pytest.approx(0.01 * 9.7 * 100 + 0.18 * 10 * 100)

    def test_tp_before_layer2_cancels_it(self) -> None:
        b = bt(); b.open = buy_trade()
        feed(b, [4000.0, 4005.0, 4010.0])   # never went 0.10 inside
        tr = b.trades[0]
        assert tr.l2.state == "CANCELLED" and tr.pnl == pytest.approx(9.7)

    def test_sl_takes_both_layers(self) -> None:
        b = bt(); b.open = buy_trade()
        feed(b, [3998.0, 4000.0, 3997.0, 3994.8])
        tr = b.trades[0]
        assert tr.exit_reason == "SL"
        assert tr.pnl == pytest.approx(-(0.01 * 5.5 + 0.18 * 5.2) * 100)

    def test_sl_without_coming_back_out(self) -> None:
        b = bt(); b.open = buy_trade()
        feed(b, [3998.0, 3994.8])
        tr = b.trades[0]
        assert tr.l2.state == "CANCELLED" and tr.pnl == pytest.approx(-5.5)


def test_stacked_limit_fills_at_outer_zone() -> None:
    plan = StackedPlan("M1", 3992.0, 3990.0, 4.0, 3992.0, 3990.0, 3989.4)
    b = bt(); b.open = buy_trade(l2_kind="LIMIT", l2_price=3992.0, stacked=plan)
    b.open.sl = 3989.4
    feed(b, [3999.0, 3995.0, 3991.7, 4000.0, 4010.0])
    tr = b.trades[0]
    assert tr.l2.entry == 3992.0 and tr.exit_reason == "TP"


def test_two_losses_block_the_day() -> None:
    b = bt()
    for _ in range(2):
        b.open = buy_trade(); feed(b, [3994.8])
    assert b._blocked(T)
    assert not b._blocked(T + pd.Timedelta(days=1))


def test_summarise() -> None:
    df = pd.DataFrame({"pnl": [100.0, -50.0, -50.0, 200.0]})
    s = summarise(df)
    assert s["net"] == 200.0 and s["win_rate"] == 50.0
    assert s["max_drawdown"] == -100.0 and s["worst_losing_streak"] == 2


def test_load_mt5_csv(tmp_path) -> None:
    p = tmp_path / "x.csv"
    p.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2026.10.01\t01:05:00\t4000.0\t4001.0\t3999.0\t4000.5\t10\t0\t40\n"
    )
    d = load_mt5_csv(str(p))
    assert d.index[0] == pd.Timestamp("2026-10-01 01:05", tz="UTC")
    assert d["spread"].iloc[0] == pytest.approx(0.40)
