"""Claude FTMO profit lock (Tommy, 25 Nov 2026): once the day's closed P&L
reaches +£100, no new trades if it falls back to half the day's best."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.main import plan_risk
from bot.plan import PlanConfig
from bot.profiles import build_profile
from bot.risk.xautism_risk import (
    DEAL_ENTRY_IN,
    DEAL_ENTRY_OUT,
    DailyRisk,
    RiskConfig,
    check_risk,
    summarise_day,
)

MAGIC = 7
START = datetime(2026, 11, 24, 23, 0, tzinfo=timezone.utc)
LOCK = RiskConfig(max_daily_loss=200, room_check=True, max_losses_per_day=50,
                  risk_per_trade=50, giveback_after=100.0, giveback_keep=0.5)


def _day(realised: float, peak: float) -> DailyRisk:
    return DailyRisk("2026-11-25", realised, 0.0, 3, 1, winning_trades=2, realised_peak=peak)


@pytest.mark.parametrize("realised, peak, blocked", [
    (90.0, 90.0, False),     # never reached +100: off
    (40.0, 90.0, False),     # dropped, but never reached +100
    (105.0, 105.0, False),   # at the best: keep going
    (60.0, 100.0, False),    # above half of +100
    (50.0, 100.0, True),     # half of +100
    (37.0, 135.0, True),
    (120.0, 200.0, False),   # best +200 stops at +100
    (100.0, 200.0, True),
    (210.0, 400.0, False),
    (200.0, 400.0, True),
])
def test_stops_at_half_the_days_best_after_100(realised, peak, blocked) -> None:
    v = check_risk(_day(realised, peak), LOCK)
    assert v.blocked is blocked
    if blocked:
        assert v.reason.startswith("profit lock")


def test_off_by_default() -> None:
    assert not check_risk(_day(10.0, 400.0), RiskConfig(max_losses_per_day=50)).blocked


def _deal(pos, entry, t, profit=0.0, commission=0.0, comment=""):
    return {"position_id": pos, "entry": entry, "time": t.timestamp(), "profit": profit,
            "commission": commission, "swap": 0.0, "magic": MAGIC if entry == DEAL_ENTRY_IN else 0,
            "comment": comment, "volume": 0.1}


def test_summarise_day_tracks_the_best_and_entry_commission() -> None:
    h = lambda hh, mm=0: datetime(2026, 11, 25, hh, mm, tzinfo=timezone.utc)  # noqa: E731
    deals = [
        _deal(1, DEAL_ENTRY_IN, h(1), commission=-0.5, comment="bot:L1:s=aaaaaaaa"),
        _deal(1, DEAL_ENTRY_OUT, h(1, 30), profit=120.0, commission=-0.5),
        _deal(2, DEAL_ENTRY_IN, h(2), commission=-0.5, comment="bot:L1:s=bbbbbbbb"),
        _deal(2, DEAL_ENTRY_OUT, h(2, 30), profit=-50.0, commission=-0.5),
    ]
    d = summarise_day(deals, [], magic=MAGIC, day_start=START, trading_day="2026-11-25")
    assert d.realised_peak == 119.0          # 120 - both commissions of trade 1
    assert d.realised == 68.0
    assert d.closed_trades == 2 and d.losing_trades == 1
    assert not check_risk(d, LOCK).blocked   # 68 > half of 119
    deals.append(_deal(3, DEAL_ENTRY_IN, h(3), comment="bot:L1:s=cccccccc"))
    deals.append(_deal(3, DEAL_ENTRY_OUT, h(3, 10), profit=-10.0))
    d = summarise_day(deals, [], magic=MAGIC, day_start=START, trading_day="2026-11-25")
    assert d.realised == 58.0 and check_risk(d, LOCK).blocked


def test_claude_ftmo_has_the_lock_and_the_m5_bot_doesnt() -> None:
    ftmo = build_profile("m1", plan=True)[0]
    assert (ftmo.risk.giveback_after, ftmo.risk.giveback_keep) == (100.0, 0.5)
    assert build_profile("m5")[0].risk.giveback_after is None
    assert build_profile("m1")[0].risk.giveback_after is None


def test_dashboard_risk_change_keeps_the_lock() -> None:
    # plan_risk is re-applied when the dashboard risk setting changes.
    r = plan_risk(RiskConfig(risk_per_trade=80.0), PlanConfig())
    assert r.giveback_after == 100.0 and r.risk_per_trade == 80.0
    assert plan_risk(RiskConfig(), PlanConfig(profit_lock_after=None)).giveback_after is None
