"""Claude FTMO's day follows FTMO's daily loss reset (00:00 Prague time,
23:00 UK), not the broker's 17:00 New York (22:00 UK). Tommy, 24 Nov 2026."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from bot.profiles import build_profile
from bot.risk.xautism_risk import (
    DailyRiskTracker,
    RiskConfig,
    trading_day_label,
    trading_day_start,
)

UTC = timezone.utc


def test_ftmo_day_starts_at_23_uk_in_winter_and_summer() -> None:
    # Winter: Prague UTC+1, so midnight there = 23:00 UTC = 23:00 UK.
    assert trading_day_start(datetime(2026, 11, 24, 22, 30, tzinfo=UTC), "ftmo") == \
        datetime(2026, 11, 23, 23, 0, tzinfo=UTC)
    assert trading_day_start(datetime(2026, 11, 24, 23, 30, tzinfo=UTC), "ftmo") == \
        datetime(2026, 11, 24, 23, 0, tzinfo=UTC)
    # Summer: Prague UTC+2, midnight = 22:00 UTC = 23:00 UK (BST).
    assert trading_day_start(datetime(2026, 7, 1, 21, 59, tzinfo=UTC), "ftmo") == \
        datetime(2026, 6, 30, 22, 0, tzinfo=UTC)
    assert trading_day_label(datetime(2026, 11, 24, 23, 30, tzinfo=UTC), "ftmo") == "2026-11-25"


def test_broker_day_unchanged() -> None:
    # 17:00 New York = 22:00 UTC in winter.
    assert trading_day_start(datetime(2026, 11, 24, 22, 30, tzinfo=UTC)) == \
        datetime(2026, 11, 24, 22, 0, tzinfo=UTC)


def test_the_gap_hour_counts_with_the_evening() -> None:
    """A -£190 loss closed at 21:30 UK still counts at 22:30 UK on the FTMO
    day (it would have been wiped at 22:00 on the broker day)."""
    mt5 = MagicMock()
    mt5.magic = 7
    mt5.get_open_positions.return_value = []
    t_in = datetime(2026, 11, 24, 21, 0, tzinfo=UTC).timestamp()
    t_out = datetime(2026, 11, 24, 21, 30, tzinfo=UTC).timestamp()
    mt5.get_deals.return_value = [
        {"entry": 0, "magic": 7, "position_id": 1, "comment": "", "volume": 0.2, "time": t_in},
        {"entry": 1, "magic": 7, "position_id": 1, "profit": -190.0, "time": t_out},
    ]
    mt5.get_tick.return_value = None
    now = datetime(2026, 11, 24, 22, 30, tzinfo=UTC)
    ftmo = RiskConfig(risk_per_trade=50, max_daily_loss=200, room_check=True,
                      max_losses_per_day=50, day_reset="ftmo")
    v = DailyRiskTracker(mt5, cache_seconds=0).check(now, ftmo)
    assert v.daily.realised == -190.0 and v.blocked
    from dataclasses import replace
    v2 = DailyRiskTracker(mt5, cache_seconds=0).check(now, replace(ftmo, day_reset="new_york"))
    assert v2.daily.realised == 0.0 and not v2.blocked


def test_claude_ftmo_uses_the_ftmo_day_m5_bot_does_not() -> None:
    assert build_profile("m1", plan=True)[0].risk.day_reset == "ftmo"
    assert build_profile("m5")[0].risk.day_reset == "new_york"
    assert build_profile("m1")[0].risk.day_reset == "new_york"
