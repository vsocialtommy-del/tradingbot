"""M5 bot quality rules (Tommy, 24 Nov 2026): H1 trend, 3 trades a day,
TP at least 0.3 x the SL distance."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from bot.profiles import build_profile
from bot.risk.xautism_risk import DailyRisk, RiskConfig, check_risk, summarise_day
from bot.strategy.h1_trend import allowed, h1_ema_trend
from bot.strategy.levels import TradeLevels
from tests.test_main import (  # noqa: F401
    _xautism,
    bot,
    make_ohlc,
    mock_mt5,
    mock_supabase,
)
from tests.test_order_manager import make_zone_for_test


def _m5(closes, start="2026-11-20 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(closes), freq="5min", tz="UTC")
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c}, index=idx)


# --------------------------------------------------------------------------- #
# H1 trend
# --------------------------------------------------------------------------- #

def test_rising_market_is_up_falling_is_down() -> None:
    n = 12 * 80                                   # 80 hours of M5
    assert h1_ema_trend(_m5(np.linspace(3500, 3700, n))) == "UP"
    assert h1_ema_trend(_m5(np.linspace(3700, 3500, n))) == "DOWN"


def test_not_enough_history_is_no_trend_and_allows_both() -> None:
    t = h1_ema_trend(_m5(np.linspace(3500, 3700, 12 * 20)))   # 20 hours < 50
    assert t is None and allowed("BUY", t) and allowed("SELL", t)


def test_buys_need_up_sells_need_down() -> None:
    assert allowed("BUY", "UP") and not allowed("BUY", "DOWN")
    assert allowed("SELL", "DOWN") and not allowed("SELL", "UP")


def test_only_closed_h1_candles_count() -> None:
    # 79 hours of rally, then the last hour drops hard. With that hour
    # still forming (one M5 candle missing) the trend is still UP.
    up = list(np.linspace(3500, 3700, 12 * 79))
    drop = list(np.linspace(3690, 3300, 12))
    assert h1_ema_trend(_m5(up + drop[:-1])) == "UP"
    assert h1_ema_trend(_m5(up + drop)) == "DOWN"


# --------------------------------------------------------------------------- #
# 3 trades a day
# --------------------------------------------------------------------------- #

CAP = RiskConfig(max_trades_per_day=3, max_daily_loss=1000)


@pytest.mark.parametrize("closed, open_, blocked", [(2, 0, False), (3, 0, True), (2, 1, True)])
def test_cap_counts_closed_and_open_trades(closed, open_, blocked) -> None:
    d = DailyRisk("x", 10.0, 0.0, closed, 0, winning_trades=closed, open_trades=open_)
    v = check_risk(d, CAP)
    assert v.blocked is blocked
    if blocked:
        assert "trades taken today (max 3)" in v.reason


def test_no_cap_by_default() -> None:
    assert not check_risk(DailyRisk("x", 10.0, 0.0, 9, 0, winning_trades=9), RiskConfig()).blocked


def test_summarise_day_counts_open_trades() -> None:
    from datetime import datetime, timezone
    pos = [{"ticket": 7, "magic": 1, "comment": "", "profit": 3.0, "swap": 0.0}]
    d = summarise_day([], pos, magic=1, day_start=datetime(2026, 11, 24, tzinfo=timezone.utc),
                      trading_day="2026-11-24")
    assert d.open_trades == 1


# --------------------------------------------------------------------------- #
# Profiles: M5 bot only
# --------------------------------------------------------------------------- #

def test_m5_bot_has_the_three_rules_claude_ftmo_does_not() -> None:
    m5 = build_profile("m5")[0]
    assert (m5.h1_trend_ema_period, m5.min_reward_frac, m5.risk.max_trades_per_day) == (50, 0.3, 3)
    for cfg in (build_profile("m1")[0], build_profile("m1", plan=True)[0]):
        assert cfg.h1_trend_ema_period is None and cfg.min_reward_frac is None
        assert cfg.risk.max_trades_per_day is None


# --------------------------------------------------------------------------- #
# The bot
# --------------------------------------------------------------------------- #

def _ready(b, mocker, *, tp: float, period=50, frac=0.3):
    _xautism(b)
    b.config = replace(
        b.config, h1_trend_ema_period=period, min_reward_frac=frac,
        direction_gate=replace(b.config.direction_gate, enabled=False),
    )
    mocker.patch("bot.main.count_bodies_through_zone", return_value=0)
    # BUY zone 1895-1900: entry 1900, SL 1893.5 (6.5 pts of risk).
    mocker.patch("bot.main.compute_levels",
                 return_value=TradeLevels(entry=1900.0, sl=1893.5, tp1=tp, sl_method="PCT_30"))
    return mocker.patch.object(b, "_place_xautism", return_value=True)


def test_low_reward_zone_skipped_once_and_never_retried(bot, mocker) -> None:  # noqa: F811
    b, _ = bot
    place = _ready(b, mocker, tp=1901.5)          # 1.5 pts for 6.5 = 0.23 < 0.3
    mocker.patch("bot.main.h1_ema_trend", return_value="UP")
    zone, zid = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0), uuid4()
    assert b._try_place_setup(zone, make_ohlc(n=30), zid, bid=1899.9, ask=1900.0) is False
    assert zid in b._low_reward_zone_ids
    cl = mocker.patch("bot.main.compute_levels")
    assert b._try_place_setup(zone, make_ohlc(n=30), zid, bid=1899.9, ask=1900.0) is False
    cl.assert_not_called()
    place.assert_not_called()


def test_enough_reward_with_the_trend_trades(bot, mocker) -> None:  # noqa: F811
    b, _ = bot
    place = _ready(b, mocker, tp=1902.0)          # 2.0 for 6.5 = 0.31
    mocker.patch("bot.main.h1_ema_trend", return_value="UP")
    zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
    assert b._try_place_setup(zone, make_ohlc(n=30), uuid4(), bid=1899.9, ask=1900.0) is True
    assert place.call_args.kwargs["tp1_price"] == 1902.0


def test_against_the_h1_trend_skipped_but_can_trade_later(bot, mocker) -> None:  # noqa: F811
    b, _ = bot
    place = _ready(b, mocker, tp=1905.0)
    trend = mocker.patch("bot.main.h1_ema_trend", return_value="DOWN")
    zone, zid = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0), uuid4()
    assert b._try_place_setup(zone, make_ohlc(n=30), zid, bid=1899.9, ask=1900.0) is False
    place.assert_not_called()
    trend.return_value = "UP"
    assert b._try_place_setup(zone, make_ohlc(n=30), zid, bid=1899.9, ask=1900.0) is True


def test_rules_off_trade_as_before(bot, mocker) -> None:  # noqa: F811
    b, _ = bot
    place = _ready(b, mocker, tp=1900.2, period=None, frac=None)
    trend = mocker.patch("bot.main.h1_ema_trend", return_value="DOWN")
    zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
    assert b._try_place_setup(zone, make_ohlc(n=30), uuid4(), bid=1899.9, ask=1900.0) is True
    trend.assert_not_called()
    place.assert_called_once()
