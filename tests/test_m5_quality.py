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

def test_m5_bot_has_the_three_rules() -> None:
    m5 = build_profile("m5")[0]
    assert (m5.h1_trend_ema_period, m5.min_reward_frac, m5.risk.max_trades_per_day) == (50, 0.3, 3)


def test_claude_ftmo_has_the_h1_trend_and_m5_minimum_reward() -> None:
    ftmo = build_profile("m1", plan=True)[0]
    assert ftmo.h1_trend_ema_period == 50
    assert ftmo.min_reward_frac == 0.3 and ftmo.min_reward_frac_m1 is None
    assert ftmo.risk.max_trades_per_day is None
    m1 = build_profile("m1")[0]
    assert m1.h1_trend_ema_period is None and m1.min_reward_frac is None


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


def test_claude_ftmo_trend_only_checks_m5_wm_zones(bot, mocker) -> None:  # noqa: F811
    b, _ = bot
    place = _ready(b, mocker, tp=1905.0, frac=None)
    b.config = replace(b.config, ohlc_timeframe="M1")
    trend = mocker.patch("bot.main.h1_ema_trend", return_value="DOWN")
    zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
    df = make_ohlc(n=30)
    # An M1 zone: no H1 trend check.
    assert b._try_place_setup(zone, df, uuid4(), bid=1899.9, ask=1900.0) is True
    trend.assert_not_called()
    # An M5 zone against the trend: skipped.
    assert b._try_place_setup(zone, df, uuid4(), bid=1899.9, ask=1900.0, timeframe="M5") is False
    # An M5 FVG: its own rules, no trend check.
    trend.reset_mock()
    mocker.patch.object(b, "_fvg_levels_for",
                        return_value=TradeLevels(entry=1900.0, sl=1893.5, tp1=1905.0, sl_method="FVG"))
    assert b._try_place_setup(zone, df, uuid4(), bid=1899.9, ask=1900.0, timeframe="M5") is True
    trend.assert_not_called()
    assert place.call_count == 2


def test_minimum_reward_is_per_timeframe(bot, mocker) -> None:  # noqa: F811
    b, _ = bot
    place = _ready(b, mocker, tp=1901.0, period=None)     # 1.0 for 6.5 = 0.15
    b.config = replace(b.config, ohlc_timeframe="M1")
    zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
    df = make_ohlc(n=30)
    # M1 zone, no M1 rule: trades.
    assert b._try_place_setup(zone, df, uuid4(), bid=1899.9, ask=1900.0) is True
    # M5 zone: 0.15 < 0.3, skipped.
    assert b._try_place_setup(zone, df, uuid4(), bid=1899.9, ask=1900.0, timeframe="M5") is False
    # With an M1 rule of 0.1 the M1 zone still trades (0.15), at 0.2 it doesn't.
    b.config = replace(b.config, min_reward_frac_m1=0.1)
    assert b._try_place_setup(zone, df, uuid4(), bid=1899.9, ask=1900.0) is True
    b.config = replace(b.config, min_reward_frac_m1=0.2)
    assert b._try_place_setup(zone, df, uuid4(), bid=1899.9, ask=1900.0) is False
    assert place.call_count == 2


def test_h1_summary_for_the_dashboard() -> None:
    from bot.strategy.h1_trend import h1_ema_summary
    n = 12 * 80
    up = h1_ema_summary(_m5(np.linspace(3500, 3700, n)))
    assert up["state"] == "UPTREND" and up["timeframe"] == "H1"
    assert up["down_below"] == up["ema"] and up["up_above"] is None and up["ema"] < 3700
    rows = list(np.linspace(3500, 3700, 12 * 79)) + list(np.linspace(3690, 3300, 12))
    down = h1_ema_summary(_m5(rows))
    assert down["state"] == "DOWNTREND" and down["up_above"] == down["ema"]
    assert pd.Timestamp(down["since"]) == _m5(rows).index[-1] + pd.Timedelta(minutes=5)
    assert h1_ema_summary(_m5(np.linspace(3500, 3700, 12 * 20)))["state"] == "RANGE"


def test_dashboard_badge_is_the_h1_trend_when_on(bot, mocker) -> None:  # noqa: F811
    b, _ = bot
    b._latest_ohlc = _m5(np.linspace(3500, 3700, 12 * 80))
    b.config = replace(b.config, h1_trend_ema_period=50)
    t = b._dashboard_trend()
    assert (t["state"], t["timeframe"], t["filter"]) == ("UPTREND", "H1", "h1")
    b.config = replace(b.config, h1_trend_ema_period=None)
    b._trend_cache = None
    assert b._dashboard_trend()["filter"] != "h1"


def test_dashboard_next_zones_are_only_ones_it_would_take(bot, mocker) -> None:  # noqa: F811
    """25 Nov 2026: the card only shows zones the bot would take now."""
    import time
    b, _ = bot
    b._latest_ohlc = _m5(np.linspace(3500, 3700, 12 * 80))          # H1 uptrend
    b.config = replace(b.config, h1_trend_ema_period=50, min_reward_frac=0.3)
    buy = make_zone_for_test(direction="BUY", top=3690.0, bottom=3685.0)
    sell = make_zone_for_test(direction="SELL", top=3715.0, bottom=3710.0)
    low = make_zone_for_test(direction="BUY", top=3680.0, bottom=3675.0)
    ids = {id(buy): uuid4(), id(sell): uuid4(), id(low): uuid4()}
    b._pending_pipeline_zones = [(z, ids[id(z)]) for z in (buy, sell, low)]
    now = time.monotonic()
    b._deferred_seen = {(z.direction, round(float(z.top), 2), round(float(z.bottom), 2)): now
                        for z in (buy, sell, low)}

    def lv(zone, df, cfg):
        e = float(zone.top) if zone.direction == "BUY" else float(zone.bottom)
        s = -1 if zone.direction == "BUY" else 1
        tp = e - s * (0.5 if zone is low else 5.0)                   # low: 0.5 for 6.5
        return TradeLevels(entry=e, sl=e + s * 6.5, tp1=tp, sl_method="PCT_30")
    mocker.patch("bot.main.compute_levels", side_effect=lv)
    views = b._waiting_zone_views(bid=3695.0, ask=3695.3)
    assert [(v.direction, v.top) for v in views] == [("BUY", 3690.0)]   # sell: against H1; low: tiny TP
    assert views[0].tf == "M5" and views[0].as_dict()["tf"] == "M5"
    # Rules off: everything shows, as before.
    b.config = replace(b.config, h1_trend_ema_period=None, min_reward_frac=None)
    assert len(b._waiting_zone_views(bid=3695.0, ask=3695.3)) == 3
