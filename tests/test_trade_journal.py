"""Trade photo journal (Tommy, 26 Nov 2026)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot.journal.journal import TradeJournal, h1_trend_at, layers_from_rows
from bot.journal.trade_chart import ChartTrade, Layer, classify_exit

OFFSET = timedelta(hours=2)                       # broker clock = UTC + 2
T0 = datetime(2026, 11, 26, 7, 34, 49, tzinfo=timezone.utc)


def _rows(key="124c34a8"):
    def r(pid, vol, o, op, c, cp, pnl):
        return {"position_id": pid, "setup_key": key, "direction": "SELL", "volume": vol,
                "open_time": (T0 + timedelta(seconds=o)).isoformat(), "open_price": op,
                "close_time": (T0 + timedelta(minutes=c)).isoformat(), "close_price": cp,
                "profit": pnl}
    return [r(3, 0.09, 9, 3572.19, 6, 3572.34, -1.04), r(1, 0.01, 0, 3572.12, 5, 3571.94, 0.14),
            r(2, 0.09, 9, 3572.19, 5, 3572.00, 1.29)]


def _candles(tf_min: int, end: datetime, n: int, start_price=3600.0, slope=-0.02):
    """Broker-time stamped candles, like MT5 gives them."""
    idx = pd.date_range(end=pd.Timestamp(end) + OFFSET, periods=n, freq=f"{tf_min}min")
    rng = np.random.default_rng(1)
    c = start_price + np.cumsum(slope * tf_min + rng.normal(0, 0.3, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.3, "low": np.minimum(o, c) - 0.3,
                         "close": c}, index=idx)


class FakeSB:
    def __init__(self, missing=False):
        self.missing = missing
        self.files: dict[str, bytes] = {}
        self.rows: list[dict] = []

    def trade_chart_keys(self, bot, account, version=0):
        if self.missing:
            raise RuntimeError('relation "trade_charts" does not exist')
        return [r["setup_key"] for r in self.rows if int(r["data"].get("v") or 1) >= version]

    def find_setup_by_key(self, key, near=None):
        return {"zone_id": "z1", "planned_layer1_price": 3572.19, "planned_sl_price": 3575.68,
                "planned_tp1_price": 3572.20, "created_at": T0.isoformat()}

    def get_zone_row(self, zid):
        # Like the real zones table: a candle time in the broker's clock (UTC+2).
        return {"top": 3573.80, "bottom": 3572.19,
                "formed_at": (T0 - timedelta(minutes=20) + OFFSET).isoformat(),
                "last_evaluation_result": {"timeframe": "M1"}}

    def upload_file(self, bucket, path, data, ctype):
        self.files[path] = data

    def upsert_rows(self, table, rows, *, on_conflict):
        assert table == "trade_charts"
        self.rows.extend(rows)


def _journal(sb, since=datetime(2026, 11, 24, 21, 45, tzinfo=timezone.utc)):
    return TradeJournal(sb, bot="M1", bot_label="CLAUDE FTMO", symbol="XAUUSD",
                        since=since, default_tf="M1")


def _get_ohlc(now):
    def get(tf, n):
        return _candles(1 if tf == "M1" else 5, now, min(n, 3000 if tf == "M1" else 2000))
    return get


def _run(j, now):
    j.step(now, _get_ohlc(now), OFFSET)
    if j._job is not None:
        j._job[1].result(timeout=60)
    j.step(now, _get_ohlc(now), OFFSET)


def test_layers_named_oldest_first_last_is_runner() -> None:
    ls = layers_from_rows(_rows())
    assert [x.name for x in ls] == ["L1", "L2", "L3 runner"]
    assert ls[0].volume == 0.01 and ls[2].profit == -1.04


def test_h1_trend_rising_is_up() -> None:
    m5 = _candles(5, T0, 2000, 3500, slope=+0.02)
    m5.index = m5.index - OFFSET
    state, close, ema, series = h1_trend_at(m5, T0)
    assert state == "UPTREND" and close > ema and len(series) > 50


def test_waits_30_minutes_after_the_close() -> None:
    j = _journal(FakeSB())
    j.note_rows(_rows(), 540334604)
    j._done = set()
    assert j.due(T0 + timedelta(minutes=20)) == []
    assert j.due(T0 + timedelta(minutes=40)) == ["124c34a8"]


def test_trades_before_the_start_date_are_left_out() -> None:
    j = _journal(FakeSB(), since=T0 + timedelta(hours=1))
    j.note_rows(_rows(), 540334604)
    j._done = set()
    assert j.due(T0 + timedelta(hours=3)) == []


def test_picture_and_numbers_saved_once() -> None:
    sb = FakeSB()
    j = _journal(sb)
    j.note_rows(_rows(), 540334604)
    now = T0 + timedelta(minutes=45)
    _run(j, now)
    assert len(sb.files) == 1
    path, png = next(iter(sb.files.items()))
    assert path == "M1/540334604/2026-11-26/124c34a8.png" and png[:4] == b"\x89PNG"
    row = sb.rows[0]
    assert (row["account"], row["setup_key"]) == (540334604, "124c34a8")
    d = row["data"]
    assert (d["direction"], d["timeframe"], d["kind"]) == ("SELL", "M1", "W/M")
    assert d["h1_trend"] in ("UPTREND", "DOWNTREND")
    assert d["reward_pts"] == -0.01 and d["profit"] == 0.39 and d["result"] == "WIN"
    assert [x["name"] for x in d["layers"]] == ["L1", "L2", "L3 runner"]
    # 27 Nov 2026 bug: the zone time is broker clock, the chart needs real UTC.
    assert d["zone_formed"] == (T0 - timedelta(minutes=20)).isoformat() and d["v"] == 2
    # Already done: not drawn again.
    _run(j, now + timedelta(minutes=5))
    assert len(sb.files) == 1


def test_missing_table_switches_it_off_quietly(caplog) -> None:
    sb = FakeSB(missing=True)
    j = _journal(sb)
    j.note_rows(_rows(), 540334604)
    j.step(T0 + timedelta(minutes=45), _get_ohlc(T0), OFFSET)
    assert sb.files == {} and j._job is None
    j.step(T0 + timedelta(minutes=46), _get_ohlc(T0), OFFSET)   # off for a while, no retry spam
    assert j._warned


def test_exit_labels() -> None:
    t = ChartTrade("X", 1, "k", "BUY", "M5", "W/M", [], entry=100.0, sl=95.0, tp1=105.0)
    lay = lambda c: Layer("L2", 0.1, T0, 100.0, T0, c, 0.0)
    assert classify_exit(t, lay(95.1)) == "SL"
    assert classify_exit(t, lay(105.2)) == "TP1"
    assert classify_exit(t, lay(100.1)) == "breakeven"
    assert classify_exit(t, lay(108.0)).startswith("trail (+3.0")


@pytest.mark.parametrize("profile, label, since", [
    (("m5",), "M5 BOT", datetime(2026, 11, 26, 18, 0, tzinfo=timezone.utc)),
])
def test_profiles(profile, label, since) -> None:
    from bot.profiles import build_profile
    cfg = build_profile(*profile)[0]
    assert (cfg.journal_label, cfg.journal_since) == (label, since)
    ftmo = build_profile("m1", plan=True)[0]
    assert ftmo.journal_label == "CLAUDE FTMO"
    assert ftmo.journal_since == datetime(2026, 11, 24, 21, 45, tzinfo=timezone.utc)
    assert build_profile("m1")[0].journal_since is None


def test_old_pictures_are_redrawn() -> None:
    sb = FakeSB()
    sb.rows.append({"setup_key": "124c34a8", "data": {}})          # v1 picture
    j = _journal(sb)
    j.note_rows(_rows(), 540334604)
    _run(j, T0 + timedelta(minutes=45))
    assert len(sb.files) == 1 and sb.rows[-1]["data"]["v"] == 2
