"""H1 trend filter for M5 W / M zones (Tommy, 24 Nov 2026).

An M5 zone only trades WITH the H1 trend:

* H1 candles are built from the bot's closed M5 candles (whole hours).
* The trend is the last CLOSED H1 candle's close against the 50 EMA of
  H1 closes: above = UP (buys only), below = DOWN (sells only).
* Read fresh every time a zone is about to trade, so a zone blocked now
  can still trade later if the trend turns its way.

Backtested 14 Nov 2025 - 13 Nov 2026 on the M5 bot (£50 a trade, with the
3 trades a day cap and the 0.3 minimum reward): 205 trades, +£737, max
drawdown -£448, vs live then 862 trades, +£358, -£871. Against the H1
trend the M5 bot's trades lost money (442 trades, -£329); with it they
made +£688.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from bot.strategy.direction_gate import source_step

H1Trend = Literal["UP", "DOWN"]


def h1_ema_trend(closed: pd.DataFrame, period: int = 50) -> H1Trend | None:
    """``closed``: the bot's closed M5 (or M1) candles, oldest first.
    Returns None if there isn't enough history to read a trend."""
    if len(closed) < 2:
        return None
    step = source_step(closed)
    h1 = closed["close"].astype(float).resample("1h").last().dropna()
    last_end = closed.index[-1] + step
    h1 = h1[h1.index + pd.Timedelta(hours=1) <= last_end]   # closed H1 candles only
    if len(h1) < period:
        return None
    ema = h1.ewm(span=period, adjust=False).mean()
    return "UP" if float(h1.iloc[-1]) > float(ema.iloc[-1]) else "DOWN"


def allowed(direction: str, trend: H1Trend | None) -> bool:
    """BUY needs an UP trend, SELL a DOWN trend. No trend yet = allowed."""
    if trend is None:
        return True
    return (direction == "BUY") == (trend == "UP")


def h1_ema_summary(closed: pd.DataFrame, period: int = 50) -> dict:
    """The H1 trend for the dashboard: state, when it last flipped (the H1
    close that did it, candle time), the 50 EMA now, and the H1 close that
    would flip it (the next H1 close above / below today's EMA value)."""
    out: dict = {"timeframe": "H1", "rule": f"ema{period}", "state": "RANGE", "since": None,
                 "up_above": None, "down_below": None, "ema": None}
    if len(closed) < 2:
        return out
    step = source_step(closed)
    h1 = closed["close"].astype(float).resample("1h").last().dropna()
    h1 = h1[h1.index + pd.Timedelta(hours=1) <= closed.index[-1] + step]
    if len(h1) < period:
        return out
    ema = h1.ewm(span=period, adjust=False).mean()
    up = (h1 > ema).to_numpy()
    flips = (up[1:] != up[:-1]).nonzero()[0]
    e = round(float(ema.iloc[-1]), 2)
    out.update(
        state="UPTREND" if up[-1] else "DOWNTREND",
        since=None if len(flips) == 0
        else (h1.index[flips[-1] + 1] + pd.Timedelta(hours=1)).isoformat(),
        ema=e,
        # The next H1 close flips it when it closes past today's EMA value.
        up_above=None if up[-1] else e,
        down_below=e if up[-1] else None,
    )
    return out
