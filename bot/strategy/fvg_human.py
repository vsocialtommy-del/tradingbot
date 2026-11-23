"""Human FVG setups (Tommy, 23 Nov 2026), for Claude FTMO.

How an FVG trader takes it: one trade per shift in structure, not every
gap in every move. Replaces the spike / V FVG (:mod:`bot.strategy.fvg_v`,
now kept as the "FVG spam bot", ``PlanConfig.fvg_mode = "spam"``).

A BUY (a SELL is the mirror):

1. **Shift (MSS)**: a candle closes above the last swing high, and that
   swing high is LOWER than the swing high before it (the trend was down
   and this is the first break against it). Swings are on the wicks, 2
   candles either side.
2. **The low**: the lowest low between that swing high and the BOS candle.
3. **Sweep**: that low took out obvious liquidity:
   * the previous broker day's low (not taken earlier today), or
   * the Asia low (broker 01:00-09:00, i.e. UK 23:00-07:00), when the
     low is after Asia and Asia's low wasn't taken since, or
   * an untaken swing low (10 candles either side) from the last 24 hours.
4. **FVG**: the first 3-candle gap in the move up from the low.
5. **Spike / V only**: the low isn't in a base (W / M or any other); the
   zone rules handle those.

* **Entry**: the middle of the gap (``entry="edge"``: the near edge). The
  zone the bot waits on is the half of the gap from the middle to the far
  edge, so price has to reach the middle to trade.
* **SL**: 0.5 past the sweep low.
* **TP**: the swing high before the broken one (the next liquidity), 0.3
  early. Skipped if it isn't past the entry.
* **Gone** after the first touch, if price reaches the TP first, or 4
  hours after the BOS without a touch.

Backtested 14 Nov 2025 - 13 Nov 2026 on Claude FTMO, £50 a trade, W / M
zones as live: 1,380 trades +£6,741, max drawdown -£519, 11 red weeks of
53 (the human FVG trades alone: 172, +£1,291). Edge entry: +£5,946.

Times are the candle times MT5 gives (broker server time), as in the
backtest data. The scan is stateless: the same candles always give the
same setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from bot.strategy.fvg_v import FvgSetup, _swings, based_candles
from bot.strategy.vertex import VertexConfig


@dataclass(frozen=True)
class HumanFvgConfig:
    swing_strength: int = 2
    liquidity_strength: int = 10
    """Candles either side of an untaken swing that counts as liquidity."""
    liquidity_lookback_minutes: int = 1440
    asia_start_hour: int = 1
    asia_end_hour: int = 9
    """Broker hours (UK +2): 01:00-09:00 = UK 23:00-07:00."""
    sl_buffer: float = 0.5
    tp_buffer: float = 0.3
    wait_minutes: int = 240
    """Dropped if not touched within this long of the BOS."""
    entry: Literal["mid", "edge"] = "mid"
    window_bars: int = 4000
    """Closed candles scanned each time."""


def _bar_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 1
    d = pd.Series(index[1:] - index[:-1]).median()
    return max(1, round(d / pd.Timedelta(minutes=1)))


def find_human_setups(
    closed: pd.DataFrame, cfg: HumanFvgConfig, based: set[tuple[str, int]],
) -> list[FvgSetup]:
    """Human FVG setups that are live now: everything above has happened
    in ``closed`` (closed candles only, oldest first), no closed candle
    has touched the entry or reached the TP since, and the next candle
    (the one forming) is where the first touch can happen.

    ``fvg_top`` / ``fvg_bottom`` of each setup are the zone the bot waits
    on: for a BUY from the entry down to the bottom of the gap, for a SELL
    from the entry up to the top. ``based``: see
    :func:`bot.strategy.fvg_v.based_candles`, on the same frame."""
    H = closed["high"].to_numpy(float)
    L = closed["low"].to_numpy(float)
    C = closed["close"].to_numpy(float)
    n = len(H)
    s = cfg.swing_strength
    if n < 2 * s + 4:
        return []
    ix = pd.DatetimeIndex(closed.index)
    bar = _bar_minutes(ix)
    wait = max(1, cfg.wait_minutes // bar)
    lookback = max(1, cfg.liquidity_lookback_minutes // bar)
    s10 = cfg.liquidity_strength
    sw10_lo = np.nonzero(_swings(L, False, s10))[0]
    sw10_hi = np.nonzero(_swings(H, True, s10))[0]

    date = np.array(ix.date)
    hour = ix.hour.to_numpy()
    day_lo = pd.Series(L).groupby(date).min()
    day_hi = pd.Series(H).groupby(date).max()
    days = list(day_lo.index)
    prev_day = {d: days[k - 1] for k, d in enumerate(days) if k}
    asia = (hour >= cfg.asia_start_hour) & (hour < cfg.asia_end_hour)
    asia_lo = pd.Series(np.where(asia, L, np.inf)).groupby(date).min()
    asia_hi = pd.Series(np.where(asia, H, -np.inf)).groupby(date).max()
    day_start: dict = {}
    for k in range(n):
        day_start.setdefault(date[k], k)

    def untaken(buy: bool, lvl: float, seg: np.ndarray) -> bool:
        return len(seg) == 0 or (seg.min() >= lvl if buy else seg.max() <= lvl)

    def swept(buy: bool, lo: int) -> bool:
        x = L[lo] if buy else H[lo]
        v = L if buy else H
        d = date[lo]
        d0 = day_start[d]
        pd_ = prev_day.get(d)
        if pd_ is not None:
            lvl = day_lo[pd_] if buy else day_hi[pd_]
            if ((buy and x < lvl) or (not buy and x > lvl)) and untaken(buy, lvl, v[d0:lo]):
                return True
        if hour[lo] >= cfg.asia_end_hour:
            lvl = asia_lo[d] if buy else asia_hi[d]
            if np.isfinite(lvl) and ((buy and x < lvl) or (not buy and x > lvl)):
                after = [k for k in range(d0, lo) if hour[k] >= cfg.asia_end_hour]
                if untaken(buy, lvl, v[after[0]:lo] if after else v[:0]):
                    return True
        cand = sw10_lo if buy else sw10_hi
        a, b = cand.searchsorted(lo - lookback), cand.searchsorted(lo - s10)
        for k in cand[a:b]:
            lvl = v[k]
            if ((buy and x < lvl) or (not buy and x > lvl)) and untaken(buy, lvl, v[k + 1:lo]):
                return True
        return False

    out: list[FvgSetup] = []
    for buy in (True, False):
        sw = _swings(H if buy else L, buy, s)
        last: int | None = None
        prevk: int | None = None
        broken = True
        for j in range(n):
            k = j - s - 1
            if k >= 0 and sw[k]:
                prevk, last, broken = last, k, False
            if last is None or broken:
                continue
            lvl = H[last] if buy else L[last]
            if not ((buy and C[j] > lvl) or (not buy and C[j] < lvl)):
                continue
            broken = True
            if prevk is None:
                continue
            # 1. a change of structure: a LOWER high broken (BUY) / HIGHER low (SELL)
            if (buy and not H[last] < H[prevk]) or (not buy and not L[last] > L[prevk]):
                continue
            seg = range(last, j + 1)
            lo = min(seg, key=lambda x: L[x]) if buy else max(seg, key=lambda x: H[x])
            if not swept(buy, lo):
                continue
            got = None
            for i in range(lo, min(j + 1, n - 2)):
                if buy and H[i] < L[i + 2]:
                    got = (i, H[i], L[i + 2])
                    break
                if not buy and L[i] > H[i + 2]:
                    got = (i, H[i + 2], L[i])
                    break
            if got is None:
                continue
            i, gap_bottom, gap_top = got
            if (("BUY" if buy else "SELL"), lo) in based:
                continue
            ext = L[lo] if buy else H[lo]
            if buy:
                sl, tp = ext - cfg.sl_buffer, H[prevk] - cfg.tp_buffer
            else:
                sl, tp = ext + cfg.sl_buffer, L[prevk] + cfg.tp_buffer
            if cfg.entry == "mid":
                entry = round((gap_top + gap_bottom) / 2, 2)
            else:
                entry = gap_top if buy else gap_bottom
            if (buy and tp <= entry) or (not buy and tp >= entry):
                continue
            avail = max(j, i + 2) + 1          # first candle it can trade on
            if avail > n or n >= avail + wait:
                continue
            dead = False
            for t in range(i + 3, n):
                if (buy and L[t] <= entry) or (not buy and H[t] >= entry):
                    dead = True                # touched (or touched too early)
                    break
                if t >= avail and ((buy and H[t] >= tp) or (not buy and L[t] <= tp)):
                    dead = True                # went to the TP without us
                    break
            if dead:
                continue
            top, bottom = (entry, gap_bottom) if buy else (gap_top, entry)
            out.append(FvgSetup(
                direction="BUY" if buy else "SELL",
                fvg_top=round(float(top), 2), fvg_bottom=round(float(bottom), 2),
                entry=round(float(entry), 2), sl=round(float(sl), 2),
                tp=round(float(tp), 2), bos_level=round(float(lvl), 2),
                formed_at=pd.Timestamp(closed.index[avail - 1]),
            ))
    return out


def scan(df: pd.DataFrame, cfg: HumanFvgConfig, vcfg: VertexConfig) -> list[FvgSetup]:
    """Live setups from a bot frame whose LAST row is the candle still
    forming (it's dropped; only closed candles are read)."""
    closed = df.iloc[:-1].iloc[-cfg.window_bars:]
    if len(closed) == 0:
        return []
    return find_human_setups(closed, cfg, based_candles(closed, vcfg))
