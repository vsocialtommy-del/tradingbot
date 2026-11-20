"""Spike / V BOS + FVG setups (Tommy, 20 Nov 2026), for Claude FTMO.

Traded alongside the real M / W zones. Only when the move came from a
spike / V, never from a base: if a W / M (or any base) shows, the normal
zone rules handle it and this doesn't trade.

* **BOS**: a candle closes beyond the last swing high (BUY) / swing low
  (SELL). Swings are on the wicks, 2 candles either side.
* **The low / high**: the lowest low (BUY) / highest high (SELL) between
  that swing and the BOS candle.
* **Spike / V only**: that low / high candle must not sit inside a vertex
  base of any shape (W / M or not, leg-in size ignored).
* **FVG**: the first 3-candle gap in the move up from the low (BUY:
  candle 1 high below candle 3 low) / down from the high (SELL).
* **Entry**: the first touch of the FVG after the BOS has closed (BUY at
  the FVG top, SELL at the FVG bottom). Touched before that, or touched
  once already: no trade. Gone after 2,000 candles without a touch.
* **TP**: the BOS line, 0.3 early. Skipped if the BOS line isn't past the
  entry.
* **SL**: M5: 0.5 past the low / high. M1: 0.5 past candle 1 of the FVG.

Backtested 14 Aug - 13 Nov 2026 on Claude FTMO (3 months, £50 a trade):
2,481 trades +£5,648, max drawdown -£763 (Claude FTMO alone: 343 trades
+£1,513, drawdown -£265). That test used the earlier wick W / M rule.

The scan is stateless: the same candles always give the same setups, so
each new candle simply re-scans the recent window.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from bot.strategy.vertex import VertexConfig, detect_vertex_patterns

SlMode = Literal["low", "candle1"]


@dataclass(frozen=True)
class FvgVConfig:
    sl_mode: SlMode = "low"
    """"low": SL past the low / high of the move (M5). "candle1": SL past
    candle 1 of the FVG (M1)."""
    swing_strength: int = 2
    sl_buffer: float = 0.5
    tp_buffer: float = 0.3
    max_wait_bars: int = 2000
    """An FVG not touched within this many candles of the BOS is dropped."""
    window_bars: int = 4000
    """Closed candles scanned each time."""


@dataclass(frozen=True)
class FvgSetup:
    direction: str
    fvg_top: float
    fvg_bottom: float
    entry: float
    sl: float
    tp: float
    bos_level: float
    formed_at: pd.Timestamp
    """Open time of the last candle needed (the BOS candle or FVG candle 3)."""


def base_vertex_config(vcfg: VertexConfig) -> VertexConfig:
    """Bases of any shape, leg-in size ignored: used to tell a spike / V
    from a base."""
    return replace(vcfg, require_mw=False, leg_in_min_frac=0.0)


def based_candles(df: pd.DataFrame, vcfg: VertexConfig) -> set[tuple[str, int]]:
    """(direction, candle index) for every candle inside a vertex base."""
    out: set[tuple[str, int]] = set()
    for p in detect_vertex_patterns(df, base_vertex_config(vcfg)):
        for x in range(p.base.start_index, p.base.end_index + 1):
            out.add((p.direction, x))
    return out


def _swings(v: np.ndarray, hi: bool, s: int) -> np.ndarray:
    n = len(v)
    out = np.zeros(n, bool)
    for k in range(s, n - s):
        w = v[k - s:k + s + 1]
        if hi:
            out[k] = v[k] == w.max() and int(np.argmax(w)) == s
        else:
            out[k] = v[k] == w.min() and int(np.argmin(w)) == s
    return out


def find_fvg_setups(
    closed: pd.DataFrame, cfg: FvgVConfig, based: set[tuple[str, int]],
) -> list[FvgSetup]:
    """Spike / V FVG setups that are live now: the BOS has closed, the FVG
    hasn't been touched by any closed candle since it formed, and the
    next candle (the one forming) is where the first touch can happen.

    ``closed``: closed candles only (oldest first). ``based``: see
    :func:`based_candles`, computed on the same frame."""
    H = closed["high"].to_numpy(float)
    L = closed["low"].to_numpy(float)
    C = closed["close"].to_numpy(float)
    n = len(H)
    s = cfg.swing_strength
    if n < 2 * s + 4:
        return []
    out: list[FvgSetup] = []
    for buy in (True, False):
        sw = _swings(H if buy else L, buy, s)
        last: int | None = None
        broken = True
        for j in range(n):
            k = j - s - 1
            if k >= 0 and sw[k]:
                last, broken = k, False
            if last is None or broken:
                continue
            lvl = H[last] if buy else L[last]
            if not ((buy and C[j] > lvl) or (not buy and C[j] < lvl)):
                continue
            broken = True
            seg = range(last, j + 1)
            lo = min(seg, key=lambda x: L[x]) if buy else max(seg, key=lambda x: H[x])
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
            i, fb, ft = got
            if (buy and lvl - cfg.tp_buffer <= ft) or (not buy and lvl + cfg.tp_buffer >= fb):
                continue
            avail = max(j, i + 2) + 1          # first candle it can trade on
            if avail > n or n >= avail + cfg.max_wait_bars:
                continue
            touched = any(
                (buy and L[t] <= ft) or (not buy and H[t] >= fb)
                for t in range(i + 3, n)
            )
            if touched:
                continue
            if (("BUY" if buy else "SELL"), lo) in based:
                continue
            if cfg.sl_mode == "low":
                ext = L[lo] if buy else H[lo]
            else:
                ext = L[i] if buy else H[i]
            if buy:
                entry, sl, tp = ft, ext - cfg.sl_buffer, lvl - cfg.tp_buffer
            else:
                entry, sl, tp = fb, ext + cfg.sl_buffer, lvl + cfg.tp_buffer
            out.append(FvgSetup(
                direction="BUY" if buy else "SELL",
                fvg_top=round(float(ft), 2), fvg_bottom=round(float(fb), 2),
                entry=round(float(entry), 2), sl=round(float(sl), 2),
                tp=round(float(tp), 2), bos_level=round(float(lvl), 2),
                formed_at=pd.Timestamp(closed.index[avail - 1]),
            ))
    return out


def scan(df: pd.DataFrame, cfg: FvgVConfig, vcfg: VertexConfig) -> list[FvgSetup]:
    """Live setups from a bot frame whose LAST row is the candle still
    forming (it's dropped; only closed candles are read)."""
    closed = df.iloc[:-1].iloc[-cfg.window_bars:]
    if len(closed) == 0:
        return []
    return find_fvg_setups(closed, cfg, based_candles(closed, vcfg))
