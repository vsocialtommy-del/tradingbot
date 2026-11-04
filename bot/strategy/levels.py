"""Trade levels for the XAUTISM method: entry, stop loss and TP1.

Entry
    The zone edge price reaches first: zone top for a BUY, zone bottom
    for a SELL.

Stop loss
    Default: 30% of the zone's size past the far edge
    (BUY: ``bottom - 0.3 * size``; SELL: ``top + 0.3 * size``).
    If a recent swing low (BUY) / swing high (SELL) on wicks sits just
    beyond the zone, inside ``swing_window_pct`` of the zone size past the
    30% level, the stop moves to ``swing_buffer_points`` beyond that swing
    (1.0 point = 10 pips) so a sweep of the obvious low/high doesn't take
    it out. The stop is never moved closer than the 30% level.

TP1
    The first break of structure the move out of the base made
    (``ValidatedZone.broken_swing``), set ``tp_spread_buffer`` early so the
    spread can't cause a near miss (buys close on the bid, sells on the
    ask). Falls back to ``None`` when the zone carries no BOS swing; the
    caller decides what to do then.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from bot.strategy.strong_point import ValidatedZone


SlMethod = Literal["PCT_30", "SWING"]


@dataclass(frozen=True)
class LevelsConfig:
    enabled: bool = True
    sl_zone_pct: float = 0.30
    swing_buffer_points: float = 1.0
    """How far past the swing low/high the stop goes (1.0 = 10 pips)."""
    swing_window_pct: float = 0.5
    """How far past the 30% stop (as a share of zone size) a swing may sit
    and still pull the stop out beyond it."""
    swing_lookback_bars: int = 50
    swing_strength: int = 2
    tp_spread_buffer: float = 0.3
    swing_adjust: bool = True
    """Push the SL beyond a nearby swing (+1 point). False = SL exactly
    ``sl_zone_pct`` past the zone (0 = right at the zone edge)."""


@dataclass(frozen=True)
class TradeLevels:
    entry: float
    sl: float
    tp1: float | None
    sl_method: SlMethod
    swing_price: float | None = None


def compute_levels(
    zone: ValidatedZone, df: pd.DataFrame, config: LevelsConfig | None = None,
) -> TradeLevels:
    cfg = config or LevelsConfig()
    top, bottom = float(zone.top), float(zone.bottom)
    size = top - bottom
    buy = zone.direction == "BUY"

    entry = top if buy else bottom
    sl = bottom - cfg.sl_zone_pct * size if buy else top + cfg.sl_zone_pct * size
    method: SlMethod = "PCT_30"
    swing_price: float | None = None

    swing = _swing_beyond_zone(zone, df, cfg, sl30=sl) if cfg.swing_adjust else None
    if swing is not None:
        cand = swing - cfg.swing_buffer_points if buy else swing + cfg.swing_buffer_points
        if (buy and cand < sl) or (not buy and cand > sl):
            sl, method, swing_price = cand, "SWING", swing

    tp1: float | None = None
    if zone.broken_swing is not None:
        lvl = float(zone.broken_swing.price)
        tp1 = lvl - cfg.tp_spread_buffer if buy else lvl + cfg.tp_spread_buffer
        if (buy and tp1 <= entry) or (not buy and tp1 >= entry):
            tp1 = None

    return TradeLevels(
        entry=round(entry, 2), sl=round(sl, 2),
        tp1=None if tp1 is None else round(tp1, 2),
        sl_method=method, swing_price=swing_price,
    )


def _swing_beyond_zone(
    zone: ValidatedZone, df: pd.DataFrame, cfg: LevelsConfig, *, sl30: float,
) -> float | None:
    """Most extreme wick swing just beyond the zone, before the base.

    BUY: swing lows (on lows) below ``zone.bottom`` and no lower than
    ``sl30 - swing_window_pct * size``. SELL mirror on highs.
    """
    try:
        base_start = zone.source_pattern.base.start_index
    except AttributeError:
        return None
    lo = max(0, base_start - cfg.swing_lookback_bars)
    if base_start - lo < 2 * cfg.swing_strength + 1:
        return None
    size = float(zone.top) - float(zone.bottom)
    k = cfg.swing_strength
    buy = zone.direction == "BUY"
    series = (df["low"] if buy else df["high"]).to_numpy()[lo:base_start + k + 1]
    found: list[float] = []
    for i in range(k, len(series) - k):
        if lo + i >= base_start:
            break
        win = series[i - k:i + k + 1]
        v = series[i]
        if buy and v == win.min() and v < zone.bottom and v >= sl30 - cfg.swing_window_pct * size:
            found.append(float(v))
        if not buy and v == win.max() and v > zone.top and v <= sl30 + cfg.swing_window_pct * size:
            found.append(float(v))
    if not found:
        return None
    return float(np.min(found)) if buy else float(np.max(found))
