"""TP1 target — nearest local peak / low to the entry price.

The loosened entry flow (May 2026 refinement) replaces the previous
BOS_LEVEL / FIXED_DISTANCE TP1 sources with a recent-structure search:

* BUY:  TP1 = lowest-priced **local high** above ``entry_price``
  within the last ``lookback_bars``.
* SELL: TP1 = highest-priced **local low** below ``entry_price``.

"Local high" = a bar whose ``high`` is strictly greater than both
neighbours' ``high`` (1-bar swing on highs). Local low: mirror on
``low``. The last bar in the df can never qualify — it has no right
shoulder yet.

If no qualifying peak exists in the lookback window the function
returns ``None`` and the caller (``main._try_place_setup``) skips
the zone. This is the "no TP1 target → no trade" gate.

Why pure-logic + caller-decides
-------------------------------
Computing TP1 in the strategy layer (not inside ``order_manager``)
lets us reject "no peak available" zones before any DB write or
broker call — same shape as the SL-distance gate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from bot.strategy.zone_marking import Direction


def find_nearest_local_peak(
    df: pd.DataFrame,
    entry_price: float,
    direction: Direction,
    *,
    lookback_bars: int = 50,
) -> float | None:
    """Return the nearest local peak/low price, or ``None`` if none found.

    Parameters
    ----------
    df
        OHLC DataFrame (needs ``high`` and ``low`` columns). The last
        row is the most recent closed bar; we scan backwards from
        ``len(df) - 2`` for ``lookback_bars`` candidate bars.
    entry_price
        The trade's reference entry. For BUY this is Layer 1's
        planned price (= zone.top); for SELL it's zone.bottom.
    direction
        ``BUY`` → look up for a local high above entry; ``SELL`` →
        look down for a local low below entry.
    lookback_bars
        How far back to search. Older peaks are out of scope; the
        idea is "recent structure that price might revisit", not
        ancient resistance.

    Returns
    -------
    float | None
        The price of the nearest peak/low (by **price**, not by time
        — i.e. the lowest qualifying high above entry for BUY, the
        highest qualifying low below entry for SELL). ``None`` when
        no bar in the lookback window is both a local extreme AND
        the right side of the entry.
    """
    if lookback_bars < 1:
        raise ValueError(f"lookback_bars must be >= 1, got {lookback_bars}")
    for col in ("high", "low"):
        if col not in df.columns:
            raise ValueError(f"df must have a '{col}' column")

    n = len(df)
    # We need a left and a right shoulder, so candidate indices are
    # 1..n-2 inclusive. The last bar (n-1) has no right shoulder.
    if n < 3:
        return None
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    # Search window: most recent `lookback_bars` qualifying bars,
    # but clipped to the valid shoulder range.
    last_candidate = n - 2
    first_candidate = max(1, last_candidate - lookback_bars + 1)

    if direction == "BUY":
        result = _nearest_local_high_above(
            highs, first_candidate, last_candidate, entry_price,
        )
    else:
        result = _nearest_local_low_below(
            lows, first_candidate, last_candidate, entry_price,
        )

    logger.debug(
        "find_nearest_local_peak: direction={} entry={:.2f} "
        "window=[{},{}] result={}",
        direction, entry_price, first_candidate, last_candidate,
        f"{result:.2f}" if result is not None else "None",
    )
    return result


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _nearest_local_high_above(
    highs: np.ndarray,
    first: int, last: int,
    entry: float,
) -> float | None:
    """Lowest-priced bar.high in [first, last] that is both:

    * a local high (strict >) on highs
    * strictly above ``entry``
    """
    best: float | None = None
    for i in range(first, last + 1):
        h = float(highs[i])
        if h <= entry:
            continue
        if not (h > highs[i - 1] and h > highs[i + 1]):
            continue
        if best is None or h < best:
            best = h
    return best


def _nearest_local_low_below(
    lows: np.ndarray,
    first: int, last: int,
    entry: float,
) -> float | None:
    """Highest-priced bar.low in [first, last] that is both:

    * a local low (strict <) on lows
    * strictly below ``entry``
    """
    best: float | None = None
    for i in range(first, last + 1):
        lo = float(lows[i])
        if lo >= entry:
            continue
        if not (lo < lows[i - 1] and lo < lows[i + 1]):
            continue
        if best is None or lo > best:
            best = lo
    return best
