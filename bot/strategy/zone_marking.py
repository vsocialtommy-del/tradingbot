"""Zone marking — initial price box around a detected W or M pattern.

This module performs **step 1** of zone construction (spec Section 3.1):

    1. Box around the W/M's reversal area  ← THIS MODULE
    2. Refine to candle bodies only        ← zone_refinement.py (next)
    3. Apply size filters (5-80 points)    ← zone_refinement.py (next)

The output here is intentionally **wide** — it includes every wick that
printed during the W/M formation. Refinement narrows it to candle
bodies; the size filter rejects pathological cases. Splitting these
steps keeps each one's logic small and testable.

Geometry
--------
**W → demand zone (BUY direction)**

* ``top`` = ``max(low1.close, low2.close)`` — the **shallower** (higher)
  of the two swing lows. Conceptually the topmost edge of where buyers
  stepped in. Two reasons over alternatives:
    - Symmetric with M (use the swing on the trade-direction side).
    - Aligns with how SnD traders draw zones — start from where price
      first reversed, work down to the deepest probe.
* ``bottom`` = ``min(df.low)`` over ``[low1.index, low2.index]`` — the
  deepest wick during the W. Captures the full demand area.

**M → supply zone (SELL direction)** is the mirror.

Wide-zone tradeoff
------------------
A long wick at ``low1`` or ``low2`` can make the initial zone unusually
wide. This is **deliberate** in v1:

* The spec explicitly separates "Box around the reversal area" from
  "Adjust to candle bodies only, exclude wicks" — the body-only step
  is :mod:`bot.strategy.zone_refinement`'s job.
* If the wick is a noisy spike, refinement will slice it off.
* If the zone is still too wide after refinement, the size filter
  (5-80 points default) rejects it before any trade is taken.

So a wide initial zone is not a bug — it's input to the next two stages.
If neither of those tightens it enough, that's the system saying "this
isn't a tradeable zone", which is the correct outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import MPattern, WPattern

Direction = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Zone:
    """An initial price box from a W or M pattern.

    Invariant: ``top >= bottom``. They can be equal in degenerate cases
    (synthetic data where lows == closes and the two pivots are exactly
    equal); always non-negative height.
    """

    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: WPattern | MPattern


def mark_zone(pattern: WPattern | MPattern, df: pd.DataFrame) -> Zone:
    """Dispatch to the W or M marker based on pattern type."""
    if isinstance(pattern, WPattern):
        return mark_zone_from_w(pattern, df)
    if isinstance(pattern, MPattern):
        return mark_zone_from_m(pattern, df)
    raise TypeError(
        f"unsupported pattern type: {type(pattern).__name__} "
        f"(expected WPattern or MPattern)"
    )


def mark_zone_from_w(pattern: WPattern, df: pd.DataFrame) -> Zone:
    """Build the initial demand zone (BUY) for a W pattern."""
    if "low" not in df.columns:
        raise ValueError("df must have a 'low' column for W zone marking")
    _check_indices(pattern.low1.index, pattern.low2.index, len(df))

    box_top = float(max(pattern.low1.price, pattern.low2.price))
    lows_in_range = df["low"].iloc[
        pattern.low1.index : pattern.low2.index + 1
    ]
    box_bottom = float(lows_in_range.min())

    if box_top < box_bottom:
        # OHLC invariant violation — close should never be below low.
        raise ValueError(
            f"W zone produced inverted box: top={box_top} < bottom={box_bottom}. "
            f"Pattern's low1/low2 closes are below the bars' low values, which "
            f"violates OHLC invariants. Check the source DataFrame."
        )

    zone = Zone(
        direction="BUY",
        top=box_top,
        bottom=box_bottom,
        formed_at=pattern.formed_at,
        source_pattern=pattern,
    )
    logger.debug(
        f"W demand zone: top={zone.top} bottom={zone.bottom} "
        f"height={zone.top - zone.bottom:.4f} formed_at={zone.formed_at}"
    )
    return zone


def mark_zone_from_m(pattern: MPattern, df: pd.DataFrame) -> Zone:
    """Build the initial supply zone (SELL) for an M pattern."""
    if "high" not in df.columns:
        raise ValueError("df must have a 'high' column for M zone marking")
    _check_indices(pattern.high1.index, pattern.high2.index, len(df))

    box_bottom = float(min(pattern.high1.price, pattern.high2.price))
    highs_in_range = df["high"].iloc[
        pattern.high1.index : pattern.high2.index + 1
    ]
    box_top = float(highs_in_range.max())

    if box_top < box_bottom:
        raise ValueError(
            f"M zone produced inverted box: top={box_top} < bottom={box_bottom}. "
            f"Pattern's high1/high2 closes are above the bars' high values, "
            f"which violates OHLC invariants. Check the source DataFrame."
        )

    zone = Zone(
        direction="SELL",
        top=box_top,
        bottom=box_bottom,
        formed_at=pattern.formed_at,
        source_pattern=pattern,
    )
    logger.debug(
        f"M supply zone: top={zone.top} bottom={zone.bottom} "
        f"height={zone.top - zone.bottom:.4f} formed_at={zone.formed_at}"
    )
    return zone


def _check_indices(idx1: int, idx2: int, n: int) -> None:
    if idx1 < 0 or idx2 < 0 or idx1 >= n or idx2 >= n:
        raise ValueError(
            f"pattern indices ({idx1}, {idx2}) out of df range (len={n})"
        )
    if idx1 > idx2:
        raise ValueError(
            f"pattern indices in wrong order: idx1={idx1} > idx2={idx2}"
        )
