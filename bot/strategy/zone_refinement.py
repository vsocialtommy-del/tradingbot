"""Zone refinement — body-only edges and the size-filter verdict.

Performs steps 2 + 3 of zone construction (spec Sections 3.1 and 7):

    1. Box around the W/M's reversal area  (zone_marking)
    2. Refine to candle bodies only        ← this module
    3. Apply size filter (5-80 points)     ← this module

Refinement scope — full pattern area
------------------------------------
Refinement reads candle bodies across **every bar between the two
swing pivots, inclusive** (``low1.index`` through ``low2.index`` for
W; ``high1.index`` through ``high2.index`` for M). This captures the
full "bottom area" of a W (or "top area" of an M) the trader treats
as the demand / supply zone — including any bars in a multi-bar low
cluster that aren't themselves classified as swing pivots.

Same scope applies to BOTH W and M — there's no asymmetry between
directions; the math is identical, just the bar range changes per
pattern type.

History note: an earlier version of this module read only the two
pivot bars (excluding everything between, including the peak/trough).
That was too narrow for the user's trading style — see the
calibration thread + PR diagnostic for the rationale behind the
widening.

Geometry
--------
For each bar in the inclusive range::

    body_top(bar)    = max(bar.open, bar.close)
    body_bottom(bar) = min(bar.open, bar.close)

Per pattern, the refined zone is::

    refined_top    = max(body_top    of every bar in range)
    refined_bottom = min(body_bottom of every bar in range)

Bullish vs bearish candles are handled identically: ``max(open, close)``
gives the body top regardless of direction. A bullish bar (close > open)
has ``body_top == close``; a bearish bar (close < open) has
``body_top == open``. Either way, the wick above the body is excluded.

Size-filter interaction
-----------------------
With the wider scope, a W with a tall middle peak (or an M with a
deep middle trough) produces a correspondingly tall refined zone.
The size filter's max width then rejects pathological cases: a W
whose peak is so high above the lows that the resulting body box
exceeds ``zone_max_size_points`` is correctly flagged as
``ZONE_TOO_WIDE``. That's the system saying "the pattern is too
stretched to be a tradeable demand zone" — desired behaviour.

Size filter (spec Section 7)
----------------------------
After refinement, width = ``refined_top - refined_bottom`` is checked
against the configured band:

    width <  zone_min_size_points  → ZONE_TOO_NARROW
    width >  zone_max_size_points  → ZONE_TOO_WIDE
    otherwise                      → tradeable

Both endpoints are **inclusive**: ``width == 5`` is tradeable;
``width == 80`` is tradeable.

Doji-only zones
---------------
A doji has ``open == close``, so ``body_top == body_bottom``. Refinement
still works — each pivot bar contributes a single point. If both pivot
bars are dojis at the same close, the refined zone has ``top == bottom``
(width 0) → fails the size filter as ``ZONE_TOO_NARROW``. This is the
correct behaviour: a zero-height zone isn't tradeable.

Invariant
---------
``refined_top >= refined_bottom`` always: ``max(body_top of two bars)
>= max(body_bottom of two bars) >= min(body_bottom of two bars)``.
The constructor guards against violations defensively but the math
doesn't allow them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.zone_marking import Direction, Zone

RejectionReason = Literal["ZONE_TOO_NARROW", "ZONE_TOO_WIDE"]


@dataclass(frozen=True)
class RefinementConfig:
    """Tunables for zone refinement and size filtering.

    Defaults match spec Section 7 zone-size band and the seeded
    ``bot_config`` keys ``zone_min_size_points`` / ``zone_max_size_points``.
    """

    zone_min_size_points: float = 5.0
    zone_max_size_points: float = 80.0


@dataclass(frozen=True)
class RefinedZone:
    """A zone with body-only edges plus a tradeability verdict."""

    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: WPattern | MPattern
    is_tradeable: bool
    rejection_reason: RejectionReason | None
    original_zone: Zone


def refine_zone(
    zone: Zone,
    df: pd.DataFrame,
    config: RefinementConfig | None = None,
) -> RefinedZone:
    """Strip wicks across the full pattern area and apply the size filter."""
    cfg = config or RefinementConfig()

    if "open" not in df.columns or "close" not in df.columns:
        raise ValueError("df must have 'open' and 'close' columns")

    start_idx, end_idx = _refinement_range(zone.source_pattern)
    n = len(df)
    if not 0 <= start_idx < n:
        raise ValueError(
            f"swing-point index {start_idx} out of df range (len={n})"
        )
    if not 0 <= end_idx < n:
        raise ValueError(
            f"swing-point index {end_idx} out of df range (len={n})"
        )

    # Vectorised body extremes across every bar in [start_idx, end_idx].
    opens = df["open"].iloc[start_idx : end_idx + 1].to_numpy()
    closes = df["close"].iloc[start_idx : end_idx + 1].to_numpy()
    body_tops_arr = np.maximum(opens, closes)
    body_bottoms_arr = np.minimum(opens, closes)
    refined_top = float(body_tops_arr.max())
    refined_bottom = float(body_bottoms_arr.min())

    # Defensive — math guarantees this can't happen, but better to fail
    # loudly than silently return a corrupted zone.
    if refined_top < refined_bottom:
        raise ValueError(
            f"refined zone is inverted: top={refined_top} < bottom={refined_bottom}"
        )

    width = refined_top - refined_bottom
    is_tradeable, rejection_reason = _apply_size_filter(width, cfg)

    refined = RefinedZone(
        direction=zone.direction,
        top=refined_top,
        bottom=refined_bottom,
        formed_at=zone.formed_at,
        source_pattern=zone.source_pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,
        original_zone=zone,
    )
    logger.debug(
        f"refined {zone.direction} zone: width={width:.4f} "
        f"top={refined_top} bottom={refined_bottom} "
        f"tradeable={is_tradeable} reason={rejection_reason}"
    )
    return refined


def _refinement_range(pattern: WPattern | MPattern) -> tuple[int, int]:
    """``(start_idx, end_idx)`` over which refinement reads body extremes.

    Inclusive on both ends — for W, that's ``[low1.index, low2.index]``;
    for M, ``[high1.index, high2.index]``. Same range semantics for both
    directions.
    """
    if isinstance(pattern, WPattern):
        return pattern.low1.index, pattern.low2.index
    if isinstance(pattern, MPattern):
        return pattern.high1.index, pattern.high2.index
    raise TypeError(
        f"unsupported source_pattern type: {type(pattern).__name__} "
        f"(expected WPattern or MPattern)"
    )


def _apply_size_filter(
    width: float, cfg: RefinementConfig
) -> tuple[bool, RejectionReason | None]:
    """Return (is_tradeable, rejection_reason)."""
    if width < cfg.zone_min_size_points:
        return False, "ZONE_TOO_NARROW"
    if width > cfg.zone_max_size_points:
        return False, "ZONE_TOO_WIDE"
    return True, None
