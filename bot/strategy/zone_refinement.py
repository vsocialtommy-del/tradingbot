"""Zone refinement — body-only edges and the size-filter verdict.

Performs steps 2 + 3 of zone construction (spec Sections 3.1 and 7):

    1. Box around the W/M's reversal area  (zone_marking)
    2. Refine to candle bodies only        ← this module
    3. Apply size filter (5-80 points)     ← this module

Refinement scope — the corrected spec
-------------------------------------
Refinement uses **only the bars at the swing-point indices** —
``low1.index`` and ``low2.index`` for W; ``high1.index`` and
``high2.index`` for M. The peak (or trough) and any bars between are
**excluded**. Including them — as the original spec wording suggested
— would expand the zone to the full vertical extent of the W/M, which
contradicts both S&D theory and the user-facing expectation that
refinement narrows the zone. See PR #7 for the full rationale.

Geometry
--------
For each pivot bar::

    body_top(bar)    = max(bar.open, bar.close)
    body_bottom(bar) = min(bar.open, bar.close)

For W (BUY) and M (SELL), the formula is the same — the math doesn't
care about direction. The only difference is *which two bars* the
refinement reads from::

    refined_top    = max(body_top(pivot_a),    body_top(pivot_b))
    refined_bottom = min(body_bottom(pivot_a), body_bottom(pivot_b))

Bullish vs bearish candles are handled identically: ``max(open, close)``
gives the body top regardless of direction. A bullish swing-low bar
(close > open) has ``body_top == close``; a bearish swing-low bar
(close < open) has ``body_top == open``. Either way, the wick above
the body is excluded.

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
    """Strip wicks at the swing-point bars and apply the size filter."""
    cfg = config or RefinementConfig()

    if "open" not in df.columns or "close" not in df.columns:
        raise ValueError("df must have 'open' and 'close' columns")

    pivot_indices = _pivot_indices(zone.source_pattern)
    n = len(df)
    for idx in pivot_indices:
        if not 0 <= idx < n:
            raise ValueError(
                f"swing-point index {idx} out of df range (len={n})"
            )

    body_tops: list[float] = []
    body_bottoms: list[float] = []
    for idx in pivot_indices:
        o = float(df["open"].iloc[idx])
        c = float(df["close"].iloc[idx])
        body_tops.append(max(o, c))
        body_bottoms.append(min(o, c))

    refined_top = max(body_tops)
    refined_bottom = min(body_bottoms)

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


def _pivot_indices(pattern: WPattern | MPattern) -> tuple[int, int]:
    """The two bar indices refinement reads body extremes from."""
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
