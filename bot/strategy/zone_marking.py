"""Zone marking — body extremes of the pattern's base candles.

Step 1 of zone construction. The remaining steps live in
:mod:`bot.strategy.zone_refinement` (size filter) and
:mod:`bot.strategy.strong_point` (Strong Point validation).

Geometry
--------

For both demand zones (RBR / DBR, BUY) and supply zones (DBD / RBD,
SELL) the zone is the **body envelope** of the pattern's base:

::

    body_top(bar)    = max(bar.open, bar.close)
    body_bottom(bar) = min(bar.open, bar.close)

    zone.top    = max(body_top    for bar in base)
    zone.bottom = min(body_bottom for bar in base)

Wicks are excluded by construction (we operate on bodies, not
highs/lows). The base candles are already known to be tightly clustered
(see :mod:`bot.strategy.pattern_detection` — base validation enforces
total-range and per-body limits), so the zone is automatically
"compact" without any further refinement step.

Pre-PR #31 history
------------------
Earlier versions of this module produced a wide initial box (including
wicks) that ``zone_refinement.refine_zone`` then stripped down. That
two-step process made sense for W/M pattern detection where the
"pattern area" was the swing-low / swing-high bars. With the new S&D
methodology the base IS the body envelope by definition — no separate
refinement step required. ``refine_zone`` still exists for the size
filter (5-80 points) but no longer mutates the zone's top/bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import Pattern

Direction = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Zone:
    """The body envelope of a pattern's base.

    Invariant: ``top >= bottom`` (they can be equal in pathological
    cases — e.g. an all-doji base where every bar has open == close;
    the size filter rejects those).
    """

    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: Pattern


def mark_zone(pattern: Pattern, df: pd.DataFrame) -> Zone:
    """Compute the zone's body envelope from the pattern's base.

    The base's candles are validated upstream by
    :func:`pattern_detection.detect_bases` — their range is bounded
    relative to the surrounding impulses. We just read the body
    extremes here.

    ``df`` is accepted (and validated) for API symmetry with the
    previous version of this module; we don't actually need to
    re-read OHLC here because ``Pattern.base`` already carries
    ``top`` and ``bottom`` (computed from bodies during base
    detection). Keeping ``df`` in the signature avoids breaking
    callers + tests during the methodology transition.
    """
    for col in ("open", "close"):
        if col not in df.columns:
            raise ValueError(f"df must have a '{col}' column")
    base = pattern.base
    n = len(df)
    if not (0 <= base.start_index <= base.end_index < n):
        raise ValueError(
            f"pattern.base indices out of df range: "
            f"start={base.start_index} end={base.end_index} len={n}"
        )
    if base.top < base.bottom:
        # Math guarantees this can't happen since base.top = max(body_top)
        # and base.bottom = min(body_bottom) over the same bar set,
        # but be loud if a malformed Pattern slips through.
        raise ValueError(
            f"pattern.base is inverted: top={base.top} < bottom={base.bottom}"
        )

    zone = Zone(
        direction=pattern.direction,
        top=base.top,
        bottom=base.bottom,
        formed_at=pattern.formed_at,
        source_pattern=pattern,
    )
    logger.debug(
        "{} zone marked: top={} bottom={} height={:.4f} pattern={}",
        zone.direction, zone.top, zone.bottom,
        zone.top - zone.bottom, pattern.pattern_type.value,
    )
    return zone
