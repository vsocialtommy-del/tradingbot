"""Zone marking — wick-inclusive envelope of the pattern's base candles.

Step 1 of zone construction. The remaining steps live in
:mod:`bot.strategy.zone_refinement` (size filter) and
:mod:`bot.strategy.strong_point` (Strong Point validation).

Geometry
--------

For both demand zones (RBR / DBR, BUY) and supply zones (DBD / RBD,
SELL) the zone is the **wick-inclusive envelope** of the base:

::

    zone.top    = max(bar.high for bar in base)
    zone.bottom = min(bar.low  for bar in base)

Wicks are INCLUDED. Rationale: long wicks at the base mark the
exact prices where institutional orders defended the level; the
zone should encompass the full rejection range, not just where bars
happened to close. The base candles are already known to be tightly
clustered (:mod:`bot.strategy.pattern_detection` — base validation
enforces total-range and per-body limits over the wick range), so
the zone is automatically "compact" without any further refinement.

Pre-PR (body-only) history
--------------------------
Earlier versions marked the zone from candle bodies only
(``max(open, close)`` / ``min(open, close)``), excluding wicks. The
move to wick-inclusive marking is documented in the PR introducing
this change; downstream Strong Point validation still uses body
closes for breaks and the SL anchor is independent of zone bounds,
so SL distance and break detection are unaffected — only zone
widths shift.
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
    """The wick-inclusive envelope of a pattern's base.

    Invariant: ``top >= bottom`` (they can be equal in pathological
    cases — e.g. a single-bar base with high == low; the size filter
    rejects those).
    """

    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: Pattern


def mark_zone(pattern: Pattern, df: pd.DataFrame) -> Zone:
    """Compute the zone's wick-inclusive envelope from the pattern's base.

    The base's candles are validated upstream by
    :func:`pattern_detection.detect_bases` — their range is bounded
    relative to the surrounding impulses. We just read the wick
    extremes here (the upstream :class:`Base` already stores
    ``top = max(high)`` and ``bottom = min(low)``).

    ``df`` is accepted (and column-checked) for API symmetry with
    callers and historical signatures; the actual zone bounds come
    from ``pattern.base`` so no OHLC re-read happens here.
    """
    for col in ("open", "high", "low", "close"):
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
        top=base.top,        # max(high) over base bars — wicks included
        bottom=base.bottom,  # min(low)  over base bars
        formed_at=pattern.formed_at,
        source_pattern=pattern,
    )
    logger.debug(
        "{} zone marked: top={} bottom={} height={:.4f} pattern={}",
        zone.direction, zone.top, zone.bottom,
        zone.top - zone.bottom, pattern.pattern_type.value,
    )
    return zone
