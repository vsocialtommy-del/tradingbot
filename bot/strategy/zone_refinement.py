"""Zone size-filter verdict.

Originally this module did two things:

1. Refine the wide initial zone (which included wicks) down to candle
   body extremes.
2. Apply a size filter (5-80 points by default) — reject zones too
   narrow or too wide to be tradeable.

After the S&D methodology switch (PR #31), step 1 is redundant:
``zone_marking.mark_zone`` now produces a wick-inclusive zone
directly from the pattern's base candles (which are themselves
validated for tightness at detection time). So this module becomes
**just the size filter** — top/bottom pass through unchanged.

The shape of :class:`RefinedZone` is preserved for backward compat
with all the consumers that already speak its API
(``bot.strategy.pipeline``, ``bot.strategy.strong_point``,
``bot.backtest.diagnose``, the deprecated ``bot.strategy.imbalance``).

Size filter
-----------

::

    width = top - bottom
    width <  zone_min_size_points  → ZONE_TOO_NARROW
    width >  zone_max_size_points  → ZONE_TOO_WIDE
    otherwise                      → tradeable

Both endpoints inclusive — width 5.0 is tradeable; width 80.0 is
tradeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import Pattern
from bot.strategy.zone_marking import Direction, Zone

RejectionReason = Literal["ZONE_TOO_NARROW", "ZONE_TOO_WIDE"]


@dataclass(frozen=True)
class RefinementConfig:
    """Tunables for the zone size filter."""

    zone_min_size_points: float = 5.0
    zone_max_size_points: float = 80.0


@dataclass(frozen=True)
class RefinedZone:
    """The size-filter verdict on a zone.

    Same shape as before the methodology change so downstream consumers
    don't break. ``top`` / ``bottom`` are now pass-through from the
    upstream :class:`Zone` (wick-inclusive); we just attach a
    tradeability verdict.
    """

    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: Pattern
    is_tradeable: bool
    rejection_reason: RejectionReason | None
    original_zone: Zone


def refine_zone(
    zone: Zone,
    df: pd.DataFrame,
    config: RefinementConfig | None = None,
) -> RefinedZone:
    """Apply the size filter; return a :class:`RefinedZone` verdict.

    ``df`` is kept in the signature for backward compat — earlier
    versions read OHLC here, the new implementation doesn't need it.
    The argument is accepted (and a couple of column-presence checks
    run) so callers don't break.
    """
    cfg = config or RefinementConfig()
    for col in ("open", "close"):
        if col not in df.columns:
            raise ValueError(f"df must have a '{col}' column")
    if zone.top < zone.bottom:
        raise ValueError(
            f"zone is inverted: top={zone.top} < bottom={zone.bottom}"
        )

    width = zone.top - zone.bottom
    is_tradeable, rejection_reason = _apply_size_filter(width, cfg)

    refined = RefinedZone(
        direction=zone.direction,
        top=zone.top,
        bottom=zone.bottom,
        formed_at=zone.formed_at,
        source_pattern=zone.source_pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,
        original_zone=zone,
    )
    logger.debug(
        "refined {} zone: width={:.4f} top={} bottom={} "
        "tradeable={} reason={}",
        zone.direction, width, zone.top, zone.bottom,
        is_tradeable, rejection_reason,
    )
    return refined


def _apply_size_filter(
    width: float, cfg: RefinementConfig,
) -> tuple[bool, RejectionReason | None]:
    if width < cfg.zone_min_size_points:
        return False, "ZONE_TOO_NARROW"
    if width > cfg.zone_max_size_points:
        return False, "ZONE_TOO_WIDE"
    return True, None
