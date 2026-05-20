"""Zone marking — wick-inclusive envelope of the base.

Step 1 of zone construction. The remaining steps live in
:mod:`bot.strategy.zone_refinement` (size filter) and
:mod:`bot.strategy.strong_point` (Strong Point validation).

Geometry
--------

For both demand zones (RBR / DBR, BUY) and supply zones (DBD / RBD,
SELL) the zone is the **wick-inclusive envelope of the base candles
only**:

::

    zone.top    = max(bar.high for bar in df[base.start : base.end + 1])
    zone.bottom = min(bar.low  for bar in df[base.start : base.end + 1])

PR #60: strict base-only by default. PR #57 had tried to widen the
zone into bordering impulse bars to catch rejection wicks, but its
symmetric extension swept in the rally's peak (for BUY) or the drop's
trough (for SELL) — inflating zone width by tens of points.

``mark_zone`` keeps an opt-in ``wick_extend_bars`` kwarg. When > 0
the extension is **direction-aware**:

* BUY (demand) — only widens ``bottom`` (lower rejection wicks).
* SELL (supply) — only widens ``top`` (upper rejection wicks).

The opposite side stays at base. Production uses ``0`` (the
:class:`StrategyPipelineConfig` default).

Pre-PR (body-only) history
--------------------------
Earlier versions (pre-loosened S&D) marked the zone from candle
bodies only (``max(open, close)`` / ``min(open, close)``), excluding
wicks. We then moved to wick-inclusive over the base, which is the
current default. Downstream Strong Point validation uses body closes
for breaks and the SL buffer is independent of zone width.
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


def mark_zone(
    pattern: Pattern, df: pd.DataFrame, *, wick_extend_bars: int = 0,
) -> Zone:
    """Compute the zone's wick-inclusive envelope from the pattern's base.

    The base's candles are validated upstream by
    :func:`pattern_detection.detect_bases` — their range is bounded
    relative to the surrounding impulses. With ``wick_extend_bars=0``
    the zone reports the base's own wick extremes (``base.top`` /
    ``base.bottom``).

    PR #60: when ``wick_extend_bars > 0``, the zone is widened only
    in the **rejection direction**:

    * BUY (demand): scan lows in the window; ``bottom`` may move
      down. ``top`` is left at ``base.top``.
    * SELL (supply): scan highs in the window; ``top`` may move up.
      ``bottom`` is left at ``base.bottom``.

    Why direction-aware: a BUY zone's rejection wicks are the *lower*
    wicks below the base (price tried to fall, demand defended).
    *Upper* wicks on the border bars are the rally itself — including
    them pulls ``top`` up to the rally's peak, not a real rejection.
    The reverse holds for SELL. PR #57 widened symmetrically and
    over-extended both sides; PR #60 fixes that.

    The default in :class:`StrategyPipelineConfig` is ``1`` — capture
    the immediate border bar. ``0`` restores strict base-only
    behaviour. ``2+`` catches multi-bar rejection sequences.

    Pattern detection, base validation, lifecycle, SL distance buffer,
    TP chain, dedup, and freshness are all unaffected — only one side
    of the zone widens, in the rejection direction.
    """
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"df must have a '{col}' column")
    if wick_extend_bars < 0:
        raise ValueError(
            f"wick_extend_bars must be >= 0, got {wick_extend_bars}"
        )
    base = pattern.base
    n = len(df)
    if not (0 <= base.start_index <= base.end_index < n):
        raise ValueError(
            f"pattern.base indices out of df range: "
            f"start={base.start_index} end={base.end_index} len={n}"
        )
    if base.top < base.bottom:
        # Math guarantees this can't happen since base.top = max(high)
        # and base.bottom = min(low) over the same bar set, but be
        # loud if a malformed Pattern slips through.
        raise ValueError(
            f"pattern.base is inverted: top={base.top} < bottom={base.bottom}"
        )

    top = base.top
    bottom = base.bottom
    if wick_extend_bars > 0:
        # Scan ``wick_extend_bars`` bars on each side of the base for
        # rejection wicks. Clip to df bounds so we don't walk off the
        # start/end of the OHLC window. Direction-aware: BUY zones
        # only widen ``bottom`` (lower rejection wicks); SELL zones
        # only widen ``top`` (upper rejection wicks).
        ext_start = max(0, base.start_index - wick_extend_bars)
        ext_end = min(n - 1, base.end_index + wick_extend_bars)
        if ext_start < base.start_index or ext_end > base.end_index:
            if pattern.direction == "BUY":
                window_lows = df["low"].iloc[ext_start : ext_end + 1]
                bottom = float(min(bottom, window_lows.min()))
            else:
                window_highs = df["high"].iloc[ext_start : ext_end + 1]
                top = float(max(top, window_highs.max()))

    zone = Zone(
        direction=pattern.direction,
        top=top,
        bottom=bottom,
        formed_at=pattern.formed_at,
        source_pattern=pattern,
    )
    logger.debug(
        "{} zone marked: top={} bottom={} height={:.4f} pattern={} extend={}",
        zone.direction, zone.top, zone.bottom,
        zone.top - zone.bottom, pattern.pattern_type.value,
        wick_extend_bars,
    )
    return zone
