"""Zone marking — wick-inclusive envelope of the base + border wicks.

Step 1 of zone construction. The remaining steps live in
:mod:`bot.strategy.zone_refinement` (size filter) and
:mod:`bot.strategy.strong_point` (Strong Point validation).

Geometry
--------

For both demand zones (RBR / DBR, BUY) and supply zones (DBD / RBD,
SELL) the zone is the **wick-inclusive envelope** of the base, with
optional rejection-wick extension on either side:

::

    ext_start  = max(0, base.start_index - wick_extend_bars)
    ext_end    = min(n - 1, base.end_index + wick_extend_bars)
    zone.top    = max(bar.high for bar in df[ext_start : ext_end + 1])
    zone.bottom = min(bar.low  for bar in df[ext_start : ext_end + 1])

PR #57: ``wick_extend_bars`` defaults to ``1`` via the strategy
config. Captures the rejection wicks on the last bar of the rally
(impulse_before) and the first bar of the drop (impulse_after).
Pre-PR-#57 behaviour was strict base-only (``wick_extend_bars=0``);
set the config field back to ``0`` to restore it.

Pre-PR (body-only) history
--------------------------
Earlier versions (pre-loosened S&D) marked the zone from candle
bodies only (``max(open, close)`` / ``min(open, close)``), excluding
wicks. Then we moved to wick-inclusive over the base. PR #57 widens
to wick-inclusive over the base plus border bars. Downstream Strong
Point validation still uses body closes for breaks and the SL buffer
is independent of zone width, so SL placement adjusts by exactly
the amount the zone widened.
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

    PR #57: when ``wick_extend_bars > 0``, the zone's top/bottom are
    widened to include the highs/lows of ``N`` bars on each side of
    the base's index range. The motivation:

    The base-detection algorithm picks a tightly-clustered 1-5 bar
    window as "the base", and the immediate border bars (last bar
    of impulse_before, first bar of impulse_after) often have
    rejection wicks that extend past the base's wick range — they're
    where institutional supply/demand defended the level before the
    move started or finished. Strict base-only marking excludes them
    and the resulting zone is too narrow relative to what manual
    traders draw and the market actually respects. Operator-observed:
    bot's zones consistently $5-10 narrower on top for SELL zones
    because the highest rejection wick was on the first drop bar,
    classified as impulse_after rather than base.

    The default in :class:`StrategyPipelineConfig` is now ``1`` —
    capture immediate border bars. ``0`` restores the strict
    base-only behaviour. ``2+`` is more aggressive (catches
    multi-bar rejection sequences).

    Pattern detection, base validation, lifecycle, SL distance buffer,
    TP chain, dedup, and freshness are all unaffected — only the
    zone's reported top/bottom widen.
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
        # higher highs / lower lows. Clip to df bounds so we don't
        # walk off the start/end of the OHLC window. Border bars that
        # don't extend the zone (their high <= top, their low >=
        # bottom) leave the zone unchanged; only genuine rejection
        # wicks widen it.
        ext_start = max(0, base.start_index - wick_extend_bars)
        ext_end = min(n - 1, base.end_index + wick_extend_bars)
        if ext_start < base.start_index or ext_end > base.end_index:
            window_highs = df["high"].iloc[ext_start : ext_end + 1]
            window_lows = df["low"].iloc[ext_start : ext_end + 1]
            top = float(max(top, window_highs.max()))
            bottom = float(min(bottom, window_lows.min()))

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
