"""W and M pattern detection on close prices.

Builds on :mod:`bot.strategy.structure` for swing detection. Pure-Python
logic on pandas DataFrames — no MT5, no Supabase, no I/O.

Definitions
-----------
W pattern (BUY signal, double-bottom)
    Two swing lows whose prices are within ``pattern_tolerance_pct`` of
    each other (computed against their **average** — see "Conventions"
    below), with a higher peak between them. The peak is the highest
    close strictly between ``low1`` and ``low2``; it must clear the
    higher of the two lows by at least ``peak_threshold_pct``.

M pattern (SELL signal, double-top)
    Mirror of W. Two swing highs within tolerance, a lower trough
    between them clearing the lower of the two highs by the threshold
    on the downside.

N pattern
    Reserved — explicitly deferred to v2 per spec Section 2.3 (often
    overlaps with CHoCH and is harder to codify in isolation). No
    detection function is provided in v1.

Conventions
-----------
1. **Tolerance reference = average of the two pivots.**
   ``diff_pct = |p1 - p2| / mean(p1, p2) * 100``.
   Symmetric (order-independent), and the most defensible choice when
   asking "are these two prices essentially the same?".

2. **Peak threshold reference = the higher of the two lows** (and
   symmetrically, the lower of the two highs for M). Requires the peak
   to clear the *more conservative* low so the pattern is meaningfully
   shaped, not just barely above the lower low.

3. **Peak / trough need not be a swing.** We take the highest (lowest)
   close strictly between the two pivots. Often it'll coincide with a
   swing high (low), but we don't require it — the spec's "higher peak
   between them" is a price-level requirement, not a structural one.

4. **All-pairs detection.** With more than two same-kind swings in the
   window, every pair ``(low_i, low_j)`` with ``i < j`` is a candidate.
   ``detect_latest_w`` then picks the W with the highest ``low2.index``
   (latest second low), tie-breaking on the highest ``low1.index`` (the
   tightest most-recent W).

5. **"Completed" = both pivots are confirmed swings.** :func:`detect_swings`
   only returns swings with a full right-shoulder (``strength`` bars after
   the pivot). So a pattern whose second low is still forming (no
   right-shoulder yet) simply never appears in the swing list and is not
   returned. The :class:`WPattern` / :class:`MPattern` dataclasses carry
   a ``completed`` flag that is always ``True`` in v1 — preserved as a
   field so a future "forming-pattern" detector (e.g. for live entry
   anticipation) can be added without an API break.

6. **Lookback window.** Both pivots must satisfy
   ``index >= latest_bar - lookback_bars + 1`` — i.e. they must lie in
   the most recent ``lookback_bars`` of the DataFrame. Patterns formed
   earlier in the data are filtered out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd
from loguru import logger

from bot.strategy.structure import Swing, detect_swings


class PatternType(str, Enum):
    """The shape of a detected pattern."""

    W = "W"  # double-bottom (BUY signal)
    M = "M"  # double-top (SELL signal)
    N = "N"  # reserved — deferred to v2 (spec Section 2.3)


@dataclass(frozen=True)
class PatternConfig:
    """Tunables for pattern detection.

    Defaults match the seeded ``bot_config`` keys; the orchestrator pulls
    values from Supabase and passes them in here.
    """

    swing_strength: int = 3
    # Maximum percentage difference between the two pivots, calculated
    # against their average (see module docstring). 0.1% is the spec's
    # default (Section 13).
    pattern_tolerance_pct: float = 0.1
    # Minimum percentage by which the peak must clear the higher low
    # (or trough below the lower high). 0.2% is a reasonable default
    # for Gold; tunable in backtest.
    peak_threshold_pct: float = 0.2
    # Both pivots must lie within the most recent N bars of the
    # DataFrame. On M5 (the v1 strategy timeframe) 50 bars ≈ 4 hours.
    lookback_bars: int = 50


@dataclass(frozen=True)
class WPattern:
    """A detected double-bottom pattern."""

    low1: Swing
    low2: Swing
    peak_index: int
    peak_time: pd.Timestamp
    peak_price: float
    formed_at: pd.Timestamp  # = low2.time
    completed: bool  # always True in v1 — see module docstring

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.W


@dataclass(frozen=True)
class MPattern:
    """A detected double-top pattern."""

    high1: Swing
    high2: Swing
    trough_index: int
    trough_time: pd.Timestamp
    trough_price: float
    formed_at: pd.Timestamp  # = high2.time
    completed: bool

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.M


# --------------------------------------------------------------------------- #
# W detection
# --------------------------------------------------------------------------- #


def detect_w_patterns(
    df: pd.DataFrame,
    config: PatternConfig | None = None,
) -> list[WPattern]:
    """Find every W pattern in ``df`` (all-pairs over confirmed swing lows)."""
    cfg = config or PatternConfig()
    if len(df) == 0:
        return []

    swings = detect_swings(df, cfg.swing_strength)
    lows = sorted(
        (s for s in swings if s.kind == "LOW"), key=lambda s: s.index
    )
    if len(lows) < 2:
        return []

    closes = df["close"].to_numpy()
    latest_bar = len(df) - 1
    lookback_cutoff = latest_bar - cfg.lookback_bars + 1

    patterns: list[WPattern] = []
    for i in range(len(lows)):
        low1 = lows[i]
        if low1.index < lookback_cutoff:
            continue
        for j in range(i + 1, len(lows)):
            low2 = lows[j]
            # low2.index is necessarily >= low1.index, so the lookback
            # check above already covers low2 — no need to recheck.

            if not _within_tolerance(
                low1.price, low2.price, cfg.pattern_tolerance_pct
            ):
                continue

            peak_idx_offset, peak_price = _highest_close_between(
                closes, low1.index, low2.index
            )
            if peak_idx_offset is None:
                continue
            peak_index = low1.index + 1 + peak_idx_offset

            if not _peak_clears_threshold(
                peak_price,
                higher_low=max(low1.price, low2.price),
                threshold_pct=cfg.peak_threshold_pct,
            ):
                continue

            patterns.append(
                WPattern(
                    low1=low1,
                    low2=low2,
                    peak_index=peak_index,
                    peak_time=df.index[peak_index],
                    peak_price=peak_price,
                    formed_at=low2.time,
                    completed=True,
                )
            )

    logger.debug(
        f"detect_w_patterns: {len(lows)} lows, {len(patterns)} W patterns"
    )
    return patterns


def detect_latest_w(
    df: pd.DataFrame,
    config: PatternConfig | None = None,
) -> WPattern | None:
    """Return the most recent W pattern, or ``None`` if none exists.

    "Most recent" = highest ``low2.index``. Ties (same second low,
    different first lows) are broken by preferring the highest
    ``low1.index`` — the tightest most-recent W.
    """
    patterns = detect_w_patterns(df, config)
    if not patterns:
        return None
    return max(patterns, key=lambda p: (p.low2.index, p.low1.index))


# --------------------------------------------------------------------------- #
# M detection
# --------------------------------------------------------------------------- #


def detect_m_patterns(
    df: pd.DataFrame,
    config: PatternConfig | None = None,
) -> list[MPattern]:
    """Find every M pattern in ``df`` (all-pairs over confirmed swing highs)."""
    cfg = config or PatternConfig()
    if len(df) == 0:
        return []

    swings = detect_swings(df, cfg.swing_strength)
    highs = sorted(
        (s for s in swings if s.kind == "HIGH"), key=lambda s: s.index
    )
    if len(highs) < 2:
        return []

    closes = df["close"].to_numpy()
    latest_bar = len(df) - 1
    lookback_cutoff = latest_bar - cfg.lookback_bars + 1

    patterns: list[MPattern] = []
    for i in range(len(highs)):
        high1 = highs[i]
        if high1.index < lookback_cutoff:
            continue
        for j in range(i + 1, len(highs)):
            high2 = highs[j]

            if not _within_tolerance(
                high1.price, high2.price, cfg.pattern_tolerance_pct
            ):
                continue

            trough_offset, trough_price = _lowest_close_between(
                closes, high1.index, high2.index
            )
            if trough_offset is None:
                continue
            trough_index = high1.index + 1 + trough_offset

            if not _trough_clears_threshold(
                trough_price,
                lower_high=min(high1.price, high2.price),
                threshold_pct=cfg.peak_threshold_pct,
            ):
                continue

            patterns.append(
                MPattern(
                    high1=high1,
                    high2=high2,
                    trough_index=trough_index,
                    trough_time=df.index[trough_index],
                    trough_price=trough_price,
                    formed_at=high2.time,
                    completed=True,
                )
            )

    logger.debug(
        f"detect_m_patterns: {len(highs)} highs, {len(patterns)} M patterns"
    )
    return patterns


def detect_latest_m(
    df: pd.DataFrame,
    config: PatternConfig | None = None,
) -> MPattern | None:
    """Return the most recent M pattern, or ``None``."""
    patterns = detect_m_patterns(df, config)
    if not patterns:
        return None
    return max(patterns, key=lambda p: (p.high2.index, p.high1.index))


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _within_tolerance(p1: float, p2: float, tolerance_pct: float) -> bool:
    """``|p1 - p2| / mean(p1, p2) * 100 <= tolerance_pct``.

    Inclusive at the boundary so a pair sitting exactly at the tolerance
    threshold is accepted.
    """
    avg = (p1 + p2) / 2.0
    if avg <= 0:
        return False
    diff_pct = abs(p1 - p2) / avg * 100.0
    return diff_pct <= tolerance_pct


def _highest_close_between(
    closes,
    left_idx: int,
    right_idx: int,
) -> tuple[int | None, float]:
    """Highest close strictly between ``left_idx`` and ``right_idx``.

    Returns ``(offset_from_left_plus_1, price)`` so the absolute bar
    index is ``left_idx + 1 + offset``. Returns ``(None, 0.0)`` if the
    interval is empty (adjacent pivots).
    """
    sl = closes[left_idx + 1 : right_idx]
    if len(sl) == 0:
        return None, 0.0
    offset = int(sl.argmax())
    return offset, float(sl[offset])


def _lowest_close_between(
    closes,
    left_idx: int,
    right_idx: int,
) -> tuple[int | None, float]:
    """Lowest close strictly between ``left_idx`` and ``right_idx``."""
    sl = closes[left_idx + 1 : right_idx]
    if len(sl) == 0:
        return None, 0.0
    offset = int(sl.argmin())
    return offset, float(sl[offset])


def _peak_clears_threshold(
    peak_price: float, higher_low: float, threshold_pct: float
) -> bool:
    """``peak_price >= higher_low * (1 + threshold_pct / 100)``."""
    return peak_price >= higher_low * (1.0 + threshold_pct / 100.0)


def _trough_clears_threshold(
    trough_price: float, lower_high: float, threshold_pct: float
) -> bool:
    """``trough_price <= lower_high * (1 - threshold_pct / 100)``."""
    return trough_price <= lower_high * (1.0 - threshold_pct / 100.0)
