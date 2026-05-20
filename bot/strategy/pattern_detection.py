"""Supply & Demand pattern detection — RBR / DBD / DBR / RBD.

The four S&D base patterns the entire methodology is built on. Every
tradeable zone in the bot's universe is the **base** of one of these:

==== === ===================== ====== =============
Code     Composition           Zone   Direction
==== === ===================== ====== =============
RBR      Rally → Base → Rally  demand BUY (cont.)
DBD      Drop  → Base → Drop   supply SELL (cont.)
DBR      Drop  → Base → Rally  demand BUY (rev.)
RBD      Rally → Base → Drop   supply SELL (rev.)
==== === ===================== ====== =============

Detection pipeline
------------------

``detect_impulses(df, config)`` finds all runs of strong same-direction
candles. Each run is one ``Impulse`` (1-5 bars). A weak / opposite
candle ends the run.

``detect_bases(df, impulses, config)`` finds compact consolidations
between adjacent impulses. A base is 1-N bars whose **total range**
is small relative to the surrounding impulses' ranges and where no
single base candle has a body comparable to an impulse body.

``classify_patterns(impulses, bases)`` glues each base to its
preceding and following impulse and classifies the resulting trio as
RBR / DBD / DBR / RBD.

``detect_patterns(df, config)`` is the top-level convenience that
chains all three. The strategy pipeline calls this.

Strength criteria
-----------------

An **impulse candle** requires BOTH:

1. ``body / total_range ≥ impulse_body_to_range_ratio_min`` (default
   0.0 — disabled). Pre-PR-46 this defaulted to 0.6 (filter out
   wick-heavy candles). PR #46 disables the filter by default
   because real impulses in XAUUSD M5 frequently include long wicks
   yet still trade through with conviction; the ATR multiple below
   is the surviving "is this a strong candle" guard. Operators can
   re-enable by setting the field explicitly.
2. ``body_size ≥ impulse_atr_multiple_min × ATR(atr_period)``
   (default 0.7 × ATR(14), loosened from 1.0 in PR #44) — filters
   small candles in low vol while still accepting impulses that
   are roughly ATR-sized.

A **tight base** requires BOTH:

1. Total base range (``max(high) - min(low)``, **wick-inclusive**)
   ≤ ``base_range_to_impulse_ratio_max`` × mean of the two adjacent
   impulses' ranges (default 1.0, loosened from 0.6 in PR #44). A
   base may now span roughly as wide as the impulses around it.
2. Largest base candle body (``max |close - open|``, body-only)
   ≤ ``base_max_body_to_impulse_body_ratio`` × the larger of the two
   adjacent impulses' largest bodies (default 0.4 — unchanged).

Both criteria must hold; either fails, the gap isn't a base.

PR-44 rationale
---------------
The strict pre-#44 defaults (1.0 × ATR, 0.6 base ratio) rejected
zones the user trades manually — visually clean S&D structures
that scored just outside the strict thresholds. The new defaults
(0.7, 1.0) lift detection rate at the cost of some false positives
that the body-break safety check and downstream gates should
catch. Strict-mode is preserved in ``test_pattern_detection.py
::TestStrictModeBaseline`` as a regression record.

PR-46 rationale
---------------
The body/range ratio (0.6) was still rejecting visually-clean
zones whose impulses had legitimate-size bodies but also
meaningful wicks (e.g. a 4-pt body on a 10-pt range = 40% — the
candle clearly moved 4 pts in one direction but the wicks fail
the 0.6 gate). The user observed real missed entries from this
filter even after PR #44. The ATR multiple (0.7) is preserved as
the primary "strong candle" gate; a candle with body ≥ 0.7 × ATR
is by definition a meaningful move regardless of wick proportion.
Strict 0.6 mode is preserved in ``TestStrictModeBaseline``.

Multi-candle impulses
---------------------

Per spec: an impulse is a **run** of 1 to ``max_impulse_run_candles``
consecutive same-direction strong candles. The run's ``range_size``
is computed once over the entire run, not summed across bars:

* RALLY: ``high(last_bar) - low(first_bar)``
* DROP:  ``high(first_bar) - low(last_bar)``

This avoids over-counting when a 3-bar rally is rolled into a single
impulse.

Bar-by-bar processing
---------------------

Detection runs over the whole df slice the pipeline passes in. A
pattern's ``formed_at`` is ``impulse_after.end_time`` — the moment
the pattern is CONFIRMED. The bot processes bars forward; nothing
is detected before its impulse_after has closed.

No retrospective magic. Backtest and live use identical logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


class PatternType(str, Enum):
    """The four S&D base patterns."""

    RBR = "RBR"  # Rally-Base-Rally  → demand zone, BUY (continuation)
    DBD = "DBD"  # Drop-Base-Drop    → supply zone, SELL (continuation)
    DBR = "DBR"  # Drop-Base-Rally   → demand zone, BUY (reversal)
    RBD = "RBD"  # Rally-Base-Drop   → supply zone, SELL (reversal)


ImpulseDirection = Literal["RALLY", "DROP"]
ZoneDirection = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Impulse:
    """A run of 1-5 consecutive strong same-direction candles."""

    direction: ImpulseDirection
    start_index: int                # first bar of the run (inclusive)
    end_index: int                  # last bar of the run (inclusive)
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    range_size: float
    """RALLY: ``high(end_bar) - low(start_bar)``;
    DROP: ``high(start_bar) - low(end_bar)``.
    Whole-run extent, NOT per-bar sum."""
    largest_body: float             # max |close − open| across the run
    candle_count: int               # 1..max_impulse_run_candles

    @property
    def is_rally(self) -> bool:
        return self.direction == "RALLY"


@dataclass(frozen=True)
class Base:
    """A compact consolidation between two impulses (1-N candles).

    ``top`` / ``bottom`` are **wick-inclusive** — max(high) / min(low)
    across the base bars. Zones are drawn at this envelope so the full
    rejection range is captured (PR adopting wick-marked zones).

    ``range_size = top - bottom`` therefore widens vs the previous
    body-only definition. The tightness check
    (``_tight_enough``, threshold ``base_range_to_impulse_ratio_max``)
    keys off this value; the 0.6 default may need re-tuning against
    demo-account data — left as a follow-up calibration.

    ``largest_body`` stays body-only (``max |close - open|``) — it's
    a "is any single candle in this 'base' actually impulse-sized?"
    sanity check that wicks shouldn't change.
    """

    start_index: int
    end_index: int
    candle_count: int
    top: float                      # max(high) across base candles (wick top)
    bottom: float                   # min(low) across base candles (wick bottom)
    range_size: float               # top - bottom (wick-inclusive)
    largest_body: float             # max |close - open| (body-only)


@dataclass(frozen=True)
class Pattern:
    """A confirmed RBR/DBD/DBR/RBD pattern."""

    pattern_type: PatternType
    impulse_before: Impulse
    base: Base
    impulse_after: Impulse
    direction: ZoneDirection        # BUY (RBR/DBR) or SELL (DBD/RBD)
    formed_at: pd.Timestamp         # impulse_after.end_time

    @property
    def base_start_index(self) -> int:
        return self.base.start_index

    @property
    def base_end_index(self) -> int:
        return self.base.end_index


@dataclass(frozen=True)
class PatternConfig:
    """Tunables for S&D pattern detection.

    Defaults calibrated for XAUUSD M5; live values come from
    ``bot_config`` (Phase E) and override these.
    """

    # Impulse strength
    impulse_body_to_range_ratio_min: float = 0.6
    """Body / total_range threshold. PR #64 restored the strict 0.6
    default (PR #46 had disabled it at 0.0). The 0.0 setting let
    wick-dominated bars qualify as impulses, which combined with the
    relaxed ATR multiple let "DBD" patterns mark inside continuous
    impulse legs — no real base, just a trending wick-heavy sequence.
    Operators who want PR #46's loose behaviour can set 0.0 explicitly."""
    impulse_atr_multiple_min: float = 1.0
    """Body ≥ 1.0 × ATR(14). PR #64 restored the strict 1.0 default
    (PR #44 had loosened to 0.7). The 0.7 setting accepted small-range
    bars in low-volatility periods as "impulses," producing patterns
    where the impulse_before and impulse_after were barely-bigger
    than the base they were supposed to be enclosing. Operators
    targeting tight pre-PR-44 calibration can stay at 1.0; those
    needing more signals can lower toward 0.7 (PR #44's choice)."""
    atr_period: int = 14
    max_impulse_run_candles: int = 5
    """Cap on consecutive strong same-direction candles in one impulse."""

    # Base shape
    min_base_candles: int = 1
    max_base_candles: int = 5
    base_range_to_impulse_ratio_max: float = 0.6
    """Total base range ≤ this × mean(impulse_before.range,
    impulse_after.range). PR #64 restored the strict 0.6 default
    (PR #44 had loosened to 1.0). The 1.0 setting allowed bases as
    wide as the impulses around them — which by definition isn't a
    base anymore, it's another impulse-class bar sequence. The 0.6
    default enforces the "tight consolidation" S&D premise. Operators
    can loosen back to 1.0 to catch wider-base setups, at the cost
    of false-positive "DBDs" in continuous trends."""
    base_max_body_to_impulse_body_ratio: float = 0.4
    """No single base body ≥ this × largest impulse body. Unchanged
    — this guards against a 'mini-impulse' candle inside the base."""

    # Pattern lookback (within-window cutoff at the caller)
    lookback_bars: int = 50


# --------------------------------------------------------------------------- #
# Top-level detection
# --------------------------------------------------------------------------- #


def detect_patterns(
    df: pd.DataFrame,
    config: PatternConfig | None = None,
) -> list[Pattern]:
    """Detect all RBR/DBD/DBR/RBD patterns in ``df``.

    Chains impulse detection → base detection → classification. The
    pipeline calls this from each per-bar invocation; lookback
    filtering (limit to recent ``lookback_bars``) is the caller's
    responsibility (it depends on the pipeline's per-bar window
    semantics).
    """
    cfg = config or PatternConfig()
    if len(df) == 0:
        return []
    impulses = detect_impulses(df, cfg)
    bases = detect_bases(df, impulses, cfg)
    patterns = classify_patterns(impulses, bases)

    # Lazy DEBUG — loguru skips str-build when filtered.
    logger.debug(
        "detect_patterns: {} impulses, {} bases, {} patterns",
        len(impulses), len(bases), len(patterns),
    )
    return patterns


# --------------------------------------------------------------------------- #
# Stage 1 — impulses
# --------------------------------------------------------------------------- #


def detect_impulses(
    df: pd.DataFrame,
    config: PatternConfig | None = None,
) -> list[Impulse]:
    """Find every impulse run in ``df``.

    Walks the df once, identifying which bars are individually
    "strong" (body/range and ATR criteria), then groups same-direction
    consecutive strong bars into runs up to ``max_impulse_run_candles``.

    A bar that fails either strength criterion ends any run in progress.
    """
    cfg = config or PatternConfig()
    if len(df) < cfg.atr_period + 1:
        return []
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"df must have a '{col}' column")

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df.index

    atr = _atr(highs, lows, closes, period=cfg.atr_period)

    n = len(df)
    impulses: list[Impulse] = []
    i = 0
    while i < n:
        # Skip until we find a strong candle.
        if not _is_strong_candle(
            i, opens, highs, lows, closes, atr, cfg,
        ):
            i += 1
            continue

        direction: ImpulseDirection = "RALLY" if closes[i] >= opens[i] else "DROP"
        run_start = i
        run_end = i  # inclusive

        # Extend the run forward while same-direction strong candles
        # continue, up to the cap.
        for j in range(i + 1, min(n, i + cfg.max_impulse_run_candles)):
            if not _is_strong_candle(j, opens, highs, lows, closes, atr, cfg):
                break
            j_direction: ImpulseDirection = (
                "RALLY" if closes[j] >= opens[j] else "DROP"
            )
            if j_direction != direction:
                break
            run_end = j

        impulses.append(_build_impulse(
            direction, run_start, run_end,
            opens, highs, lows, closes, times,
        ))
        i = run_end + 1

    logger.debug("detect_impulses: {} impulses found", len(impulses))
    return impulses


def _is_strong_candle(
    i: int,
    opens: np.ndarray, highs: np.ndarray,
    lows: np.ndarray, closes: np.ndarray,
    atr: np.ndarray, cfg: PatternConfig,
) -> bool:
    """Both strength criteria: body/range ratio AND body ≥ ATR multiple."""
    body = abs(closes[i] - opens[i])
    total_range = highs[i] - lows[i]
    if total_range <= 0:
        return False
    if body / total_range < cfg.impulse_body_to_range_ratio_min:
        return False
    if not np.isfinite(atr[i]) or atr[i] <= 0:
        return False
    if body < cfg.impulse_atr_multiple_min * atr[i]:
        return False
    return True


def _build_impulse(
    direction: ImpulseDirection,
    start_index: int, end_index: int,
    opens: np.ndarray, highs: np.ndarray,
    lows: np.ndarray, closes: np.ndarray,
    times: pd.DatetimeIndex,
) -> Impulse:
    """Compose an Impulse dataclass from the run boundaries."""
    if direction == "RALLY":
        range_size = float(highs[end_index] - lows[start_index])
    else:
        range_size = float(highs[start_index] - lows[end_index])
    bodies = np.abs(closes[start_index : end_index + 1]
                    - opens[start_index : end_index + 1])
    largest_body = float(bodies.max())
    return Impulse(
        direction=direction,
        start_index=start_index,
        end_index=end_index,
        start_time=times[start_index],
        end_time=times[end_index],
        range_size=range_size,
        largest_body=largest_body,
        candle_count=end_index - start_index + 1,
    )


# --------------------------------------------------------------------------- #
# Stage 2 — bases
# --------------------------------------------------------------------------- #


def detect_bases(
    df: pd.DataFrame,
    impulses: list[Impulse],
    config: PatternConfig | None = None,
) -> list[Base]:
    """Find every compact consolidation between adjacent impulses.

    For each consecutive pair ``(imp_a, imp_b)``, the gap between
    ``imp_a.end_index + 1`` and ``imp_b.start_index - 1`` (inclusive)
    is a candidate base. It must satisfy:

    * ``min_base_candles ≤ gap_length ≤ max_base_candles``
    * Total base range ≤ ``base_range_to_impulse_ratio_max`` × mean
      of imp_a and imp_b range sizes
    * Largest base body ≤ ``base_max_body_to_impulse_body_ratio`` ×
      max(imp_a.largest_body, imp_b.largest_body)
    """
    cfg = config or PatternConfig()
    if len(impulses) < 2:
        return []

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    bases: list[Base] = []
    for imp_a, imp_b in zip(impulses, impulses[1:]):
        gap_start = imp_a.end_index + 1
        gap_end = imp_b.start_index - 1
        n_gap = gap_end - gap_start + 1
        if n_gap < cfg.min_base_candles or n_gap > cfg.max_base_candles:
            continue

        base = _build_base(gap_start, gap_end, opens, highs, lows, closes)
        if not _tight_enough(base, imp_a, imp_b, cfg):
            continue
        bases.append(base)

    logger.debug("detect_bases: {} bases from {} impulse pairs",
                 len(bases), len(impulses) - 1)
    return bases


def _build_base(
    start_index: int, end_index: int,
    opens: np.ndarray, highs: np.ndarray,
    lows: np.ndarray, closes: np.ndarray,
) -> Base:
    """Wick-inclusive extents; body-only ``largest_body``.

    ``top = max(high)``, ``bottom = min(low)`` so zones drawn from this
    base include wicks (institutional rejection range). ``largest_body``
    stays body-only — wicks shouldn't make a candle look impulse-sized.
    """
    o = opens[start_index : end_index + 1]
    h = highs[start_index : end_index + 1]
    lo = lows[start_index : end_index + 1]
    c = closes[start_index : end_index + 1]
    top = float(h.max())
    bottom = float(lo.min())
    largest_body = float(np.abs(c - o).max())
    return Base(
        start_index=start_index,
        end_index=end_index,
        candle_count=end_index - start_index + 1,
        top=top,
        bottom=bottom,
        range_size=top - bottom,
        largest_body=largest_body,
    )


def _tight_enough(
    base: Base, imp_a: Impulse, imp_b: Impulse, cfg: PatternConfig,
) -> bool:
    """Both tightness criteria: total range AND no big-body candle."""
    mean_impulse_range = (imp_a.range_size + imp_b.range_size) / 2.0
    if mean_impulse_range <= 0:
        return False
    if base.range_size > cfg.base_range_to_impulse_ratio_max * mean_impulse_range:
        return False
    max_imp_body = max(imp_a.largest_body, imp_b.largest_body)
    if max_imp_body <= 0:
        return False
    if base.largest_body > cfg.base_max_body_to_impulse_body_ratio * max_imp_body:
        return False
    return True


# --------------------------------------------------------------------------- #
# Stage 3 — classification
# --------------------------------------------------------------------------- #


_PATTERN_TABLE: dict[
    tuple[ImpulseDirection, ImpulseDirection],
    tuple[PatternType, ZoneDirection],
] = {
    ("RALLY", "RALLY"): (PatternType.RBR, "BUY"),   # demand, continuation
    ("DROP",  "DROP"):  (PatternType.DBD, "SELL"),  # supply, continuation
    ("DROP",  "RALLY"): (PatternType.DBR, "BUY"),   # demand, reversal
    ("RALLY", "DROP"):  (PatternType.RBD, "SELL"),  # supply, reversal
}


def classify_patterns(
    impulses: list[Impulse], bases: list[Base],
) -> list[Pattern]:
    """Glue bases to their adjacent impulses and assign pattern_type."""
    if not bases:
        return []
    by_start: dict[int, Base] = {b.start_index: b for b in bases}
    patterns: list[Pattern] = []
    for imp_a, imp_b in zip(impulses, impulses[1:]):
        gap_start = imp_a.end_index + 1
        base = by_start.get(gap_start)
        if base is None or base.end_index != imp_b.start_index - 1:
            continue
        pattern_type, zone_direction = _PATTERN_TABLE[
            (imp_a.direction, imp_b.direction)
        ]
        patterns.append(Pattern(
            pattern_type=pattern_type,
            impulse_before=imp_a,
            base=base,
            impulse_after=imp_b,
            direction=zone_direction,
            formed_at=imp_b.end_time,
        ))
    return patterns


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int,
) -> np.ndarray:
    """ATR via Wilder's smoothing; returns ``NaN`` for the first ``period`` bars.

    Result is aligned to the input arrays — index ``i`` is the ATR
    *available at the close of bar i*. The first ``period`` entries
    are NaN because there isn't enough history to compute.
    """
    n = len(closes)
    if n == 0:
        return np.array([], dtype=float)
    true_range = np.empty(n, dtype=float)
    true_range[0] = highs[0] - lows[0]
    for i in range(1, n):
        a = highs[i] - lows[i]
        b = abs(highs[i] - closes[i - 1])
        c = abs(lows[i] - closes[i - 1])
        true_range[i] = max(a, b, c)
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    # Seed: simple mean over the first ``period`` true ranges.
    atr[period - 1] = true_range[:period].mean()
    # Wilder smoothing afterwards.
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + true_range[i]) / period
    return atr
