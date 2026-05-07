"""Strong Point validation — three quality gates for a tradeable zone.

Per spec Section 3.1, a Strong Point demand / supply zone requires
*proof of institutional involvement*. We test three things:

1. **Move out broke structure (BoS)** — after the pattern completed,
   a later bar's close must have broken a prior swing high (W →
   UP BoS) or low (M → DOWN BoS). Reuses :func:`detect_bos` output
   from :mod:`bot.strategy.structure`.

2. **Base is compact** — the two pivot bars (the W's lows or the M's
   highs) had small ranges relative to the impulse. We require
   ``base_range / impulse_range <= base_max_range_ratio`` (default
   0.5) for *both* base bars.

3. **Impulse is strong** — the BoS bar itself must be a decisive
   directional candle:
       * ``body / range >= impulse_min_body_ratio`` (default 0.6).
       * Closes in the BoS direction (bullish for UP, bearish for
         DOWN). Filters out cases where a wick poked through the
         level on a bar that closed against the breakout.

If any gate fails the zone is not a Strong Point. Failures are
**collected**, not short-circuited — a zone that fails two gates
returns both reasons in ``validation_failures`` so the dashboard
and backtest analytics get the full picture.

Failure codes
-------------
``NOT_TRADEABLE``         Zone failed the size filter upstream
                          (``RefinedZone.is_tradeable == False``);
                          we don't bother running gates 1-3.
``NO_BOS_YET``            No BoS event of the matching direction
                          exists after the pattern's completion bar.
                          Includes the case where only opposite-
                          direction events exist (DOWN events for a
                          W zone) — same outcome, no usable BoS.
``IMPULSE_TOO_WEAK``      BoS bar's body / range below threshold.
``IMPULSE_WRONG_DIRECTION`` BoS bar closes against the BoS direction
                          (bearish on UP BoS, bullish on DOWN BoS,
                          or doji).
``BASE_NOT_COMPACT``      One or both pivot bars have a range
                          exceeding the configured ratio of the
                          impulse bar's range.

Decisions called out in the PR description:

* **Impulse bar = the BoS bar itself.** Defensible because it's the
  bar whose close *caused* the structural break. An alternative —
  "largest-range bar in the leg from low2 to BoS" — could be added
  later as a config flag if backtests show it's needed.
* **Boundaries inclusive.** ``body/range == 0.6`` passes; ``base_range
  / impulse_range == 0.5`` passes. Strict inequality used on the
  failure side.
* **Multiple BoS events** → we use the *first* (smallest bar_index)
  matching the zone's direction. ``bos_events`` from
  :func:`detect_bos` is already chronological.
* **Opposite-direction events** are silently filtered out. They're
  not "errors"; they just don't qualify as the move-out for this
  zone.
* **Zero-range bars** are non-pathological:
    * Zero-range *impulse* → fails ``IMPULSE_TOO_WEAK`` (can't be
      decisive). No division-by-zero.
    * Zero-range *base* → trivially passes the compactness gate
      (smaller than any positive impulse).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.structure import BosEvent
from bot.strategy.zone_marking import Direction, Zone
from bot.strategy.zone_refinement import RefinedZone, RejectionReason

ValidationFailure = Literal[
    "NOT_TRADEABLE",
    "NO_BOS_YET",
    "IMPULSE_TOO_WEAK",
    "IMPULSE_WRONG_DIRECTION",
    "BASE_NOT_COMPACT",
]


@dataclass(frozen=True)
class StrongPointConfig:
    """Tunables for the three validation gates."""

    impulse_min_body_ratio: float = 0.6
    base_max_range_ratio: float = 0.5


@dataclass(frozen=True)
class ValidatedZone:
    """A refined zone with the Strong Point verdict attached."""

    # Passthrough from RefinedZone:
    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: WPattern | MPattern
    is_tradeable: bool
    rejection_reason: RejectionReason | None
    original_zone: Zone
    refined_zone: RefinedZone

    # Strong Point verdict:
    is_strong_point: bool
    validation_failures: list[ValidationFailure]
    bos_event: BosEvent | None  # the move-out BoS (if found)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def validate_strong_point(
    zone: RefinedZone,
    df: pd.DataFrame,
    bos_events: list[BosEvent],
    config: StrongPointConfig | None = None,
) -> ValidatedZone:
    """Run the three Strong Point gates against ``zone``.

    ``bos_events`` should be the output of :func:`bot.strategy.structure.detect_bos`
    (or equivalently ``StructureSnapshot.bos_events``) computed on the
    same DataFrame.
    """
    cfg = config or StrongPointConfig()

    # Short-circuit: zone failed the size filter upstream.
    if not zone.is_tradeable:
        return _build(zone, is_strong_point=False, failures=["NOT_TRADEABLE"])

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"df must have a '{col}' column")

    first_bos = _first_matching_bos(zone, bos_events)
    if first_bos is None:
        return _build(zone, is_strong_point=False, failures=["NO_BOS_YET"])

    n = len(df)
    if not 0 <= first_bos.bar_index < n:
        raise ValueError(
            f"BoS bar_index {first_bos.bar_index} out of df range (len={n})"
        )

    base_indices = _base_bar_indices(zone)
    for idx in base_indices:
        if not 0 <= idx < n:
            raise ValueError(f"base bar index {idx} out of df range (len={n})")

    impulse_bar = df.iloc[first_bos.bar_index]
    base_bars = [df.iloc[idx] for idx in base_indices]

    failures: list[ValidationFailure] = []

    if not _impulse_strong_enough(impulse_bar, cfg.impulse_min_body_ratio):
        failures.append("IMPULSE_TOO_WEAK")
    if not _impulse_in_correct_direction(impulse_bar, first_bos.direction):
        failures.append("IMPULSE_WRONG_DIRECTION")
    if not _base_is_compact(base_bars, impulse_bar, cfg.base_max_range_ratio):
        failures.append("BASE_NOT_COMPACT")

    is_strong = len(failures) == 0
    logger.debug(
        f"strong_point({zone.direction}): is_strong={is_strong} "
        f"failures={failures} bos@bar={first_bos.bar_index}"
    )
    return _build(
        zone,
        is_strong_point=is_strong,
        failures=failures,
        bos_event=first_bos,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _first_matching_bos(
    zone: RefinedZone, bos_events: list[BosEvent]
) -> BosEvent | None:
    """First BoS in the zone's direction occurring after pattern completion."""
    completion_idx = _pattern_completion_index(zone.source_pattern)
    target_direction = "UP" if zone.direction == "BUY" else "DOWN"
    matches = sorted(
        (
            e
            for e in bos_events
            if e.direction == target_direction and e.bar_index > completion_idx
        ),
        key=lambda e: e.bar_index,
    )
    return matches[0] if matches else None


def _pattern_completion_index(pattern: WPattern | MPattern) -> int:
    """The bar index after which the move-out should occur."""
    if isinstance(pattern, WPattern):
        return pattern.low2.index
    if isinstance(pattern, MPattern):
        return pattern.high2.index
    raise TypeError(f"unsupported pattern type: {type(pattern).__name__}")


def _base_bar_indices(zone: RefinedZone) -> tuple[int, int]:
    """The two pivot bars whose ranges define the base."""
    p = zone.source_pattern
    if isinstance(p, WPattern):
        return p.low1.index, p.low2.index
    if isinstance(p, MPattern):
        return p.high1.index, p.high2.index
    raise TypeError(f"unsupported pattern type: {type(p).__name__}")


def _impulse_strong_enough(bar: pd.Series, min_body_ratio: float) -> bool:
    """``body / range >= min_body_ratio``. Zero-range bars fail."""
    body = abs(float(bar["close"]) - float(bar["open"]))
    bar_range = float(bar["high"]) - float(bar["low"])
    if bar_range <= 0:
        return False
    return (body / bar_range) >= min_body_ratio


def _impulse_in_correct_direction(bar: pd.Series, bos_direction: str) -> bool:
    """For UP: ``close > open``; for DOWN: ``close < open``. Doji fails both."""
    o = float(bar["open"])
    c = float(bar["close"])
    if bos_direction == "UP":
        return c > o
    if bos_direction == "DOWN":
        return c < o
    raise ValueError(f"unknown BoS direction: {bos_direction}")


def _base_is_compact(
    base_bars: list[pd.Series],
    impulse_bar: pd.Series,
    max_ratio: float,
) -> bool:
    """Each base bar's range must be ``<= max_ratio * impulse_range``."""
    impulse_range = float(impulse_bar["high"]) - float(impulse_bar["low"])
    if impulse_range <= 0:
        # Already failing IMPULSE_TOO_WEAK; report base as also failing.
        return False
    threshold = max_ratio * impulse_range
    for bar in base_bars:
        bar_range = float(bar["high"]) - float(bar["low"])
        if bar_range > threshold:
            return False
    return True


def _build(
    zone: RefinedZone,
    is_strong_point: bool,
    failures: list[ValidationFailure],
    bos_event: BosEvent | None = None,
) -> ValidatedZone:
    return ValidatedZone(
        direction=zone.direction,
        top=zone.top,
        bottom=zone.bottom,
        formed_at=zone.formed_at,
        source_pattern=zone.source_pattern,
        is_tradeable=zone.is_tradeable,
        rejection_reason=zone.rejection_reason,
        original_zone=zone.original_zone,
        refined_zone=zone,
        is_strong_point=is_strong_point,
        validation_failures=failures,
        bos_event=bos_event,
    )
