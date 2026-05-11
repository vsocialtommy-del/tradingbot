"""Strong Point setup — break-and-close validation.

The v1 trade setup (spec ``docs/strategy_reference/README.md``,
Setup 1). Operates on RBR/DBR demand zones (BUY) and DBD/RBD supply
zones (SELL). For each pattern, the validator asks one question:

    Has price broken the nearest opposite-side structural swing with
    a body close, after the pattern formed?

* For a **BUY** zone (demand): "nearest opposite" = the nearest swing
  HIGH above the zone, after pattern formation. A bar's body must
  close above that swing's price.
* For a **SELL** zone (supply): mirror — nearest swing LOW below the
  zone, body close below.

If yes, the zone is a confirmed Strong Point and tradeable on every
subsequent retest until either TP1 hits or the zone is invalidated by
an opposite-side body close.

Output: :class:`ValidatedZone`. Carries the broken swing (for TP1)
and the SL anchor swing (the swing on the OPPOSITE side of the zone —
nearest high for SELL, nearest low for BUY). The engine reads
``sl_anchor_swing`` to position the stop; the spec says SL = 15-20
pips above/below this swing, **not** above/below the zone itself.

Reasons for failure (mutually exclusive, in priority order)
-----------------------------------------------------------

* ``NOT_TRADEABLE``           — upstream zone refinement rejected it
                                (size filter); short-circuit, don't
                                bother with the break check.
* ``NO_SWING_ABOVE`` /
  ``NO_SWING_BELOW``           — no structural swing exists on the
                                opposite side, so there's nothing to
                                break.
* ``NO_SL_ANCHOR``             — no structural swing on the same side
                                as the zone (need one for SL).
* ``NO_BREAK_YET``             — swing exists, no bar has body-closed
                                past it yet. The zone is still a
                                "pending" candidate — pipeline will
                                re-evaluate on subsequent bars.
* ``INVALIDATED``              — an opposite-side body close past the
                                zone happened before any valid break.
                                Zone is dead.

Re-entry semantics
------------------

Once ``is_strong_point=True``, the zone stays tradeable until either:

* A bar's body closes on the opposite side of the zone (BUY:
  body close below zone.bottom; SELL mirror) — invalidates.
* The engine reports TP1 hit — cycle complete.

This module doesn't track invalidation-after-validation; that's the
engine's job (it sees each bar's close). The validator just reports
"yes Strong Point as of bar X" or "not yet".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import Pattern
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Direction
from bot.strategy.zone_refinement import RefinedZone


ValidationFailure = Literal[
    "NOT_TRADEABLE",
    "NO_SWING_ABOVE",
    "NO_SWING_BELOW",
    "NO_SL_ANCHOR",
    "NO_BREAK_YET",
    "INVALIDATED",
]


@dataclass(frozen=True)
class StrongPointConfig:
    """Tunables for Strong Point validation.

    Most thresholds moved to ``PatternConfig`` (impulse / base
    criteria — that's pattern detection's job). What remains here:
    the SL buffer, applied to the anchor swing's price.
    """

    sl_buffer_points: float = 17.5
    """Buffer in price units (= $17.50 for XAUUSD) added to the anchor
    swing's price. Spec wording: '15-20 pips above the nearest high'
    for SELL; mirror for BUY. 17.5 is the midpoint of that band.
    """


@dataclass(frozen=True)
class ValidatedZone:
    """Output of :func:`validate_strong_point`."""

    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: Pattern
    refined_zone: RefinedZone

    is_strong_point: bool
    validation_failures: list[ValidationFailure]

    broken_swing: Swing | None
    """The swing whose break confirmed the Strong Point.
    BUY: nearest high above zone; SELL: nearest low below zone.
    None when not yet validated."""

    broken_at: pd.Timestamp | None
    """Bar time at which the body-close break was confirmed.
    None when not yet validated."""

    sl_anchor_swing: Swing | None
    """The OPPOSITE-side swing the SL pins to.
    BUY zone → nearest swing LOW below the zone (SL = anchor.price -
    sl_buffer_points).
    SELL zone → nearest swing HIGH above the zone (SL = anchor.price +
    sl_buffer_points).
    None when no such swing exists in the lookback window."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def validate_strong_point(
    refined: RefinedZone,
    df: pd.DataFrame,
    swings: list[Swing],
    config: StrongPointConfig | None = None,
) -> ValidatedZone:
    """Return a Strong Point verdict for ``refined``.

    Parameters
    ----------
    refined
        Output of :func:`zone_refinement.refine_zone`.
    df
        The same DataFrame the pattern was detected in. We read bar
        close prices (post-pattern bars) to find the break candidate.
    swings
        All confirmed swings in the df (typically from
        :func:`structure.analyze_structure`). We filter to swings
        relevant to this zone.
    """
    cfg = config or StrongPointConfig()
    failures: list[ValidationFailure] = []

    # 1. Short-circuit: zone failed the size filter upstream.
    if not refined.is_tradeable:
        failures.append("NOT_TRADEABLE")
        return _build_unvalidated(refined, failures)

    # 2. Find the SL anchor swing (same side as the zone — opposite
    #    direction from the trade). Without it we can't position SL.
    sl_anchor = _find_sl_anchor(refined, swings)
    if sl_anchor is None:
        failures.append("NO_SL_ANCHOR")
        return _build_unvalidated(refined, failures)

    # 3. Find the structural swing the break candidate must clear.
    break_target = _find_break_target(refined, swings)
    if break_target is None:
        failures.append(
            "NO_SWING_ABOVE" if refined.direction == "BUY"
            else "NO_SWING_BELOW"
        )
        return _build_unvalidated(refined, failures, sl_anchor=sl_anchor)

    # 4. Scan bars AFTER pattern formation for either:
    #    - a body close past break_target → Strong Point validated
    #    - an opposite-side body close past the zone → INVALIDATED first
    pattern_end_idx = refined.source_pattern.impulse_after.end_index
    break_outcome = _scan_for_break(
        refined, df, pattern_end_idx, break_target,
    )
    if break_outcome.invalidated_before_break:
        failures.append("INVALIDATED")
        return _build_unvalidated(refined, failures, sl_anchor=sl_anchor)
    if break_outcome.broken_at is None:
        failures.append("NO_BREAK_YET")
        return _build_unvalidated(refined, failures, sl_anchor=sl_anchor)

    # All gates passed — Strong Point confirmed.
    logger.debug(
        "Strong Point confirmed: {} zone {:.2f}-{:.2f} "
        "broke {} at {:.2f} (sl anchor {:.2f})",
        refined.direction, refined.bottom, refined.top,
        break_target.kind, break_target.price, sl_anchor.price,
    )
    return ValidatedZone(
        direction=refined.direction,
        top=refined.top,
        bottom=refined.bottom,
        formed_at=refined.formed_at,
        source_pattern=refined.source_pattern,
        refined_zone=refined,
        is_strong_point=True,
        validation_failures=[],
        broken_swing=break_target,
        broken_at=break_outcome.broken_at,
        sl_anchor_swing=sl_anchor,
    )


def compute_sl_price(
    validated: ValidatedZone, config: StrongPointConfig | None = None,
) -> float:
    """SL = anchor_swing.price ± buffer.

    Strategy-side helper for the engine. BUY zone: SL = anchor.price -
    buffer (anchor is a swing LOW below the zone). SELL zone: SL =
    anchor.price + buffer (anchor is a swing HIGH above the zone).
    """
    if validated.sl_anchor_swing is None:
        raise ValueError(
            "cannot compute SL for a zone without sl_anchor_swing "
            "(validation must have failed upstream)"
        )
    cfg = config or StrongPointConfig()
    anchor = validated.sl_anchor_swing
    if validated.direction == "BUY":
        return float(anchor.price - cfg.sl_buffer_points)
    return float(anchor.price + cfg.sl_buffer_points)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _BreakOutcome:
    broken_at: pd.Timestamp | None
    invalidated_before_break: bool


def _find_break_target(
    refined: RefinedZone, swings: list[Swing],
) -> Swing | None:
    """The OPPOSITE-side swing the break candidate must body-close past.

    BUY zone (demand): need to break a swing HIGH that exists ABOVE
    the zone. Of all qualifying swings, take the NEAREST — i.e. the
    LOWEST-priced high above (closest to zone top).

    SELL zone (supply): mirror — HIGHEST-priced low below zone.

    Only swings at-or-before pattern formation are considered (a
    swing formed AFTER the pattern can't be a structural target
    that existed when the pattern emerged).
    """
    pattern_end_idx = refined.source_pattern.impulse_after.end_index
    eligible = [s for s in swings if s.index <= pattern_end_idx]
    if refined.direction == "BUY":
        candidates = [
            s for s in eligible
            if s.kind == "HIGH" and s.price > refined.top
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.price)
    # SELL
    candidates = [
        s for s in eligible
        if s.kind == "LOW" and s.price < refined.bottom
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.price)


def _find_sl_anchor(
    refined: RefinedZone, swings: list[Swing],
) -> Swing | None:
    """The SAME-side swing the SL pins to.

    BUY zone: nearest swing LOW below zone — closest to zone bottom,
    i.e. HIGHEST-priced low below.

    SELL zone: nearest swing HIGH above zone — closest to zone top,
    i.e. LOWEST-priced high above.
    """
    pattern_end_idx = refined.source_pattern.impulse_after.end_index
    eligible = [s for s in swings if s.index <= pattern_end_idx]
    if refined.direction == "BUY":
        candidates = [
            s for s in eligible
            if s.kind == "LOW" and s.price < refined.bottom
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.price)
    # SELL
    candidates = [
        s for s in eligible
        if s.kind == "HIGH" and s.price > refined.top
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.price)


def _scan_for_break(
    refined: RefinedZone,
    df: pd.DataFrame,
    pattern_end_idx: int,
    break_target: Swing,
) -> _BreakOutcome:
    """Walk forward from pattern_end_idx + 1; report the first conclusive bar.

    On each bar, check in priority order:
      1. Invalidation (opposite-side body close past zone) → INVALIDATED
      2. Validation (body close past break_target) → Strong Point confirmed
    Stops on the first conclusive event. If we hit the end of df
    without either, returns pending state.
    """
    closes = df["close"].to_numpy()
    times = df.index
    n = len(closes)

    for i in range(pattern_end_idx + 1, n):
        bar_close = float(closes[i])
        if refined.direction == "BUY":
            if bar_close < refined.bottom:
                return _BreakOutcome(
                    broken_at=None, invalidated_before_break=True,
                )
            if bar_close > break_target.price:
                return _BreakOutcome(
                    broken_at=times[i], invalidated_before_break=False,
                )
        else:  # SELL
            if bar_close > refined.top:
                return _BreakOutcome(
                    broken_at=None, invalidated_before_break=True,
                )
            if bar_close < break_target.price:
                return _BreakOutcome(
                    broken_at=times[i], invalidated_before_break=False,
                )

    return _BreakOutcome(broken_at=None, invalidated_before_break=False)


def _build_unvalidated(
    refined: RefinedZone,
    failures: list[ValidationFailure],
    *, sl_anchor: Swing | None = None,
) -> ValidatedZone:
    """ValidatedZone for any non-success path."""
    return ValidatedZone(
        direction=refined.direction,
        top=refined.top,
        bottom=refined.bottom,
        formed_at=refined.formed_at,
        source_pattern=refined.source_pattern,
        refined_zone=refined,
        is_strong_point=False,
        validation_failures=failures,
        broken_swing=None,
        broken_at=None,
        sl_anchor_swing=sl_anchor,
    )
