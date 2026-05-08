"""Imbalance Zone qualification — count failed approaches without a tap.

Per spec Section 3.2, a Strong Point demand / supply zone becomes an
**Imbalance Zone** when price has approached it multiple times without
ever entering. The intuition is "pent-up energy" — orders accumulate
above (or below) the zone edge, and when price finally taps the zone
the reaction tends to be sharper than a first-touch Strong Point.

Last module in Phase B. Pipeline so far::

    Zone (zone_marking)
       → RefinedZone (zone_refinement)
       → ValidatedZone (strong_point)
       → ImbalanceZone (this module)

State machine
-------------
For each bar after the pattern's completion bar we track:

    IDLE        no current approach in progress
    APPROACHING price has entered the approach band but not yet retreated

Bar-by-bar handling (BUY zone — SELL is mirror):

1. **Tap takes precedence.** If ``bar.low <= zone.top`` we set
   ``is_tapped = True`` and stop tracking. This handles the gap-through
   case automatically — a bar that vaults the approach band and
   probes the zone is a tap, not an approach.
2. **IDLE → APPROACHING** when ``zone.top < bar.low <= zone.top + approach_distance``.
3. **APPROACHING** stays APPROACHING while neither tap nor full retreat
   has happened. Tracks the closest price seen.
4. **APPROACHING → IDLE (event recorded)** when
   ``bar.low >= zone.top + approach_distance + retreat_distance``.

The "complete" requirement (retreat) means an approach is only counted
once it's clearly finished — preventing double-counting of price
oscillating within the band.

Boundary handling
-----------------
* Outer edge of the approach band: **inclusive**
  (``bar.low <= zone.top + approach_distance`` for BUY).
* Zone edge: **inclusive on the tap side** —
  ``bar.low == zone.top`` is a tap, not an approach. Symmetric for SELL.
* Retreat threshold: **inclusive** (``bar.low >= retreat_threshold``).

Tapped semantics
----------------
The user-facing rule is "all approach counting resets and the zone is
no longer Imbalance-eligible". We interpret this as:

* ``is_imbalance = False`` once tapped (regardless of historical count).
* ``approach_count`` and ``approach_events`` retain the count of
  *completed* approaches before the tap — useful for analytics.
* ``qualified_at`` is preserved if the zone qualified as Imbalance
  *before* the tap; this is a historical record, not a current verdict.

Pure functional
---------------
The function takes the full DataFrame and computes from scratch every
call. No persisted approach history, no caching. Works fine for
backtest replays and the bot's per-iteration recompute (worst case ~50
bars to walk; negligible vs everything else the bot does each loop).
If profile data later shows it matters, an incremental version that
caches the last state can replace this one without API change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.structure import BosEvent
from bot.strategy.strong_point import ValidatedZone, ValidationFailure
from bot.strategy.zone_marking import Direction, Zone
from bot.strategy.zone_refinement import RefinedZone, RejectionReason

ApproachState = Literal["IDLE", "APPROACHING"]


@dataclass(frozen=True)
class ApproachEvent:
    """One completed approach: entered the band, then retreated."""

    start_bar_index: int
    completed_bar_index: int
    start_time: pd.Timestamp
    completed_time: pd.Timestamp
    closest_price: float
    distance_from_zone: float


@dataclass(frozen=True)
class ImbalanceConfig:
    """Tunables for Imbalance qualification.

    ``imbalance_approach_distance`` and ``imbalance_approach_threshold``
    are seeded in ``bot_config`` (spec Section 13). The retreat distance
    is currently hard-coded; can be added to ``bot_config`` later if
    backtest tuning shows it matters.
    """

    imbalance_approach_distance: float = 7.5
    imbalance_retreat_distance: float = 5.0
    imbalance_approach_threshold: int = 2


@dataclass(frozen=True)
class ImbalanceZone:
    """A Strong Point zone with the Imbalance verdict attached."""

    # Passthrough from ValidatedZone:
    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: WPattern | MPattern
    is_tradeable: bool
    rejection_reason: RejectionReason | None
    original_zone: Zone
    refined_zone: RefinedZone
    is_strong_point: bool
    validation_failures: list[ValidationFailure]
    bos_event: BosEvent | None
    validated_zone: ValidatedZone

    # Imbalance verdict:
    approach_count: int
    is_imbalance: bool
    approach_events: list[ApproachEvent]
    qualified_at: pd.Timestamp | None
    is_tapped: bool
    tapped_at: pd.Timestamp | None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def track_imbalance(
    zone: ValidatedZone,
    df: pd.DataFrame,
    config: ImbalanceConfig | None = None,
) -> ImbalanceZone:
    """Walk bars after pattern formation, count completed approaches."""
    cfg = config or ImbalanceConfig()

    # Skip everything if zone isn't a Strong Point — Imbalance requires SP.
    if not zone.is_strong_point:
        return _build(
            zone,
            approach_count=0,
            is_imbalance=False,
            approach_events=[],
            qualified_at=None,
            is_tapped=False,
            tapped_at=None,
        )

    for col in ("high", "low"):
        if col not in df.columns:
            raise ValueError(f"df must have a '{col}' column")

    formation_idx = _formation_index(zone)
    n = len(df)
    if not 0 <= formation_idx < n:
        raise ValueError(
            f"formation index {formation_idx} out of df range (len={n})"
        )

    events, is_tapped, tapped_at_idx = _detect_approaches(
        zone, df, formation_idx, cfg
    )

    approach_count = len(events)
    qualified_at: pd.Timestamp | None = None
    if approach_count >= cfg.imbalance_approach_threshold:
        # Use the completion time of the threshold-th approach.
        qualified_at = events[cfg.imbalance_approach_threshold - 1].completed_time

    is_imbalance = (
        approach_count >= cfg.imbalance_approach_threshold and not is_tapped
    )
    tapped_at = df.index[tapped_at_idx] if tapped_at_idx is not None else None

    logger.debug(
        f"imbalance({zone.direction}): approaches={approach_count} "
        f"is_imbalance={is_imbalance} tapped={is_tapped} "
        f"qualified_at={qualified_at}"
    )

    return _build(
        zone,
        approach_count=approach_count,
        is_imbalance=is_imbalance,
        approach_events=events,
        qualified_at=qualified_at,
        is_tapped=is_tapped,
        tapped_at=tapped_at,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _formation_index(zone: ValidatedZone) -> int:
    p = zone.source_pattern
    if isinstance(p, WPattern):
        return p.low2.index
    if isinstance(p, MPattern):
        return p.high2.index
    raise TypeError(f"unsupported pattern type: {type(p).__name__}")


def _detect_approaches(
    zone: ValidatedZone,
    df: pd.DataFrame,
    formation_idx: int,
    cfg: ImbalanceConfig,
) -> tuple[list[ApproachEvent], bool, int | None]:
    """The state-machine walk. Returns (events, is_tapped, tapped_at_idx)."""
    if zone.direction == "BUY":
        zone_edge = zone.top
        retreat_threshold = (
            zone_edge + cfg.imbalance_approach_distance + cfg.imbalance_retreat_distance
        )

        def in_zone(bar: pd.Series) -> bool:
            return float(bar["low"]) <= zone_edge

        def in_band(bar: pd.Series) -> bool:
            low = float(bar["low"])
            return zone_edge < low <= zone_edge + cfg.imbalance_approach_distance

        def moved_away(bar: pd.Series) -> bool:
            return float(bar["low"]) >= retreat_threshold

        def get_extreme(bar: pd.Series) -> float:
            return float(bar["low"])

        def is_closer(new: float, prev: float) -> bool:
            return new < prev

    else:  # SELL
        zone_edge = zone.bottom
        retreat_threshold = (
            zone_edge - cfg.imbalance_approach_distance - cfg.imbalance_retreat_distance
        )

        def in_zone(bar: pd.Series) -> bool:
            return float(bar["high"]) >= zone_edge

        def in_band(bar: pd.Series) -> bool:
            high = float(bar["high"])
            return zone_edge - cfg.imbalance_approach_distance <= high < zone_edge

        def moved_away(bar: pd.Series) -> bool:
            return float(bar["high"]) <= retreat_threshold

        def get_extreme(bar: pd.Series) -> float:
            return float(bar["high"])

        def is_closer(new: float, prev: float) -> bool:
            return new > prev

    state: ApproachState = "IDLE"
    events: list[ApproachEvent] = []
    is_tapped = False
    tapped_at_idx: int | None = None
    closest_price: float | None = None
    approach_start_idx: int | None = None

    for i in range(formation_idx + 1, len(df)):
        bar = df.iloc[i]

        # Tap takes absolute precedence — handles the gap-through case.
        if in_zone(bar):
            is_tapped = True
            tapped_at_idx = i
            break

        if state == "IDLE":
            if in_band(bar):
                state = "APPROACHING"
                approach_start_idx = i
                closest_price = get_extreme(bar)

        elif state == "APPROACHING":
            current = get_extreme(bar)
            assert closest_price is not None  # for type narrowing
            if is_closer(current, closest_price):
                closest_price = current

            if moved_away(bar):
                assert approach_start_idx is not None
                events.append(
                    ApproachEvent(
                        start_bar_index=approach_start_idx,
                        completed_bar_index=i,
                        start_time=df.index[approach_start_idx],
                        completed_time=df.index[i],
                        closest_price=closest_price,
                        distance_from_zone=abs(closest_price - zone_edge),
                    )
                )
                state = "IDLE"
                closest_price = None
                approach_start_idx = None

    return events, is_tapped, tapped_at_idx


def _build(
    zone: ValidatedZone,
    *,
    approach_count: int,
    is_imbalance: bool,
    approach_events: list[ApproachEvent],
    qualified_at: pd.Timestamp | None,
    is_tapped: bool,
    tapped_at: pd.Timestamp | None,
) -> ImbalanceZone:
    return ImbalanceZone(
        direction=zone.direction,
        top=zone.top,
        bottom=zone.bottom,
        formed_at=zone.formed_at,
        source_pattern=zone.source_pattern,
        is_tradeable=zone.is_tradeable,
        rejection_reason=zone.rejection_reason,
        original_zone=zone.original_zone,
        refined_zone=zone.refined_zone,
        is_strong_point=zone.is_strong_point,
        validation_failures=zone.validation_failures,
        bos_event=zone.bos_event,
        validated_zone=zone,
        approach_count=approach_count,
        is_imbalance=is_imbalance,
        approach_events=approach_events,
        qualified_at=qualified_at,
        is_tapped=is_tapped,
        tapped_at=tapped_at,
    )
