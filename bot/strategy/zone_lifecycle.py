"""Zone lifecycle — state machine + per-bar transition detectors.

Persisted states (the in-memory pre-states FRESH / TRADEABLE are
upstream of the first DB write and live only in pipeline output)::

    CONFIRMED ─► ACTIVE ─► CONSUMED ─► VIOLATED ─► FLIPPED
        │          │          │            │
        └──────────┴──────────┴────────────┘     (multiple entry points)

Allowed transitions (everything else raises ``IllegalZoneTransitionError``)::

    CONFIRMED → {ACTIVE, CONSUMED, VIOLATED}
    ACTIVE    → {CONSUMED, VIOLATED}
    CONSUMED  → {VIOLATED}
    VIOLATED  → {FLIPPED}
    FLIPPED   → (terminal)

Detectors
---------

* :func:`check_consumption` — does the bar's ``[low, high]`` overlap
  the zone's ``[bottom, top]``? Q1 says **any** touch consumes
  (fill-agnostic, no follow-through threshold).

* :func:`check_violation` — has a bar **body-closed** past the
  wrong-side zone bound? For a BUY zone that's ``close < zone.bottom``;
  SELL mirror. Wick-only pokes don't count (consistent with the
  Strong-Point "body close" semantics in :mod:`structure`).

* :func:`check_flip` — given a VIOLATED zone, scan forward for a
  body close past the nearest opposite-side swing (BoS). Per design
  decision Q2, the BoS may happen on the **same bar** or **any
  subsequent bar** — no time window. The nearest swing is
  **recomputed from current structure** at flip-check time (option B
  in the design doc) rather than reusing the zone's original
  ``sl_anchor_swing``: institutional reading says "the most recent
  swing low at the moment of the break", and structure typically
  shifts between pattern formation and the eventual violation.

No I/O here — the orchestrator (``bot.main``) is responsible for
pulling DataFrames + persisting the resulting status updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.structure import (
    Swing,
    analyze_structure,
    StructureConfig,
)
from bot.strategy.zone_marking import Direction


# --------------------------------------------------------------------------- #
# Public types
# --------------------------------------------------------------------------- #


ZoneStatus = Literal[
    "CONFIRMED", "ACTIVE", "CONSUMED", "VIOLATED", "FLIPPED",
]


_VALID_ZONE_TRANSITIONS: dict[ZoneStatus, frozenset[ZoneStatus]] = {
    "CONFIRMED": frozenset({"ACTIVE", "CONSUMED", "VIOLATED"}),
    "ACTIVE":    frozenset({"CONSUMED", "VIOLATED"}),
    "CONSUMED":  frozenset({"VIOLATED"}),
    "VIOLATED":  frozenset({"FLIPPED"}),
    # PR #38 (SnD Flip trading): FLIPPED is no longer terminal. A
    # freshly-flipped zone can be traded in ``flipped_direction``;
    # placing a setup transitions FLIPPED → ACTIVE. The subsequent
    # ACTIVE → CONSUMED path is unchanged.
    "FLIPPED":   frozenset({"ACTIVE"}),
}


TERMINAL_ZONE_STATUSES: frozenset[ZoneStatus] = frozenset()
"""Zones in this status never transition again.

Empty since PR #38: every persisted status has at least one outgoing
edge. A zone that's been flipped, traded, and consumed CAN in theory
re-flip (CONSUMED → VIOLATED → FLIPPED), though this is rare in
practice."""


SKIP_NEW_SETUP_STATUSES: frozenset[ZoneStatus] = frozenset(
    {"CONSUMED", "VIOLATED", "FLIPPED"}
)
"""If a zone with overlapping bounds + same direction is in any of these,
a freshly-detected Strong Point should be skipped (re-trade guard)."""


@dataclass(frozen=True)
class ZoneRef:
    """The minimal view of a zone the lifecycle detectors need.

    Keeps :mod:`zone_lifecycle` decoupled from ``supabase_logger.Zone``
    (read model) and the in-memory ``ValidatedZone`` — both can be
    projected into a :class:`ZoneRef` at call sites. The lifecycle
    logic doesn't need anything else (FK ids, formed_at, etc. live in
    the caller).
    """

    direction: Direction
    top: float
    bottom: float


class IllegalZoneTransitionError(ValueError):
    """Raised when a transition is not in ``_VALID_ZONE_TRANSITIONS``."""


@dataclass(frozen=True)
class FlipResult:
    """Output of :func:`check_flip`."""

    flipped: bool
    new_direction: Direction | None
    """Opposite of the zone's original direction; populated iff ``flipped``."""
    bos_swing: Swing | None
    """The swing whose body-close break confirmed the flip."""
    broken_at: pd.Timestamp | None


# --------------------------------------------------------------------------- #
# Transition validation
# --------------------------------------------------------------------------- #


def validate_zone_transition(current: ZoneStatus, new: ZoneStatus) -> None:
    """Raise :class:`IllegalZoneTransitionError` if ``current → new`` is illegal."""
    allowed = _VALID_ZONE_TRANSITIONS.get(current, frozenset())
    if new not in allowed:
        raise IllegalZoneTransitionError(
            f"invalid zone transition: {current} → {new}. "
            f"valid next: "
            f"{sorted(allowed) if allowed else 'none (terminal)'}"
        )


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #


def check_consumption(zone: ZoneRef, bar_high: float, bar_low: float) -> bool:
    """True iff the bar's ``[low, high]`` overlaps the zone bounds.

    Q1: any wick or body that enters ``[zone.bottom, zone.top]``
    consumes the zone. Fill-agnostic; doesn't matter whether a layer
    actually filled. Endpoints inclusive (a wick touching exactly
    zone.bottom counts).
    """
    return bar_low <= zone.top and bar_high >= zone.bottom


def check_violation(
    zone: ZoneRef, bar_close: float,
) -> bool:
    """True iff a body close has passed the wrong-side zone bound.

    * BUY zone (demand):  close < zone.bottom → VIOLATED
    * SELL zone (supply): close > zone.top    → VIOLATED

    Body-only (wick pokes don't qualify) — same convention as
    :mod:`bot.strategy.structure` BoS detection.
    """
    if zone.direction == "BUY":
        return bar_close < zone.bottom
    return bar_close > zone.top


def flipped_zone_body_broken_since_flip(
    zone_top: float,
    zone_bottom: float,
    flipped_direction: Direction,
    flipped_at: pd.Timestamp,
    df: pd.DataFrame,
) -> bool:
    """True iff any bar after ``flipped_at`` body-closed past the wrong side.

    Pre-trade safety net for the FLIPPED → ACTIVE path (PR #38):
    symmetric to :func:`bot.strategy.strong_point._zone_body_broken_since_formation`,
    but keyed off ``flipped_at`` instead of pattern formation. Once a
    zone has been flipped, any subsequent body close past the wrong
    side of the (new) direction kills the trade opportunity — price
    has invalidated the flip premise.

    For a BUY-direction flipped zone (originally SELL): reject if any
    bar's ``close < zone.bottom``. SELL mirror: reject if any
    ``close > zone.top``. Wick-only pokes don't count (same body-close
    convention as :func:`check_violation`).

    Returns ``False`` when there are no bars after ``flipped_at`` —
    a freshly-flipped zone with no follow-through bars yet is
    tradeable. The lookup is strict ``time > flipped_at`` so the flip
    bar itself isn't re-checked.
    """
    times = df.index
    start_idx: int | None = None
    for i, t in enumerate(times):
        if t > flipped_at:
            start_idx = i
            break
    if start_idx is None:
        return False

    closes = df["close"].to_numpy()
    for i in range(start_idx, len(df)):
        bar_close = float(closes[i])
        if flipped_direction == "BUY" and bar_close < zone_bottom:
            return True
        if flipped_direction == "SELL" and bar_close > zone_top:
            return True
    return False


def check_flip(
    zone: ZoneRef,
    df: pd.DataFrame,
    violation_index: int,
    *,
    structure_config: StructureConfig | None = None,
) -> FlipResult:
    """Decide whether a VIOLATED zone has now FLIPPED.

    The flip is confirmed when, on or after ``violation_index``, the
    bar's body close passes the **nearest opposite-side swing**
    relative to the violation direction:

    * BUY zone (violated downward): need a body close BELOW the
      nearest swing LOW that exists at or before each candidate bar.
    * SELL zone (violated upward):  body close ABOVE the nearest
      swing HIGH.

    Structure is **recomputed** at flip-check time (option B from the
    design doc): structure can shift between pattern formation and the
    eventual violation, and the institutional reading is "break the
    nearest swing as of the break bar." The zone's original
    ``sl_anchor_swing`` is intentionally ignored here.

    Returns
    -------
    FlipResult
        ``flipped=False`` when no qualifying close has been found yet
        (caller should re-check on later bars). Other fields are
        populated iff ``flipped=True``.
    """
    if violation_index < 0 or violation_index >= len(df):
        raise ValueError(
            f"violation_index {violation_index} out of df range "
            f"(0..{len(df)-1})"
        )
    snapshot = analyze_structure(df, structure_config or StructureConfig())
    swings = list(snapshot.swings)
    closes = df["close"].to_numpy()
    times = df.index

    # For each bar from the violation forward, look for a body close
    # past the nearest opposite-side swing whose index is < that bar.
    # "Nearest" = closest to zone in price, on the right side, as of
    # the bar being checked.
    for i in range(violation_index, len(df)):
        bar_close = float(closes[i])
        target = _nearest_bos_target(zone, swings, up_to_index=i)
        if target is None:
            continue
        if zone.direction == "BUY":
            # Violation is downward; BoS is also downward — close
            # must be BELOW the swing LOW's price.
            if bar_close < target.price:
                return FlipResult(
                    flipped=True,
                    new_direction="SELL",
                    bos_swing=target,
                    broken_at=times[i],
                )
        else:  # SELL zone, upward violation, upward BoS
            if bar_close > target.price:
                return FlipResult(
                    flipped=True,
                    new_direction="BUY",
                    bos_swing=target,
                    broken_at=times[i],
                )

    return FlipResult(
        flipped=False, new_direction=None,
        bos_swing=None, broken_at=None,
    )


def _nearest_bos_target(
    zone: ZoneRef, swings: list[Swing], *, up_to_index: int,
) -> Swing | None:
    """The nearest opposite-side swing to BoS through, ≤ ``up_to_index``.

    BUY zone (looking for downward BoS): nearest swing LOW BELOW
    ``zone.bottom`` — closest to zone bottom (= highest-priced low).

    SELL zone (looking for upward BoS): nearest swing HIGH ABOVE
    ``zone.top`` — closest to zone top (= lowest-priced high).
    """
    if zone.direction == "BUY":
        eligible = [
            s for s in swings
            if s.kind == "LOW"
            and s.price < zone.bottom
            and s.index <= up_to_index
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda s: s.price)
    eligible = [
        s for s in swings
        if s.kind == "HIGH"
        and s.price > zone.top
        and s.index <= up_to_index
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda s: s.price)


# --------------------------------------------------------------------------- #
# Re-trade dedup
# --------------------------------------------------------------------------- #


def zone_bounds_overlap(
    a: ZoneRef, b: ZoneRef, *, tolerance: float = 0.5,
) -> bool:
    """True iff two zones cover effectively the same price band + direction.

    Used for the new-setup dedup guard: if a zone in CONSUMED /
    VIOLATED / FLIPPED already exists with overlapping bounds in the
    same direction, the freshly-detected Strong Point should be
    skipped — re-arming a CONSUMED zone is explicitly disallowed
    (design decision Q3). ``tolerance`` (default 0.5 price points)
    is the maximum allowed gap that still counts as overlap; raw
    overlap requires no slack, two zones with a 0.3-point gap also
    qualify, two with a 0.6-point gap do not. Tune as data accumulates.
    """
    if a.direction != b.direction:
        return False
    return (
        a.top + tolerance >= b.bottom
        and a.bottom - tolerance <= b.top
    )


# --------------------------------------------------------------------------- #
# Logging helper
# --------------------------------------------------------------------------- #


def log_transition(
    zone_id: str, current: ZoneStatus, new: ZoneStatus,
    *, reason: str = "",
) -> None:
    """Single-line INFO log; useful from main loop where multiple zones
    transition on the same bar.
    """
    suffix = f" ({reason})" if reason else ""
    logger.info(
        f"zone {zone_id} transitioned: {current} → {new}{suffix}"
    )
