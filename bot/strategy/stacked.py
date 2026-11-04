"""XAUTISM stacked (two-zone) layering (Nov 2026, step 6b).

When another fresh zone in the same direction sits just past the zone
being traded, the setup is split across both zones:

* Layer 1: 0.01 at the first zone's edge (first touch, as normal).
* Layer 2: the main size as a LIMIT at the outer zone's near edge.
* One SL for both: 30 % of the outer zone past its far edge.

The outer zone can be an M5 zone or an M1 zone. This is the only place
M1 zones are used.

Example (29 Oct, sell): M5 zone 4018.0-4025.5, M1 zone 4030.4-4033.7
above it, gap 4.9 points → 0.01 sell at 4018.0, main sell limit at
4030.4, SL 30 % of the M1 zone above 4033.7.

"Just past" = the gap between the first zone's far edge and the outer
zone's near edge is between 0 and ``max_gap_points`` (10 points).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from bot.strategy.strong_point import ValidatedZone

Timeframe = Literal["M5", "M1"]


@dataclass(frozen=True)
class StackedConfig:
    enabled: bool = True
    max_gap_points: float = 10.0
    """Max distance between the first zone's far edge and the outer
    zone's near edge (10 points = $10 on gold)."""
    use_m1_zones: bool = True
    allow_m5_outer: bool = True
    """False = only an M1 zone can be the outer zone (Tommy's original
    rule: an M5 zone with an M1 zone right next to it)."""
    m1_ohlc_count: int = 5000
    """M1 bars scanned for outer zones (5000 ≈ 3.5 trading days)."""
    sl_zone_pct: float = 0.30
    """SL = this fraction of the OUTER zone past its far edge."""


@dataclass(frozen=True)
class StackedPlan:
    outer_timeframe: Timeframe
    outer_top: float
    outer_bottom: float
    gap_points: float
    outer_entry: float
    """Layer 2 limit price: outer top for buys, outer bottom for sells."""
    outer_far_edge: float
    sl: float


def find_outer_zone(
    first: ValidatedZone,
    candidates: Iterable[tuple[Timeframe, ValidatedZone]],
    config: StackedConfig | None = None,
) -> tuple[Timeframe, ValidatedZone, float] | None:
    """Nearest fresh same-direction zone just past ``first``.

    Returns ``(timeframe, zone, gap)`` or None. Ties on gap go to M5.
    """
    cfg = config or StackedConfig()
    best: tuple[Timeframe, ValidatedZone, float] | None = None
    for tf, z in candidates:
        if tf == "M5" and not cfg.allow_m5_outer:
            continue
        if z.direction != first.direction:
            continue
        if not (z.is_strong_point and z.refined_zone.is_tradeable):
            continue
        if z.top == first.top and z.bottom == first.bottom:
            continue  # the first zone itself
        if first.direction == "BUY":
            gap = first.bottom - z.top   # outer sits below
        else:
            gap = z.bottom - first.top   # outer sits above
        if gap < 0 or gap > cfg.max_gap_points:
            continue
        if (
            best is None
            or gap < best[2]
            or (gap == best[2] and tf == "M5" and best[0] == "M1")
        ):
            best = (tf, z, gap)
    return best


def stacked_plan(
    first: ValidatedZone,
    candidates: Iterable[tuple[Timeframe, ValidatedZone]],
    config: StackedConfig | None = None,
) -> StackedPlan | None:
    cfg = config or StackedConfig()
    found = find_outer_zone(first, candidates, cfg)
    if found is None:
        return None
    tf, outer, gap = found
    size = outer.top - outer.bottom
    if first.direction == "BUY":
        entry, far = outer.top, outer.bottom
        sl = outer.bottom - cfg.sl_zone_pct * size
    else:
        entry, far = outer.bottom, outer.top
        sl = outer.top + cfg.sl_zone_pct * size
    return StackedPlan(
        outer_timeframe=tf,
        outer_top=float(outer.top),
        outer_bottom=float(outer.bottom),
        gap_points=float(gap),
        outer_entry=float(entry),
        outer_far_edge=float(far),
        sl=round(float(sl), 2),
    )
