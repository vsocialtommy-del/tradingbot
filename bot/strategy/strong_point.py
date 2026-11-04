"""Strong Point gate — loosened entry rules (May 2026 refinement).

Background
----------
Earlier versions required a **break-and-close** confirmation before a
zone became tradeable: a body close past the nearest opposite-side
structural swing. That waited for a "Strong Point" structural signal
before letting a setup fire. It also pinned SL to a structural swing
(``sl_anchor_swing``).

The user has loosened both rules: trade on the **first retest** of any
fresh size-filter-passing zone, with SL pinned to the zone bound
itself plus a fixed buffer. The TP1 target moves out of this module
into :mod:`bot.strategy.tp_target` (nearest local peak/low).

What this module now does
-------------------------

1. :func:`validate_strong_point` is a **passthrough** that returns
   ``is_strong_point=True`` for every zone whose size filter passed
   (:attr:`RefinedZone.is_tradeable`), **except** when the zone has
   already been body-closed past in the wrong direction since
   pattern formation. That last check is a minimal safety net:
   ``zones`` rows that have transitioned to VIOLATED are caught by
   the dedup pre-flight (PR #35), but a freshly-detected pattern
   may have been broken between formation and detection without
   ever being persisted — the lifecycle system can't catch that.

2. :func:`compute_sl_price` derives SL straight from the zone bound:
   ``zone.bottom - sl_buffer_points`` for BUY, mirror for SELL.

What this module does NOT do
----------------------------
- It does not consume :class:`~bot.strategy.structure.Swing` lists
  any more — the BoS target / SL anchor logic is gone.
- It does not compute TP1 — that lives in
  :mod:`bot.strategy.tp_target` so the orchestrator can decide
  whether a fresh zone has a tradeable TP1 before committing to
  order placement.

``ValidatedZone`` shape
-----------------------
Kept stable for downstream type-stability — ``broken_swing``,
``broken_at``, and ``sl_anchor_swing`` are all permanently ``None``
in the loosened flow but the fields stay on the dataclass so a
future tightening doesn't churn the schema. ``validation_failures``
now carries at most one of ``NOT_TRADEABLE`` or
``ZONE_VIOLATED_BEFORE_RETEST``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import Pattern
from bot.strategy.structure import Swing, detect_swings
from bot.strategy.zone_marking import Direction
from bot.strategy.zone_refinement import RefinedZone


ValidationFailure = Literal[
    "NOT_TRADEABLE",
    "ZONE_VIOLATED_BEFORE_RETEST",
    "PATTERN_NOT_ALLOWED",
    "NO_BOS",
    "ZONE_USED_SINCE_BOS",
]


@dataclass(frozen=True)
class StrongPointConfig:
    """Tunables for the (now minimal) Strong Point gate."""

    sl_buffer_points: float = 17.5
    """Buffer in price units below the zone bottom (BUY) or above the
    zone top (SELL). Applied directly to the zone bound — no swing
    anchor. Spec wording: '$17.50 below the zone for BUY' (mirror SELL).
    """

    # ---- XAUTISM zone rules (Nov 2026) ----------------------------------
    allowed_pattern_types: tuple[str, ...] = ("DBR", "RBD")
    """Only reversal bases are traded. RBR / DBD continuation bases are
    weaker and rejected with ``PATTERN_NOT_ALLOWED``. Pass all four to
    restore the old behaviour."""

    require_bos: bool = True
    """The move out of the base must body-close past the last swing
    before the base (swing HIGH for a DBR, swing LOW for an RBD). That
    swing is stored on ``broken_swing`` and is the TP1 level; the
    breaking bar's time is ``broken_at`` (the zone's birth)."""

    bos_swing_strength: int = 2
    bos_beyond_wick: bool = False
    """Vertex zones: the BOS candle must CLOSE beyond the wick of the high
    (buy) / low (sell) before the zone, not just beyond its body close.
    TP stays at the body level."""
    """Swing pivot strength on closes (the line-chart W / M points)."""

    require_fresh_since_bos: bool = True
    """Only the first retest is traded: if any bar after the BOS bar has
    already traded into the zone edge (BUY: low <= top, SELL:
    high >= bottom) the zone is used (``ZONE_USED_SINCE_BOS``). Bars
    between the base and the BOS don't count. A near miss that never
    reaches the edge leaves the zone fresh."""


@dataclass(frozen=True)
class ValidatedZone:
    """Output of :func:`validate_strong_point`.

    Field surface kept stable across the methodology change so
    downstream consumers (``order_manager``, ``main.py``, dashboard)
    don't need to fork their typing. In the loosened flow,
    ``broken_swing`` / ``broken_at`` / ``sl_anchor_swing`` are always
    ``None``.
    """

    direction: Direction
    top: float
    bottom: float
    formed_at: pd.Timestamp
    source_pattern: Pattern
    refined_zone: RefinedZone

    is_strong_point: bool
    validation_failures: list[ValidationFailure]

    broken_swing: Swing | None
    """Always ``None`` in the loosened flow. Kept for shape stability."""

    broken_at: pd.Timestamp | None
    """Always ``None`` in the loosened flow."""

    sl_anchor_swing: Swing | None
    """Always ``None`` in the loosened flow. SL is zone-bound based;
    see :func:`compute_sl_price`."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def validate_strong_point(
    refined: RefinedZone,
    df: pd.DataFrame,
    config: StrongPointConfig | None = None,
    *,
    swings: list[Swing] | None = None,
) -> ValidatedZone:
    """Decide whether ``refined`` is currently tradeable.

    Two gates only:

    1. Size filter (delegated to :func:`refine_zone` upstream).
    2. Body-break safety: if any bar **after** pattern formation has
       already body-closed past the wrong-side zone bound, the zone
       is dead and we won't enter on retest. This is the minimal
       protection against trading "obviously broken" zones whose
       persisted state hasn't been written yet (so the lifecycle
       dedup can't catch them).

    No break-and-close confirmation. No SL anchor lookup. No swing
    list. Pure pass / fail.
    """
    cfg = config or StrongPointConfig()
    failures: list[ValidationFailure] = []

    if not refined.is_tradeable:
        failures.append("NOT_TRADEABLE")
        return _build_unvalidated(refined, failures)

    ptype = _pattern_code(refined.source_pattern)
    if ptype not in cfg.allowed_pattern_types:
        failures.append("PATTERN_NOT_ALLOWED")
        return _build_unvalidated(refined, failures)

    if _zone_body_broken_since_formation(refined, df):
        failures.append("ZONE_VIOLATED_BEFORE_RETEST")
        return _build_unvalidated(refined, failures)

    broken_swing: Swing | None = None
    broken_at: pd.Timestamp | None = None
    if cfg.require_bos:
        bos = find_bos(
            refined, df, swing_strength=cfg.bos_swing_strength, swings=swings,
            beyond_wick=cfg.bos_beyond_wick,
        )
        if bos is None:
            failures.append("NO_BOS")
            return _build_unvalidated(refined, failures)
        broken_swing, bos_idx = bos
        broken_at = df.index[bos_idx]
        if cfg.require_fresh_since_bos and _zone_touched_after(
            refined, df, bos_idx,
        ):
            failures.append("ZONE_USED_SINCE_BOS")
            return _build_unvalidated(refined, failures)

    logger.debug(
        "Strong Point (passthrough): {} zone {:.2f}-{:.2f} tradeable",
        refined.direction, refined.bottom, refined.top,
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
        broken_swing=broken_swing,
        broken_at=broken_at,
        sl_anchor_swing=None,
    )


def find_bos(
    refined: RefinedZone,
    df: pd.DataFrame,
    *,
    swing_strength: int = 2,
    swings: list[Swing] | None = None,
    beyond_wick: bool = False,
) -> tuple[Swing, int] | None:
    """Find the first break of structure out of the base.

    The reference is the **last swing before the base** on closes (the
    line-chart point the move into the base came from):

    * BUY (DBR): last swing HIGH before ``base.start_index``; BOS is the
      first bar after the base whose close is above it.
    * SELL (RBD): last swing LOW before the base; BOS is the first bar
      after the base whose close is below it.

    Returns ``(swing, bos_bar_index)`` or ``None`` when there is no
    reference swing or the level has not been broken yet.
    """
    pattern = refined.source_pattern
    base_start = pattern.base.start_index
    base_end = pattern.base.end_index
    want = "HIGH" if refined.direction == "BUY" else "LOW"
    foot = getattr(pattern, "foot_index", None)
    if foot is not None and 0 <= foot < len(df):
        # Vertex pattern: the BOS level is the foot of the leg in.
        all_swings = [Swing(index=foot, time=df.index[foot],
                            price=float(df["close"].iloc[foot]), kind=want)]
    else:
        all_swings = swings if swings is not None else detect_swings(df, swing_strength)
    before = [
        sw for sw in all_swings
        if sw.kind == want and sw.index < base_start
    ]
    if not before:
        return None
    ref = before[-1]
    level = ref.price
    if beyond_wick and foot is not None and 0 <= foot < len(df):
        # Wick extreme of the swing: from just before the foot up to the
        # start of the vertex (the leg in, wicks included).
        lo = max(0, foot - 2)
        if refined.direction == "BUY":
            level = max(level, float(df["high"].iloc[lo:base_start].max()))
        else:
            level = min(level, float(df["low"].iloc[lo:base_start].min()))
    closes = df["close"].to_numpy()
    for i in range(base_end + 1, len(df)):
        c = float(closes[i])
        if refined.direction == "BUY" and c > level:
            return ref, i
        if refined.direction == "SELL" and c < level:
            return ref, i
    return None


def compute_sl_price(
    validated: ValidatedZone, config: StrongPointConfig | None = None,
) -> float:
    """SL = zone bound ± buffer.

    * BUY:  ``zone.bottom - sl_buffer_points``
    * SELL: ``zone.top    + sl_buffer_points``

    The previous swing-anchor variant is gone — under the loosened
    rules the zone bound is the structural reference.
    """
    cfg = config or StrongPointConfig()
    if validated.direction == "BUY":
        return float(validated.bottom - cfg.sl_buffer_points)
    return float(validated.top + cfg.sl_buffer_points)


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _zone_body_broken_since_formation(
    refined: RefinedZone, df: pd.DataFrame,
) -> bool:
    """True iff any bar after the pattern body-closed past the wrong side.

    For BUY: bar's close < zone.bottom (any bar).
    For SELL: bar's close > zone.top.

    Body-close only — a wick poking through doesn't count, matching
    the lifecycle module's :func:`zone_lifecycle.check_violation`
    convention. The scan begins one bar after
    ``pattern.impulse_after.end_index`` (the zone's formation bar);
    the formation bar itself can't be a "broken since formation"
    candidate by definition.
    """
    pattern_end_idx = refined.source_pattern.impulse_after.end_index
    n = len(df)
    if pattern_end_idx + 1 >= n:
        return False
    closes = df["close"].to_numpy()
    for i in range(pattern_end_idx + 1, n):
        bar_close = float(closes[i])
        if refined.direction == "BUY" and bar_close < refined.bottom:
            return True
        if refined.direction == "SELL" and bar_close > refined.top:
            return True
    return False


def _zone_touched_after(
    refined: RefinedZone, df: pd.DataFrame, bos_idx: int,
) -> bool:
    """True iff any bar after ``bos_idx`` traded into the zone edge."""
    if bos_idx + 1 >= len(df):
        return False
    if refined.direction == "BUY":
        lows = df["low"].to_numpy()[bos_idx + 1:]
        return bool((lows <= refined.top).any())
    highs = df["high"].to_numpy()[bos_idx + 1:]
    return bool((highs >= refined.bottom).any())


def _pattern_code(pattern: Pattern) -> str:
    pt = getattr(pattern, "pattern_type", None)
    return str(getattr(pt, "value", pt))


def _build_unvalidated(
    refined: RefinedZone,
    failures: list[ValidationFailure],
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
        sl_anchor_swing=None,
    )
