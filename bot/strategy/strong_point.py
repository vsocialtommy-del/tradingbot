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
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Direction
from bot.strategy.zone_refinement import RefinedZone


ValidationFailure = Literal[
    "NOT_TRADEABLE",
    "ZONE_VIOLATED_BEFORE_RETEST",
]


@dataclass(frozen=True)
class StrongPointConfig:
    """Tunables for the (now minimal) Strong Point gate."""

    sl_buffer_points: float = 17.5
    """Buffer in price units below the zone bottom (BUY) or above the
    zone top (SELL). Applied directly to the zone bound — no swing
    anchor. Spec wording: '$17.50 below the zone for BUY' (mirror SELL).
    """


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
    del config  # unused — kept in signature for symmetry / future use
    failures: list[ValidationFailure] = []

    if not refined.is_tradeable:
        failures.append("NOT_TRADEABLE")
        return _build_unvalidated(refined, failures)

    if _zone_body_broken_since_formation(refined, df):
        failures.append("ZONE_VIOLATED_BEFORE_RETEST")
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
        broken_swing=None,
        broken_at=None,
        sl_anchor_swing=None,
    )


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
