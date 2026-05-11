"""DEPRECATED — Imbalance Zone qualification (placeholder for Setup 4).

.. warning::
   This module's previous implementation was based on the legacy W/M
   pattern methodology and did **NOT** match the user's actual
   Imbalance Zone spec. As of PR #31 (S&D methodology pivot) the
   module is reduced to a scaffolding stub.

   The real spec (Setup 4 in
   ``docs/strategy_reference/README.md``): a **fresh** Strong Point
   zone that has been approached **twice without being tapped** →
   Imbalance Zone → high-probability push to TP2 on first real
   touch. v1 doesn't implement this; v3/v4 will rebuild this module
   from the spec.

   The dataclasses below are preserved as the rough API shape future
   work will start from. They will likely change shape during the
   rebuild. Do not import from this module in new code; use the
   Strong Point pipeline (``bot.strategy.pipeline``) instead.

The pipeline does not call ``track_imbalance``. Calling it raises
``NotImplementedError`` with a pointer at the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from bot.strategy.strong_point import ValidatedZone


@dataclass(frozen=True)
class ApproachEvent:
    """One completed approach to the zone (enters band, then retreats)."""

    start_bar_index: int
    completed_bar_index: int
    start_time: pd.Timestamp
    completed_time: pd.Timestamp
    closest_price: float
    distance_from_zone: float


@dataclass(frozen=True)
class ImbalanceConfig:
    """Tunables for Imbalance Zone qualification (Setup 4)."""

    imbalance_approach_distance: float = 7.5
    imbalance_retreat_distance: float = 5.0
    imbalance_approach_threshold: int = 2


@dataclass(frozen=True)
class ImbalanceZone:
    """Placeholder result type — will be redesigned in Setup 4 work."""

    validated_zone: ValidatedZone
    approach_count: int = 0
    is_imbalance: bool = False
    approach_events: list[ApproachEvent] = field(default_factory=list)
    qualified_at: pd.Timestamp | None = None
    is_tapped: bool = False
    tapped_at: pd.Timestamp | None = None


def track_imbalance(
    zone: ValidatedZone,
    df: pd.DataFrame,
    config: ImbalanceConfig | None = None,
) -> ImbalanceZone:
    """Setup 4 entry point — not implemented in v1.

    Raises ``NotImplementedError``. See the module docstring for the
    spec to implement when we get to Setup 4.
    """
    raise NotImplementedError(
        "Imbalance Zone (Setup 4) is not implemented in v1. "
        "See docs/strategy_reference/README.md and rebuild this "
        "function from the canonical spec when tackling Setup 4."
    )
