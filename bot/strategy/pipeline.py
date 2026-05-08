"""Strategy detection pipeline (Phase B composition).

Stitches the per-stage Phase B modules into a single
:func:`run_strategy_pipeline` entry point used by ``bot.main``. Each
stage is already tested in isolation; this module composes them in
the spec's order:

::

    OHLC df ─► analyze_structure (swings + BoS events)
            ─► detect_w_patterns + detect_m_patterns
            ─► mark_zone (initial box)         per pattern
            ─► refine_zone (body + size filter) per pattern
            ─► validate_strong_point             per pattern
            ─► track_imbalance                   per pattern
            ─► [ImbalanceZone, …]  (only Strong Points / Imbalances,
                                    not yet tapped)

Design decisions called out in the PR
-------------------------------------

1. **Stateless function, M5-close gating in the caller.** The pipeline
   has no internal cache: every call runs every stage. Caching is the
   main loop's job (re-run only when the latest M5 candle changes).
   Stateless = trivially testable, no surprise interactions.

2. **Per-pattern try/except.** A single bad pattern shouldn't stop the
   pipeline producing any zones. We log + skip and continue with the
   remaining patterns. The per-stage modules already raise on actually
   broken inputs; this is just defence against unexpected combinations
   (e.g. a refined pattern whose pivots are out of df bounds).

3. **Tapped zones are filtered out at exit.** ``imbalance.is_tapped``
   means price has already entered the zone since formation — the
   first-touch entry has been used. Re-trading a tapped zone violates
   the strategy. Caller still sees only fresh setups.

4. **Returns ``ImbalanceZone``s — even Strong-Point-only zones.** The
   ImbalanceZone wrapper around a non-imbalance Strong Point still has
   ``is_strong_point=True`` and ``is_imbalance=False``; ``order_manager``
   already routes ``IMBALANCE_FIRST_TOUCH`` vs ``STRONG_POINT_FIRST_TOUCH``
   off these flags, so a single uniform return type works.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from bot.strategy.imbalance import (
    ImbalanceConfig,
    ImbalanceZone,
    track_imbalance,
)
from bot.strategy.pattern_detection import (
    MPattern,
    PatternConfig,
    WPattern,
    detect_m_patterns,
    detect_w_patterns,
)
from bot.strategy.strong_point import (
    StrongPointConfig,
    validate_strong_point,
)
from bot.strategy.structure import (
    StructureConfig,
    analyze_structure,
)
from bot.strategy.zone_marking import mark_zone
from bot.strategy.zone_refinement import (
    RefinementConfig,
    refine_zone,
)


# --------------------------------------------------------------------------- #
# Config + result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyPipelineConfig:
    """Aggregate of all per-stage tunables.

    Mirrors the keys orchestrators pull from ``bot_config``. Defaults
    mirror the per-module defaults so a no-arg call works for tests
    and dry-runs.
    """

    # Structure / patterns
    swing_strength: int = 3
    pattern_tolerance_pct: float = 0.001
    pattern_lookback_bars: int = 50
    peak_threshold_pct: float = 0.002

    # Zone refinement
    zone_min_size_points: float = 5.0
    zone_max_size_points: float = 80.0

    # Strong Point validation
    impulse_min_body_ratio: float = 0.6
    base_max_range_ratio: float = 0.5

    # Imbalance tracking
    imbalance_approach_distance: float = 7.5
    imbalance_retreat_distance: float = 5.0
    imbalance_approach_threshold: int = 2


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def run_strategy_pipeline(
    df: pd.DataFrame,
    config: StrategyPipelineConfig | None = None,
) -> list[ImbalanceZone]:
    """Run all Phase B stages, return tradeable, untapped zones."""
    cfg = config or StrategyPipelineConfig()

    structure = analyze_structure(
        df,
        StructureConfig(swing_strength=cfg.swing_strength),
    )

    pattern_cfg = PatternConfig(
        swing_strength=cfg.swing_strength,
        pattern_tolerance_pct=cfg.pattern_tolerance_pct,
        peak_threshold_pct=cfg.peak_threshold_pct,
        lookback_bars=cfg.pattern_lookback_bars,
    )
    patterns: list[WPattern | MPattern] = []
    patterns.extend(detect_w_patterns(df, pattern_cfg))
    patterns.extend(detect_m_patterns(df, pattern_cfg))

    refinement_cfg = RefinementConfig(
        zone_min_size_points=cfg.zone_min_size_points,
        zone_max_size_points=cfg.zone_max_size_points,
    )
    strong_cfg = StrongPointConfig(
        impulse_min_body_ratio=cfg.impulse_min_body_ratio,
        base_max_range_ratio=cfg.base_max_range_ratio,
    )
    imbal_cfg = ImbalanceConfig(
        imbalance_approach_distance=cfg.imbalance_approach_distance,
        imbalance_retreat_distance=cfg.imbalance_retreat_distance,
        imbalance_approach_threshold=cfg.imbalance_approach_threshold,
    )

    zones: list[ImbalanceZone] = []
    for pattern in patterns:
        try:
            initial = mark_zone(pattern, df)
            refined = refine_zone(initial, df, refinement_cfg)
            if not refined.is_tradeable:
                continue
            validated = validate_strong_point(
                refined, df, structure.bos_events, strong_cfg,
            )
            if not validated.is_strong_point:
                continue
            imbalance = track_imbalance(validated, df, imbal_cfg)
            if imbalance.is_tapped:
                # First-touch already consumed; not a fresh setup.
                continue
            zones.append(imbalance)
        except Exception:
            logger.exception(
                "strategy pipeline: per-pattern error; skipping and continuing"
            )
            continue

    logger.debug(
        f"strategy pipeline: {len(patterns)} patterns → {len(zones)} "
        f"tradeable zones"
    )
    return zones
