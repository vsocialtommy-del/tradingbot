"""Strategy detection pipeline — Supply & Demand methodology.

Stitches the per-stage modules into a single ``run_strategy_pipeline``
the bot loop calls. v1 only implements the **Strong Point** setup;
CHoCH / Imbalance / SnD Flip are layered on the same RBR/DBD/DBR/RBD
foundation in later phases.

::

    OHLC df ─► analyze_structure (swings + BoS events)
            ─► detect_patterns (impulse → base → RBR/DBD/DBR/RBD)
            ─► mark_zone (body envelope of base)        per pattern
            ─► refine_zone (size filter)                per pattern
            ─► validate_strong_point (break-and-close + SL anchor)
            ─► [ValidatedZone, …]  (only is_strong_point=True returned)

Design decisions
----------------

1. **Stateless function, M5-close gating in the caller.** No internal
   cache; every call processes the slice the engine passes in. The
   engine ensures that slice is ``pipeline_window_bars`` long
   (default 250) so per-call cost is bounded.

2. **Per-pattern try/except.** A single malformed pattern shouldn't
   nuke the batch; we log + skip.

3. **Imbalance not called.** The deprecated ``bot.strategy.imbalance``
   was W/M-based and doesn't match the user's actual Imbalance
   setup spec (setup #4 — fresh Strong Point + 2 failed approaches).
   It'll be rebuilt when we tackle setup #4; until then it sits as
   dead code and the pipeline routes around it.

4. **Only confirmed Strong Points returned.** Patterns that haven't
   yet had their break-and-close are filtered out at this layer
   (their ``ValidatedZone.is_strong_point`` is False). The caller
   doesn't need to know about pending candidates — they'll show up
   in subsequent pipeline calls once the break confirms.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from bot.strategy.pattern_detection import (
    PatternConfig,
    detect_patterns,
)
from bot.strategy.strong_point import (
    StrongPointConfig,
    ValidatedZone,
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


@dataclass(frozen=True)
class StrategyPipelineConfig:
    """Aggregate of all per-stage tunables. Defaults mirror per-module defaults."""

    # Structure (still used for SL anchor + break target swings)
    swing_strength: int = 2

    # Pattern detection (S&D)
    impulse_body_to_range_ratio_min: float = 0.6
    impulse_atr_multiple_min: float = 1.0
    atr_period: int = 14
    max_impulse_run_candles: int = 5
    min_base_candles: int = 1
    max_base_candles: int = 5
    base_range_to_impulse_ratio_max: float = 0.6
    base_max_body_to_impulse_body_ratio: float = 0.4
    pattern_lookback_bars: int = 50

    # Zone size filter
    zone_min_size_points: float = 5.0
    zone_max_size_points: float = 80.0

    # Strong Point
    sl_buffer_points: float = 17.5


def run_strategy_pipeline(
    df: pd.DataFrame,
    config: StrategyPipelineConfig | None = None,
) -> list[ValidatedZone]:
    """Run all strategy stages; return confirmed Strong Point zones."""
    cfg = config or StrategyPipelineConfig()

    structure = analyze_structure(
        df,
        StructureConfig(swing_strength=cfg.swing_strength),
    )
    swings = list(structure.swings)

    pattern_cfg = PatternConfig(
        impulse_body_to_range_ratio_min=cfg.impulse_body_to_range_ratio_min,
        impulse_atr_multiple_min=cfg.impulse_atr_multiple_min,
        atr_period=cfg.atr_period,
        max_impulse_run_candles=cfg.max_impulse_run_candles,
        min_base_candles=cfg.min_base_candles,
        max_base_candles=cfg.max_base_candles,
        base_range_to_impulse_ratio_max=cfg.base_range_to_impulse_ratio_max,
        base_max_body_to_impulse_body_ratio=cfg.base_max_body_to_impulse_body_ratio,
        lookback_bars=cfg.pattern_lookback_bars,
    )
    patterns = detect_patterns(df, pattern_cfg)

    refinement_cfg = RefinementConfig(
        zone_min_size_points=cfg.zone_min_size_points,
        zone_max_size_points=cfg.zone_max_size_points,
    )
    sp_cfg = StrongPointConfig(sl_buffer_points=cfg.sl_buffer_points)

    validated: list[ValidatedZone] = []
    for pattern in patterns:
        try:
            zone = mark_zone(pattern, df)
            refined = refine_zone(zone, df, refinement_cfg)
            vz = validate_strong_point(refined, df, swings, sp_cfg)
            if vz.is_strong_point:
                validated.append(vz)
        except Exception:
            logger.exception(
                "strategy pipeline: per-pattern error; skipping and continuing"
            )
            continue

    logger.debug(
        "strategy pipeline: {} patterns → {} Strong Points",
        len(patterns), len(validated),
    )
    return validated
