"""Strategy detection pipeline — loosened entry rules (May 2026).

Stitches the per-stage modules into a single ``run_strategy_pipeline``
the bot loop calls.

The previous break-and-close Strong Point gate is gone; zones are
tradeable on the first retest once they pass the size filter and
haven't been body-broken since formation. TP1 is computed by the
orchestrator (``main._try_place_setup``) via
:mod:`bot.strategy.tp1_target` so it can skip zones without a
qualifying peak before committing to order placement.

::

    OHLC df ─► detect_patterns (impulse → base → RBR/DBD/DBR/RBD)
            ─► mark_zone (wick envelope of base)            per pattern
            ─► refine_zone (size filter)                    per pattern
            ─► validate_strong_point (passthrough + body-break safety)
            ─► [ValidatedZone, …]  (only is_strong_point=True returned)

Design decisions
----------------

1. **No structure analysis here.** The old break-target swing lookup
   is removed. Structure is still computed elsewhere (the lifecycle
   FLIP detector in ``main._run_zone_lifecycle``), but not as part of
   the per-pattern strategy pipeline. That drops one
   :func:`analyze_structure` call per M5 close.

2. **Per-pattern try/except.** A single malformed pattern shouldn't
   nuke the batch.

3. **Imbalance not called.** Same as before — deferred to setup #4.

4. **Only tradeable zones returned.** Patterns whose validator
   returned ``is_strong_point=False`` (size-filter reject or
   body-broken pre-retest) are filtered out at this layer.
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
from bot.strategy.zone_marking import mark_zone
from bot.strategy.zone_refinement import (
    RefinementConfig,
    refine_zone,
)


@dataclass(frozen=True)
class StrategyPipelineConfig:
    """Aggregate of all per-stage tunables. Defaults mirror per-module defaults."""

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

    # Strong Point (loosened — only the SL buffer matters now)
    sl_buffer_points: float = 17.5

    # TP1 target — read by ``main`` when computing TP1 for each zone.
    # Kept on the pipeline config so the operator only tunes one config
    # object even though :func:`run_strategy_pipeline` doesn't consume
    # this field directly.
    tp1_local_peak_lookback_bars: int = 50


def run_strategy_pipeline(
    df: pd.DataFrame,
    config: StrategyPipelineConfig | None = None,
) -> list[ValidatedZone]:
    """Run all strategy stages; return tradeable zones."""
    cfg = config or StrategyPipelineConfig()

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
            vz = validate_strong_point(refined, df, sp_cfg)
            if vz.is_strong_point:
                validated.append(vz)
        except Exception:
            logger.exception(
                "strategy pipeline: per-pattern error; skipping and continuing"
            )
            continue

    logger.debug(
        "strategy pipeline: {} patterns → {} tradeable zones",
        len(patterns), len(validated),
    )
    return validated
