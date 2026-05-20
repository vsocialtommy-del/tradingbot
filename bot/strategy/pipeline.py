"""Strategy detection pipeline — loosened entry rules (May 2026).

Stitches the per-stage modules into a single ``run_strategy_pipeline``
the bot loop calls.

The previous break-and-close Strong Point gate is gone; zones are
tradeable on the first retest once they pass the size filter and
haven't been body-broken since formation. TP1 is computed by the
orchestrator (``main._try_place_setup``) via
:mod:`bot.strategy.tp_target` so it can skip zones without a
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

    # Pattern detection (S&D). Strict thresholds — reverted from
    # PR #44 (impulse_atr_multiple_min, base_range_to_impulse_ratio_max)
    # and PR #46 (impulse_body_to_range_ratio_min) by PR #64.
    #
    # Why the revert: PR #44/#46 loosened detection to catch zones
    # the strict thresholds rejected. In production this turned out
    # to mark "DBDs" inside continuous impulse legs — no real base,
    # just a 1-3 bar hesitation in a trend. Operator's chart review
    # confirmed: no visible base where the bot was marking zones.
    # These junk patterns were briefly masked by the pipeline's natural
    # re-detection filter (pre-PR-#55), then exposed by PR #55's DB
    # loader; PR #63 reverted PR #55, but the underlying detection
    # is still producing junk that the lifecycle scanner then
    # flips into tradeable BUY/SELL signals.
    #
    # Strict-mode defaults restored here:
    #   * impulse_body_to_range_ratio_min = 0.6
    #     Impulse candle bodies must cover ≥60% of the bar range.
    #     Filters out wick-dominated bars (pin bars / dojis) that
    #     the loose detection happily counted as impulse.
    #   * impulse_atr_multiple_min = 1.0
    #     Impulse range must be ≥1× ATR(14). Filters out small-range
    #     bars in low-volatility periods.
    #   * base_range_to_impulse_ratio_max = 0.6
    #     Base range must be ≤60% of the impulse range. Enforces the
    #     "tight consolidation" S&D premise; loose 1.0 allowed bases
    #     as wide as the impulse, which isn't a base — it's another
    #     impulse-class candle.
    #
    # Trade-off: 3-5× fewer signals than the loose mode. The
    # ``TestStrictModeBaseline`` regression class in
    # ``bot.strategy.pattern_detection`` captures the strict
    # behaviour these defaults restore.
    impulse_body_to_range_ratio_min: float = 0.6  # PR #64: 0.0 → 0.6
    impulse_atr_multiple_min: float = 1.0  # PR #64: 0.7 → 1.0
    atr_period: int = 14
    max_impulse_run_candles: int = 5
    min_base_candles: int = 1
    max_base_candles: int = 5
    base_range_to_impulse_ratio_max: float = 0.6  # PR #64: 1.0 → 0.6
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

    # PR #56 (partial): zone freshness window for the dedup pre-flight.
    #
    # ``Bot._zone_already_used`` only blocks new setups on overlapping
    # CONSUMED / VIOLATED / FLIPPED zones whose ``created_at`` is
    # within the last ``zone_freshness_hours``. Older "burnt" zones
    # don't block fresh patterns.
    #
    # Solves the production graveyard problem (87 dead zones
    # accumulated over 8 days blocking every fresh pattern in a
    # 250-point band) without permanently turning off dedup. The
    # 6-hour default treats "more than one session ago" as ancient
    # history.
    #
    # PR #56's loader half (``_load_confirmed_candidates`` freshness)
    # was removed when PR #55 was reverted — the bot is back to
    # acting on pipeline-fresh detections only.
    #
    # Tunable: smaller = more aggressive re-engagement; larger = more
    # respect for recent dead zones. Set to 0 to disable (restores
    # the pre-PR-#56 "any age" behaviour).
    zone_freshness_hours: float = 6.0

    # PR #60: zone is strict wick-to-wick of the base candles only.
    # No extension into bordering impulse bars by default.
    #
    # History: PR #57 introduced a symmetric N-bar extension to catch
    # rejection wicks on the border bars. That over-extended zones —
    # for a BUY (RBR) the impulse_before bar's high is the rally's
    # peak, pulling ``top`` way up; for a SELL (RBD) the impulse_after
    # bar's low is the drop's trough, pulling ``bottom`` way down.
    # Real-world example: a BUY zone marked 4476.60-4514.54 ($37.94
    # wide) when the actual rejection was ~$10-15 — the rally peak
    # got swept in as if it were the zone.
    #
    # ``mark_zone`` retains the ``wick_extend_bars`` kwarg as an
    # opt-in. When > 0 it is **direction-aware**: BUY widens only
    # ``bottom`` (lower rejection wicks), SELL widens only ``top``
    # (upper rejection wicks). The opposite side stays at base. The
    # default is ``0`` — strict base-only.
    zone_wick_extend_bars: int = 0


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
            zone = mark_zone(
                pattern, df,
                wick_extend_bars=cfg.zone_wick_extend_bars,
            )
            refined = refine_zone(zone, df, refinement_cfg)
            vz = validate_strong_point(refined, df, sp_cfg)
            # Return every validated zone (tradeable or not). The caller
            # decides what to do with each — main.py persists the
            # is_tradeable=True subset and only attempts placement on
            # is_strong_point=True. Non-tradeable zones are still useful
            # for downstream callers (debug / backtest diagnostics).
            validated.append(vz)
        except Exception:
            logger.exception(
                "strategy pipeline: per-pattern error; skipping and continuing"
            )
            continue

    logger.debug(
        "strategy pipeline: {} patterns → {} validated zones "
        "({} tradeable, {} strong point)",
        len(patterns), len(validated),
        sum(1 for v in validated if v.refined_zone.is_tradeable),
        sum(1 for v in validated if v.is_strong_point),
    )
    return validated
