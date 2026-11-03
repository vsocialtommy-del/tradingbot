"""Direction gate: trade with the M15 trend, or against it only from a fresh
H1 zone.

XAUTISM rule (Nov 2026)
-----------------------
* M15 market structure decides the direction. Closes are resampled from
  the bot's M5 history, swings detected on closes, and the last
  ``trend_min_swings`` labels give UPTREND (all HH/HL), DOWNTREND (all
  LL/LH) or RANGE (mixed).
* BUY is blocked in a DOWNTREND, SELL is blocked in an UPTREND. RANGE
  allows both.
* Exception: a counter-trend trade is allowed when the M5 zone sits inside
  or touches a **fresh H1 zone in the same direction** (H1 demand for a
  buy, H1 supply for a sell). H1 zones are found with the same rules as
  M5 zones (DBR/RBD, BOS out of the base, untouched since the BOS) on H1
  bars resampled from the same history. Counter-trend trades use the same
  size and rules as any other trade.

Pure logic on DataFrames; the orchestrator calls :func:`check_direction`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import pandas as pd
from loguru import logger

from bot.strategy.pipeline import StrategyPipelineConfig, run_strategy_pipeline
from bot.strategy.structure import StructureConfig, StructureState, analyze_structure


Direction = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class DirectionGateConfig:
    enabled: bool = True
    trend_timeframe: str = "15min"
    """Pandas resample rule for the direction timeframe (M15)."""
    trend_swing_strength: int = 3
    trend_min_swings: int = 4
    allow_counter_trend_from_htf_zone: bool = True
    htf_timeframe: str = "1h"
    """Timeframe whose fresh zones allow a counter-trend trade (H1)."""
    htf_zone_min_size_points: float = 3.0


@dataclass(frozen=True)
class DirectionDecision:
    allowed: bool
    trend: StructureState
    reason: str
    htf_zone: tuple[float, float] | None = None
    """(bottom, top) of the fresh H1 zone that allowed a counter-trend trade."""


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M5 OHLC to a higher timeframe (complete bars only)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(
        subset=["open", "high", "low", "close"],
    )
    # Drop the last bar if it is still forming (its window extends past
    # the final M5 bar we have).
    if len(out) and len(df):
        step = pd.Timedelta(rule)
        last_m5_end = df.index[-1] + pd.Timedelta(minutes=5)
        if out.index[-1] + step > last_m5_end:
            out = out.iloc[:-1]
    return out


def trend_state(df_m5: pd.DataFrame, cfg: DirectionGateConfig) -> StructureState:
    htf = resample_ohlc(df_m5, cfg.trend_timeframe)
    if len(htf) < 2 * cfg.trend_swing_strength + 1:
        return StructureState.RANGE
    snap = analyze_structure(
        htf,
        StructureConfig(
            swing_strength=cfg.trend_swing_strength,
            trend_min_swings=cfg.trend_min_swings,
        ),
    )
    return snap.state


def fresh_htf_zones(
    df_m5: pd.DataFrame,
    cfg: DirectionGateConfig,
    pipeline_cfg: StrategyPipelineConfig | None = None,
) -> list[tuple[Direction, float, float]]:
    """Fresh (tradeable) zones on the HTF, as (direction, bottom, top)."""
    htf = resample_ohlc(df_m5, cfg.htf_timeframe)
    if len(htf) < 20:
        return []
    base = pipeline_cfg or StrategyPipelineConfig()
    pcfg = replace(base, zone_min_size_points=cfg.htf_zone_min_size_points)
    zones = run_strategy_pipeline(htf, pcfg)
    return [(z.direction, float(z.bottom), float(z.top)) for z in zones if z.is_strong_point]


def check_direction(
    *,
    direction: Direction,
    zone_bottom: float,
    zone_top: float,
    df_m5: pd.DataFrame,
    config: DirectionGateConfig | None = None,
    pipeline_cfg: StrategyPipelineConfig | None = None,
) -> DirectionDecision:
    cfg = config or DirectionGateConfig()
    if not cfg.enabled:
        return DirectionDecision(True, StructureState.RANGE, "direction gate disabled")

    state = trend_state(df_m5, cfg)
    against = (
        (direction == "BUY" and state == StructureState.DOWNTREND)
        or (direction == "SELL" and state == StructureState.UPTREND)
    )
    if not against:
        return DirectionDecision(True, state, f"with M15 trend ({state.value})")

    if cfg.allow_counter_trend_from_htf_zone:
        for zdir, zb, zt in fresh_htf_zones(df_m5, cfg, pipeline_cfg):
            # "Inside or touches": the price ranges overlap or meet.
            if zdir == direction and zone_bottom <= zt and zone_top >= zb:
                return DirectionDecision(
                    True, state,
                    f"counter-trend allowed: inside fresh H1 {zdir} zone "
                    f"{zb:.2f}-{zt:.2f}",
                    htf_zone=(zb, zt),
                )

    logger.debug("direction gate: {} blocked, M15 {}", direction, state.value)
    return DirectionDecision(
        False, state,
        f"against M15 trend ({state.value}) and not inside a fresh H1 zone",
    )
