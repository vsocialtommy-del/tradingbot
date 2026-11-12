"""Direction gate: trade with the M15 trend, or against it only from a fresh
H1 zone.

XAUTISM rule (Nov 2026)
-----------------------
* Market structure on the trend timeframe (M15 for the M5 bot, M5 for the
  M1 bot) decides the direction. Bars are resampled from the bot's M5
  history and swings found on closes (``trend_swing_strength`` bars each
  side).
* ``trend_rule="bos"`` (default since 12 Nov 2026, Tommy's rule): a close
  below the last confirmed swing low = DOWNTREND straight away, and it
  stays a downtrend until a close above the last confirmed swing high
  (= UPTREND). RANGE only until the first break in the history.
* ``trend_rule="swings"`` (the old rule): the last ``trend_min_swings``
  labels give UPTREND (all HH/HL), DOWNTREND (all LL/LH) or RANGE (mixed).
  It lagged badly in one-way moves (the 11-12 Nov drops read as RANGE).
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
from bot.strategy.structure import (
    StructureConfig,
    StructureState,
    analyze_structure,
    detect_swings,
)

Direction = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class DirectionGateConfig:
    enabled: bool = True
    trend_timeframe: str = "15min"
    """Pandas resample rule for the direction timeframe. The live bots set
    their own (12 Nov 2026): M5 bot reads M5, M1 bot reads M1."""
    trend_swing_strength: int = 3
    trend_min_swings: int = 4
    trend_rule: str = "bos"
    """"bos": a close past the last swing = trend (Tommy's rule).
    "swings": last ``trend_min_swings`` labels must agree (old rule)."""
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


def source_step(df: pd.DataFrame) -> pd.Timedelta:
    """Length of one source bar (M1 or M5), from the smallest gap between
    the last few bars (weekend gaps are bigger, so the smallest is right)."""
    if len(df) < 2:
        return pd.Timedelta(minutes=5)
    gaps = df.index[-12:].to_series().diff().dropna()
    gaps = gaps[gaps > pd.Timedelta(0)]
    return gaps.min() if len(gaps) else pd.Timedelta(minutes=5)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample closed M1/M5 bars to the trend timeframe (complete bars
    only). Same timeframe in and out (M5 trend on M5 bars) = the bars as
    they are."""
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
        last_end = df.index[-1] + source_step(df)
        if out.index[-1] + step > last_end:
            out = out.iloc[:-1]
    return out


def tf_label(rule: str) -> str:
    """"15min" -> "M15", "5min" -> "M5", "1h" -> "H1" (for log lines)."""
    td = pd.Timedelta(rule)
    mins = int(td.total_seconds() // 60)
    return f"H{mins // 60}" if mins >= 60 and mins % 60 == 0 else f"M{mins}"


@dataclass(frozen=True)
class BosTrend:
    state: StructureState
    since: pd.Timestamp | None
    """Open time of the bar whose close set the current state."""
    last_high: float | None
    """Last confirmed swing high: a close above it = uptrend."""
    last_low: float | None
    """Last confirmed swing low: a close below it = downtrend."""


def bos_trend_detail(htf: pd.DataFrame, strength: int) -> BosTrend:
    """Tommy's rule: walk the bars in order; a close below the last
    confirmed swing low = DOWNTREND, a close above the last confirmed swing
    high = UPTREND. A swing at bar i is only known once bar i+strength has
    closed, so nothing is used before it could be seen."""
    closes = htf["close"].to_numpy()
    swings = {sw.index: sw for sw in detect_swings(htf, strength)}
    last_high = last_low = None
    state = StructureState.RANGE
    since = None
    for t in range(len(closes)):
        sw = swings.get(t - strength)
        if sw is not None:
            if sw.kind == "HIGH":
                last_high = sw.price
            else:
                last_low = sw.price
        c = closes[t]
        new = state
        if last_low is not None and c < last_low:
            new = StructureState.DOWNTREND
        elif last_high is not None and c > last_high:
            new = StructureState.UPTREND
        if new != state:
            state, since = new, htf.index[t]
    return BosTrend(state, since, last_high, last_low)


def bos_trend(htf: pd.DataFrame, strength: int) -> StructureState:
    return bos_trend_detail(htf, strength).state


def trend_summary(df_m5: pd.DataFrame, cfg: DirectionGateConfig) -> dict:
    """What the direction gate thinks right now, for the dashboard."""
    out: dict = {"timeframe": tf_label(cfg.trend_timeframe), "rule": cfg.trend_rule,
                 "state": StructureState.RANGE.value, "since": None,
                 "up_above": None, "down_below": None}
    htf = resample_ohlc(df_m5, cfg.trend_timeframe)
    if len(htf) < 2 * cfg.trend_swing_strength + 1:
        return out
    if cfg.trend_rule == "bos":
        d = bos_trend_detail(htf, cfg.trend_swing_strength)
        out.update(
            state=d.state.value,
            since=None if d.since is None
            else (d.since + pd.Timedelta(cfg.trend_timeframe)).isoformat(),  # the close that did it
            # The level that would flip it the other way.
            up_above=None if d.state == StructureState.UPTREND or d.last_high is None
            else round(float(d.last_high), 2),
            down_below=None if d.state == StructureState.DOWNTREND or d.last_low is None
            else round(float(d.last_low), 2),
        )
    else:
        out["state"] = trend_state(df_m5, cfg).value
    return out


def trend_state(df_m5: pd.DataFrame, cfg: DirectionGateConfig) -> StructureState:
    htf = resample_ohlc(df_m5, cfg.trend_timeframe)
    if len(htf) < 2 * cfg.trend_swing_strength + 1:
        return StructureState.RANGE
    if cfg.trend_rule == "bos":
        return bos_trend(htf, cfg.trend_swing_strength)
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
        return DirectionDecision(
            True, state, f"with {tf_label(cfg.trend_timeframe)} trend ({state.value})",
        )

    if cfg.allow_counter_trend_from_htf_zone:
        for zdir, zb, zt in fresh_htf_zones(df_m5, cfg, pipeline_cfg):
            # "Inside or touches": the price ranges overlap or meet.
            if zdir == direction and zone_bottom <= zt and zone_top >= zb:
                return DirectionDecision(
                    True, state,
                    f"counter-trend allowed: inside fresh {tf_label(cfg.htf_timeframe)} {zdir} zone "
                    f"{zb:.2f}-{zt:.2f}",
                    htf_zone=(zb, zt),
                )

    logger.debug("direction gate: {} blocked, {} {}", direction, tf_label(cfg.trend_timeframe), state.value)
    return DirectionDecision(
        False, state,
        f"against {tf_label(cfg.trend_timeframe)} trend ({state.value}) and not inside a fresh "
        f"{tf_label(cfg.htf_timeframe)} zone",
    )
