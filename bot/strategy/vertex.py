"""Vertex (M / W) zone detection (Nov 2026).

Tommy's zones are the *vertex* of an M (supply) or W (demand) on the
line chart: the cluster of candles where price turned, between a clear
leg in and a clear leg out. The zone is wick to wick of those candles.

Unlike the classic detector, the vertex candles don't have to be small
next to the legs, and there is no size rule on the legs: a leg is a few
same-colour candles that clearly take price into / out of the vertex.

Supply (RBD), mirrored for demand (DBR):

* **Vertex**: ``min_base``–``max_base`` consecutive candles. Zone top =
  highest wick, bottom = lowest wick.
* **Leg in**: the run of consecutive bullish candles ending right
  before the vertex (up to ``max_leg`` candles). It must start below
  the zone (lowest open of the run < zone bottom) and its last close
  must be inside / above the zone bottom. Price came up from below.
* **Leg out**: the run of consecutive bearish candles starting right
  after the vertex (up to ``max_leg``). One of them must close below the
  zone bottom by at least ``exit_frac`` (50 %) of the zone height:
  price clearly left the vertex.
* **Foot**: the line-chart low (sell) / high (buy) the leg in started
  from. Leg in and leg out must each move at least one zone height
  (``leg_min_frac``). The foot is the BOS level and the TP.
* When several vertex lengths fit, the shortest one (latest start) is
  used: the candles after the last leg-in candle.

The resulting :class:`Pattern` goes through the normal pipeline (size
filter, body-break, BOS, fresh-since-BOS) exactly like classic patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.strategy.pattern_detection import Base, Impulse, Pattern, PatternType


@dataclass(frozen=True)
class VertexConfig:
    min_base: int = 3
    max_base: int = 8
    max_leg: int = 5
    leg_min_frac: float = 1.0
    """Leg in and leg out must each move at least this many zone heights
    on the line chart (cuts flat, dead-market 'zones')."""
    exit_frac: float = 0.5
    """The leg out must close beyond the zone by at least this fraction
    of the zone height. Stops a small dip just under the vertex low from
    counting as the exit (that would cut the vertex short)."""


def detect_vertex_patterns(df: pd.DataFrame, config: VertexConfig | None = None) -> list[Pattern]:
    cfg = config or VertexConfig()
    n = len(df)
    if n < cfg.min_base + 2:
        return []
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    bull = c > o
    bear = c < o

    found: list[Pattern] = []
    for out_start in range(cfg.min_base + 1, n):
        for direction in ("SELL", "BUY"):
            leg_out_dir = bear if direction == "SELL" else bull
            leg_in_dir = bull if direction == "SELL" else bear
            if not leg_out_dir[out_start]:
                continue
            e = out_start - 1
            for size in range(cfg.min_base, cfg.max_base + 1):
                s = e - size + 1
                if s < 1:
                    break
                top = float(h[s:e + 1].max())
                bottom = float(l[s:e + 1].min())
                p = _try(direction, s, e, out_start, top, bottom, o, h, l, c,
                         leg_in_dir, leg_out_dir, cfg, n)
                if p is None:
                    continue
                (in_s, in_e), (out_s, out_e), foot = p
                found.append(_pattern(direction, idx, o, h, l, c, in_s, in_e, s, e,
                                      out_s, out_e, top, bottom, foot))
                break
    return _dedupe(found)


def _try(direction, s, e, out_start, top, bottom, o, h, l, c,
         leg_in_dir, leg_out_dir, cfg, n):
    sell = direction == "SELL"
    # Leg in: consecutive same-colour candles ending at s-1.
    in_e = s - 1
    if not leg_in_dir[in_e]:
        return None
    in_s = in_e
    while in_s - 1 >= 0 and leg_in_dir[in_s - 1] and in_e - in_s + 1 < cfg.max_leg:
        in_s -= 1
    if sell:
        if o[in_s:in_e + 1].min() >= bottom or c[in_e] <= bottom:
            return None
    else:
        if o[in_s:in_e + 1].max() <= top or c[in_e] >= top:
            return None
    # Foot: the line-chart low (sell) / high (buy) the leg in came from.
    foot = in_s - 1
    if foot < 0:
        return None
    if sell:
        while foot - 1 >= 0 and c[foot - 1] < c[foot]:
            foot -= 1
    else:
        while foot - 1 >= 0 and c[foot - 1] > c[foot]:
            foot -= 1
    height = top - bottom
    if abs(c[in_e] - c[foot]) < cfg.leg_min_frac * height:
        return None
    # Leg out: consecutive same-colour candles from out_start; one closes beyond.
    k = out_start
    margin = cfg.exit_frac * height
    while k < n and leg_out_dir[k] and k - out_start < cfg.max_leg:
        beyond = (sell and c[k] < bottom - margin) or (not sell and c[k] > top + margin)
        if beyond and abs(c[e] - c[k]) >= cfg.leg_min_frac * height:
            return (in_s, in_e), (out_start, k), foot
        k += 1
    return None


def _impulse(direction, idx, o, h, l, c, a, b) -> Impulse:
    if direction == "RALLY":
        rng = float(h[b] - l[a])
    else:
        rng = float(h[a] - l[b])
    return Impulse(
        direction=direction, start_index=a, end_index=b,
        start_time=idx[a], end_time=idx[b], range_size=rng,
        largest_body=float(np.abs(c[a:b + 1] - o[a:b + 1]).max()),
        candle_count=b - a + 1,
    )


def _pattern(direction, idx, o, h, l, c, in_s, in_e, s, e, out_s, out_e, top, bottom, foot) -> Pattern:
    sell = direction == "SELL"
    leg_in = _impulse("RALLY" if sell else "DROP", idx, o, h, l, c, in_s, in_e)
    leg_out = _impulse("DROP" if sell else "RALLY", idx, o, h, l, c, out_s, out_e)
    base = Base(
        start_index=s, end_index=e, candle_count=e - s + 1,
        top=top, bottom=bottom, range_size=top - bottom,
        largest_body=float(np.abs(c[s:e + 1] - o[s:e + 1]).max()),
    )
    return Pattern(
        pattern_type=PatternType.RBD if sell else PatternType.DBR,
        impulse_before=leg_in, base=base, impulse_after=leg_out,
        direction=direction,  # type: ignore[arg-type]
        formed_at=leg_out.end_time,
        foot_index=foot,
    )


def _dedupe(patterns: list[Pattern]) -> list[Pattern]:
    """Same direction, overlapping vertex candles → keep the earliest."""
    patterns = sorted(patterns, key=lambda p: (p.base.end_index, p.base.start_index))
    kept: list[Pattern] = []
    last_end: dict[str, int] = {}
    for p in patterns:
        if p.base.start_index <= last_end.get(p.direction, -1):
            continue
        kept.append(p)
        last_end[p.direction] = p.base.end_index
    return kept
