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
* **M / W only** (``require_mw``, on for both bots from 18 Nov 2026):
  on the line chart (closes, from 2 candles before the vertex to the
  first leg-out candle) price must make two tops (sell) / two lows (buy)
  inside the zone, with a pullback between them that is also inside the
  zone and at least ``mw_min_pull`` (10 %) of the zone height deep. A
  single spike (one top, price straight in and out) is not a zone.

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
    foot_zigzag_frac: float = 0.0
    """0 = foot is the first line-chart dip before the leg in (current).
    > 0 = 'zigzag' foot: keep walking back past dips whose bounce is
    smaller than this many zone heights, so the foot is where the whole
    move into the zone started."""
    foot_max_lookback: int = 120
    exit_frac: float = 0.5
    """The leg out must close beyond the zone by at least this fraction
    of the zone height. Stops a small dip just under the vertex low from
    counting as the exit (that would cut the vertex short)."""
    require_mw: bool = False
    """Keep only clear M (sell) / W (buy) zones; drop single spikes."""
    mw_min_pull: float = 0.10
    """The pullback between the two tops / lows must be at least this
    fraction of the zone height (a tiny wiggle doesn't make an M)."""


def detect_vertex_patterns(df: pd.DataFrame, config: VertexConfig | None = None) -> list[Pattern]:
    cfg = config or VertexConfig()
    return _finish(_raw_patterns(df, cfg), df, cfg)


def _raw_patterns(df: pd.DataFrame, cfg: VertexConfig, first_out: int = 0) -> list[Pattern]:
    """Every vertex candidate whose leg out starts at or after ``first_out``
    (before de-duplication and the M / W filter). Each candidate only
    depends on the candles around it, which is what lets
    :class:`VertexPatternCache` reuse the old ones."""
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
    for out_start in range(max(cfg.min_base + 1, first_out), n):
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
    return found


def _finish(found: list[Pattern], df: pd.DataFrame, cfg: VertexConfig) -> list[Pattern]:
    out = _dedupe(found)
    if cfg.require_mw:
        c = df["close"].to_numpy(float)
        out = [p for p in out if is_mw_shape(
            c, p.base.start_index, p.base.end_index, p.base.bottom, p.base.top,
            p.direction == "SELL", cfg.mw_min_pull)]
    return out


def _shift_pattern(p: Pattern, off: int) -> Pattern:
    """The same pattern with every candle index moved back by ``off``."""
    if off == 0:
        return p
    from dataclasses import replace as _r
    return _r(
        p,
        impulse_before=_r(p.impulse_before, start_index=p.impulse_before.start_index - off,
                          end_index=p.impulse_before.end_index - off),
        base=_r(p.base, start_index=p.base.start_index - off, end_index=p.base.end_index - off),
        impulse_after=_r(p.impulse_after, start_index=p.impulse_after.start_index - off,
                         end_index=p.impulse_after.end_index - off),
        foot_index=None if p.foot_index is None else p.foot_index - off,
    )


class VertexPatternCache:
    """Same answer as :func:`detect_vertex_patterns`, without re-scanning
    the whole history on every new candle (19 Nov 2026, for the M5 bot's
    100,000-candle lookback).

    Vertex candidates are local: each one only looks at a few dozen
    candles around its leg out. So candidates whose leg out starts more
    than ``tail`` candles before the end can't change when new candles
    arrive. Those are kept from the last full scan (re-indexed as old
    candles drop off the front); only the last ``tail`` candles (plus
    ``context`` candles before them to look back into) are scanned again, along with
    the oldest 500 candles (a candidate right at the front can lose part
    of its leg in as old candles drop off).
    A full scan runs on the first call, when the candles don't line up
    with the cached ones, and once the re-scanned part grows to
    ``2 * tail``. Short histories (``min_bars`` or fewer) are always
    scanned in full, so the M1 bot's 5,000-candle window is unchanged.
    """

    def __init__(self, config: VertexConfig, *, tail: int = 3000,
                 context: int = 1000, min_bars: int = 20000) -> None:
        self.cfg = config
        self.tail = tail
        self.context = context
        self.min_bars = min_bars
        self.front = 500
        self._idx: pd.DatetimeIndex | None = None
        self._frozen: list[Pattern] = []   # raw candidates, leg out before _cut
        self._cut = 0                      # index (in the cached frame) of the first re-scanned leg out
        self.full_scans = 0

    def _full(self, df: pd.DataFrame) -> list[Pattern]:
        self.full_scans += 1
        raw = _raw_patterns(df, self.cfg)
        self._idx = df.index.copy()
        self._cut = max(0, len(df) - self.tail)
        self._frozen = [p for p in raw if p.impulse_after.start_index < self._cut]
        return _finish(raw, df, self.cfg)

    def patterns(self, df: pd.DataFrame) -> list[Pattern]:
        n = len(df)
        if n <= self.min_bars:
            return detect_vertex_patterns(df, self.cfg)
        if self._idx is None or len(df.index) == 0:
            return self._full(df)
        pos = self._idx.get_indexer([df.index[0]])[0]
        if pos < 0:
            return self._full(df)
        d = int(pos)                        # candles dropped off the front since the full scan
        cut = self._cut - d                 # first re-scanned leg out, in this frame
        overlap = len(self._idx) - d        # candles this frame shares with the cached one
        if (cut < self.front + self.context or n - cut > 2 * self.tail or overlap <= cut
                or not df.index[:overlap].equals(self._idx[d:])):
            return self._full(df)
        # The oldest candles are re-scanned too: as they drop off the front,
        # a candidate there can lose part of its leg in or foot.
        front = self.front
        head = [p for p in _raw_patterns(df.iloc[:front + 2 * self.context], self.cfg)
                if p.impulse_after.start_index < front]
        frozen = []
        for p in self._frozen:
            q = _shift_pattern(p, d)
            if q.impulse_after.start_index < front:
                continue
            frozen.append(q)
        start = cut - self.context
        recent = [_shift_pattern(p, -start)
                  for p in _raw_patterns(df.iloc[start:], self.cfg, first_out=self.context)]
        return _finish(head + frozen + recent, df, self.cfg)


def is_mw_shape(c, s: int, e: int, bottom: float, top: float, sell: bool,
                min_pull: float = 0.10) -> bool:
    """True when the line chart makes an M (sell) / W (buy) inside the zone.

    Looks at the closes from 2 candles before the vertex (``s``) to the first
    leg-out candle (``e + 1``). Needs two tops (sell) / lows (buy) whose
    closes are inside ``bottom``-``top``, with a pullback between them that
    is inside the zone too and at least ``min_pull`` of the zone height deep.
    """
    if s < 2:
        return False
    seq = np.asarray(c[s - 2:e + 2], dtype=float)
    v = seq if sell else -seq
    peaks: list[int] = []
    troughs: list[int] = []
    for i in range(1, len(v) - 1):
        a = i - 1
        while a > 0 and v[a] == v[i]:
            a -= 1
        b = i + 1
        while b < len(v) - 1 and v[b] == v[i]:
            b += 1
        if v[i] > v[a] and v[i] > v[b]:
            peaks.append(i)
        if v[i] < v[a] and v[i] < v[b]:
            troughs.append(i)

    def inside(i: int) -> bool:
        return bottom - 1e-9 <= seq[i] <= top + 1e-9

    tops = [i for i in peaks if inside(i)]
    need = min_pull * (top - bottom) - 1e-9
    for x in range(len(tops)):
        for y in range(x + 1, len(tops)):
            between = [k for k in troughs if tops[x] < k < tops[y]]
            if between and all(inside(k) for k in between):
                depth = min(v[tops[x]], v[tops[y]]) - min(v[k] for k in between)
                if depth >= need:
                    return True
    return False


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
    height = top - bottom
    if cfg.foot_zigzag_frac > 0:
        foot = _zigzag_foot(c, foot, sell, cfg.foot_zigzag_frac * height, cfg.foot_max_lookback)
    elif sell:
        while foot - 1 >= 0 and c[foot - 1] < c[foot]:
            foot -= 1
    else:
        while foot - 1 >= 0 and c[foot - 1] > c[foot]:
            foot -= 1
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


def _zigzag_foot(c, start, sell, thr, max_back):
    """Walk back from ``start``: track the lowest close (sell) / highest
    (buy); stop once price before it was ``thr`` away (a real turn)."""
    f = start
    best = c[start]
    j = start - 1
    lo = max(0, start - max_back)
    while j >= lo:
        v = c[j]
        if (sell and v < best) or (not sell and v > best):
            best, f = v, j
        elif (sell and v - best >= thr) or (not sell and best - v >= thr):
            break
        j -= 1
    return f


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
