"""Vertex (M / W) zone detection."""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.pipeline import StrategyPipelineConfig, run_strategy_pipeline
from bot.strategy.vertex import VertexConfig, detect_vertex_patterns


def frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2026-10-29 16:00", periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1)


# 29 Oct-style sell: rally in, M-shaped top of 5 candles, big drop out.
# Real M on the wicks (20 Nov 2026 rule): tops 4025.3 and 4024.5, dip to 4020.0 between.
FLAT = [(4014.0, 4014.6, 4013.6, 4014.2)] * 6
M_TOP = FLAT + [
    (4014.2, 4016.8, 4014.0, 4016.6),   # leg in (bull)
    (4016.6, 4018.6, 4014.8, 4018.4),   # leg in (bull)
    (4017.5, 4023.2, 4017.4, 4023.0),   # leg in (bull, big)
    (4023.0, 4025.3, 4020.0, 4020.4),   # vertex
    (4020.4, 4023.0, 4020.0, 4022.3),   # vertex
    (4022.3, 4024.5, 4019.9, 4020.2),   # vertex: second top (wick 4024.5, 0.8 under the first)
    (4020.2, 4021.0, 4018.8, 4019.4),   # vertex
    (4019.4, 4021.1, 4018.0, 4019.45),  # vertex (doji-ish)
    (4019.4, 4019.5, 4015.4, 4015.6),   # leg out (bear) closes below 4018.0
    (4015.6, 4016.2, 4011.0, 4011.5),   # leg out continues: a full zone height below
]


def test_m_top_is_one_sell_zone_wick_to_wick() -> None:
    ps = detect_vertex_patterns(frame(M_TOP))
    sells = [p for p in ps if p.direction == "SELL"]
    assert len(sells) == 1
    p = sells[0]
    assert p.pattern_type.value == "RBD"
    assert p.base.candle_count == 5
    assert p.base.top == pytest.approx(4025.3)
    assert p.base.bottom == pytest.approx(4018.0)
    assert p.impulse_before.end_index == p.base.start_index - 1
    assert p.impulse_after.start_index == p.base.end_index + 1


def test_w_bottom_mirror() -> None:
    mirrored = [(8000 - o, 8000 - l, 8000 - h, 8000 - c) for o, h, l, c in M_TOP]
    ps = [p for p in detect_vertex_patterns(frame(mirrored)) if p.direction == "BUY"]
    assert len(ps) == 1
    assert ps[0].pattern_type.value == "DBR"
    assert ps[0].base.bottom == pytest.approx(8000 - 4025.3)
    assert ps[0].base.top == pytest.approx(8000 - 4018.0)


def test_no_exit_close_no_zone() -> None:
    rows = M_TOP[:-2] + [(4019.4, 4019.5, 4017.0, 4018.5), (4018.5, 4021.0, 4018.2, 4020.8)]
    assert [p for p in detect_vertex_patterns(frame(rows)) if p.direction == "SELL"] == []


def test_min_base_respected() -> None:
    assert [p for p in detect_vertex_patterns(frame(M_TOP), VertexConfig(min_base=6)) if p.direction == "SELL"] == []


def test_pipeline_uses_vertex_detector() -> None:
    zs = run_strategy_pipeline(frame(M_TOP), StrategyPipelineConfig(pattern_detector="vertex", require_bos=False))
    sells = [z for z in zs if z.direction == "SELL"]
    assert sells and sells[0].top == pytest.approx(4025.3) and sells[0].bottom == pytest.approx(4018.0)


def test_foot_is_the_bos_level_and_tp() -> None:
    p = [p for p in detect_vertex_patterns(frame(M_TOP)) if p.direction == "SELL"][0]
    # Leg in started from the flat area: close 4014.2 is the foot.
    assert p.foot_index is not None
    assert frame(M_TOP)["close"].iloc[p.foot_index] == pytest.approx(4014.2)
    zs = run_strategy_pipeline(frame(M_TOP), StrategyPipelineConfig(pattern_detector="vertex"))
    z = [z for z in zs if z.direction == "SELL"][0]
    assert z.broken_swing is not None and z.broken_swing.price == pytest.approx(4014.2)


def test_small_leg_out_rejected() -> None:
    rows = M_TOP[:-1] + [(4015.6, 4016.0, 4014.0, 4014.2)]  # only 5.25 below the vertex close
    assert [p for p in detect_vertex_patterns(frame(rows)) if p.direction == "SELL"] == []


def test_vertex_is_the_default_detector() -> None:
    assert StrategyPipelineConfig().pattern_detector == "vertex"


def test_zigzag_foot_skips_small_pullback() -> None:
    # Rally with a small dip part way up: 4092.6 start, dip to 4101.9.
    rows = [
        (4101.0, 4101.5, 4100.6, 4101.2),
        (4101.2, 4101.3, 4092.0, 4092.6),   # real start of the move
        (4092.6, 4098.0, 4092.4, 4097.6),
        (4097.6, 4106.0, 4097.4, 4105.5),
        (4105.5, 4105.7, 4101.5, 4101.9),   # small pullback (3.6 bounce before it)
        (4101.9, 4106.2, 4101.8, 4106.0),   # leg in
        (4106.0, 4110.6, 4105.8, 4110.4),
        (4110.4, 4113.2, 4110.2, 4113.0),
        (4113.0, 4114.8, 4111.0, 4111.3),   # vertex (zone ~4107.2-4115.3)
        (4111.3, 4112.0, 4107.2, 4110.8),
        (4110.8, 4115.3, 4110.5, 4114.3),
        (4114.3, 4114.6, 4108.8, 4109.0),   # leg out
        (4109.0, 4109.2, 4101.0, 4101.4),
        (4101.4, 4101.6, 4097.0, 4097.5),
    ]
    df = frame(rows)
    near = [p for p in detect_vertex_patterns(df) if p.direction == "SELL"]
    far = [p for p in detect_vertex_patterns(df, VertexConfig(foot_zigzag_frac=0.5)) if p.direction == "SELL"]
    assert near and far
    assert df["close"].iloc[near[0].foot_index] == pytest.approx(4101.9)
    assert df["close"].iloc[far[0].foot_index] == pytest.approx(4092.6)


def test_default_foot_is_zigzag_quarter() -> None:
    assert StrategyPipelineConfig().vertex_foot_zigzag_frac == 0.25


# 18 Nov 2026: M / W only. Same rally in and drop out as M_TOP, but the
# vertex just slides down from one top: a single spike, not a zone.
SPIKE_TOP = FLAT + [
    (4014.2, 4016.8, 4014.0, 4016.6),   # leg in (bull)
    (4016.6, 4018.6, 4014.8, 4018.4),   # leg in (bull)
    (4017.5, 4023.2, 4017.4, 4023.0),   # leg in (bull, big): the only top
    (4023.0, 4025.3, 4022.0, 4022.4),   # vertex, each close lower
    (4022.4, 4022.8, 4021.0, 4021.5),
    (4021.5, 4021.9, 4020.0, 4020.6),
    (4020.6, 4021.0, 4018.8, 4019.6),
    (4019.6, 4020.0, 4018.0, 4019.2),
    (4019.2, 4019.5, 4015.4, 4015.6),   # leg out (bear)
    (4015.6, 4016.2, 4011.0, 4011.5),
]


def test_spike_found_without_the_mw_rule() -> None:
    sells = [p for p in detect_vertex_patterns(frame(SPIKE_TOP), VertexConfig(require_mw=False))
             if p.direction == "SELL"]
    assert len(sells) == 1


def test_mw_rule_keeps_the_m_and_drops_the_spike() -> None:
    cfg = VertexConfig(require_mw=True)
    assert len([p for p in detect_vertex_patterns(frame(M_TOP), cfg) if p.direction == "SELL"]) == 1
    assert [p for p in detect_vertex_patterns(frame(SPIKE_TOP), cfg) if p.direction == "SELL"] == []


def test_mw_rule_mirror_for_buys() -> None:
    cfg = VertexConfig(require_mw=True)
    flip = lambda rows: [(8000 - o, 8000 - l, 8000 - h, 8000 - c) for o, h, l, c in rows]
    assert len([p for p in detect_vertex_patterns(frame(flip(M_TOP)), cfg) if p.direction == "BUY"]) == 1
    assert [p for p in detect_vertex_patterns(frame(flip(SPIKE_TOP)), cfg) if p.direction == "BUY"] == []


class TestIsMwShape:
    # closes: 2 before the vertex, the vertex, the first leg-out candle
    def test_two_tops_with_a_real_pullback(self) -> None:
        from bot.strategy.vertex import is_mw_shape
        c = [4010, 4019, 4018.2, 4016.6, 4018.0, 4012]   # tops 4018.2 / 4018.0, pullback 4016.6
        assert is_mw_shape(c, 2, 4, 4016.0, 4020.0, sell=True)   # 1.4 pts = 35% of a 4 pt zone

    def test_tiny_wiggle_is_a_spike(self) -> None:
        from bot.strategy.vertex import is_mw_shape
        c = [4010, 4019, 4018.2, 4018.0, 4018.1, 4012]   # 0.1 pt dip = 2.5% of the zone
        assert not is_mw_shape(c, 2, 4, 4016.0, 4020.0, sell=True)

    def test_pullback_outside_the_zone_doesnt_count(self) -> None:
        from bot.strategy.vertex import is_mw_shape
        c = [4010, 4019, 4018.2, 4015.0, 4018.0, 4012]   # pullback below the zone bottom
        assert not is_mw_shape(c, 2, 4, 4016.0, 4020.0, sell=True)

    def test_w_for_buys(self) -> None:
        from bot.strategy.vertex import is_mw_shape
        c = [4030, 4021, 4021.8, 4023.4, 4022.0, 4028]
        assert is_mw_shape(c, 2, 4, 4020.0, 4024.0, sell=False)


# --------------------------------------------------------------------------- #
# VertexPatternCache (19 Nov 2026): the M5 bot's 100,000-candle lookback
# re-scans only the newest candles; it must give exactly the same zones.
# --------------------------------------------------------------------------- #

import numpy as np

from bot.strategy.vertex import VertexPatternCache


def _walk(n: int, seed: int = 7) -> pd.DataFrame:
    """A random-walk gold-ish chart, rounded to cents so ties happen."""
    rng = np.random.default_rng(seed)
    close = 4000 + np.cumsum(rng.normal(0, 1.2, n))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + rng.exponential(0.6, n)
    low = np.minimum(open_, close) - rng.exponential(0.6, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close},
                        index=idx).round(2).assign(volume=1)


def _key(ps):
    return [(p.direction, p.base.start_index, p.base.end_index,
             p.impulse_before.start_index, p.impulse_after.end_index,
             p.foot_index, p.base.top, p.base.bottom) for p in ps]


@pytest.mark.parametrize("mw", [False, True])
def test_cache_matches_full_scan_as_the_window_slides(mw: bool) -> None:
    cfg = VertexConfig(require_mw=mw)
    data = _walk(9000)
    window = 4000
    cache = VertexPatternCache(cfg, tail=400, context=150, min_bars=1000)
    cache.front = 120
    checked = 0
    for end in list(range(window, window + 30)) + list(range(window + 30, 9000, 61)):
        df = data.iloc[end - window:end]
        got = cache.patterns(df)
        assert _key(got) == _key(detect_vertex_patterns(df, cfg)), end
        checked += 1
    assert checked > 100
    assert len(got) > (5 if mw else 20)   # the chart really has zones in it
    assert cache.full_scans < checked / 5  # mostly served from the cache


def test_cache_rescans_when_the_candles_dont_line_up() -> None:
    cfg = VertexConfig()
    data = _walk(6000, seed=3)
    cache = VertexPatternCache(cfg, tail=400, context=150, min_bars=1000)
    cache.patterns(data.iloc[:4000])
    other = _walk(4000, seed=11)          # a different chart, different times
    other.index = other.index + pd.Timedelta(days=400)
    assert _key(cache.patterns(other)) == _key(detect_vertex_patterns(other, cfg))
    assert cache.full_scans == 2


def test_short_history_is_always_a_full_scan() -> None:
    cfg = VertexConfig()
    df = _walk(800)
    cache = VertexPatternCache(cfg, min_bars=1000)
    assert _key(cache.patterns(df)) == _key(detect_vertex_patterns(df, cfg))
    assert cache.full_scans == 0


# 19 Nov 2026: M5 zones may have a 2-candle vertex; the 10% M / W dip
# still decides (Tommy's 18:30-18:35 sell, 19 Nov).
TWO_CANDLE_M = [
    (3676.2, 3677.8, 3675.7, 3677.6),   # leg in
    (3677.6, 3679.3, 3677.2, 3678.9),   # leg in
    (3678.9, 3685.5, 3678.6, 3684.5),   # leg in (big)
    (3684.6, 3686.3, 3682.3, 3683.8),   # vertex
    (3683.8, 3687.3, 3682.1, 3685.0),   # vertex
    (3685.0, 3685.0, 3678.5, 3678.7),   # leg out
    (3678.7, 3679.0, 3672.0, 3672.4),   # leg out, more than half a zone below
]


def test_two_candle_vertex_is_not_a_real_m() -> None:
    # 20 Nov 2026: a real M needs two tops on base candles with a dip
    # BETWEEN them, so a 2-candle vertex can't be one. Without the M / W
    # rule the detector still finds it.
    df = frame([(3681.0, 3681.2, 3676.9, 3679.1), (3679.1, 3679.3, 3675.9, 3676.1)] + TWO_CANDLE_M)
    loose = detect_vertex_patterns(df, VertexConfig(require_mw=False, min_base=2))
    real = detect_vertex_patterns(df, VertexConfig(require_mw=True, min_base=2))
    assert [p for p in loose if p.direction == "SELL" and p.base.candle_count == 2]
    assert not [p for p in real if p.direction == "SELL" and p.base.candle_count == 2]


def test_two_candle_spike_still_rejected() -> None:
    # same move, but the second vertex candle closes LOWER (no second top)
    rows = list(TWO_CANDLE_M)
    rows[4] = (3683.8, 3687.3, 3682.1, 3683.0)
    df = frame([(3681.0, 3681.2, 3676.9, 3679.1), (3679.1, 3679.3, 3675.9, 3676.1)] + rows)
    two = detect_vertex_patterns(df, VertexConfig(require_mw=True, min_base=2))
    assert not [p for p in two if p.direction == "SELL" and p.base.candle_count == 2]


# --------------------------------------------------------------------------- #
# 20 Nov 2026: Tommy's real W / M rule, on the wicks, for every bot and
# timeframe. Buys shown (W); sells are the mirror (M).
# --------------------------------------------------------------------------- #

from bot.strategy.vertex import is_real_mw


def _hl(rows):
    """rows of (high, low) -> arrays; index 0 = candle before the base."""
    import numpy as _np
    return _np.array([r[0] for r in rows], float), _np.array([r[1] for r in rows], float)


def _w(second_low: float):
    # candle before the base (end of the drop in), 5 base candles, first leg-out candle.
    return _hl([(24.0, 16.0),                       # end of the drop in
                (17.0, 13.0),                       # bottom 1 (wick 13.0)
                (19.0, 13.5), (19.5, 16.5),         # middle top 19.5
                (17.5, second_low),                 # bottom 2
                (18.0, 13.6) if second_low < 13.6 else (18.0, second_low + 0.4),
                (26.0, 17.0)])                      # leg out


@pytest.mark.parametrize("second_low,ok", [
    (13.2, True),    # A: level
    (14.6, True),    # B: about 25% of the zone higher
    (16.2, True),    # C: about half the zone higher
    (18.4, False),   # D: most of the zone away
])
def test_bottoms_at_most_half_the_zone_apart(second_low: float, ok: bool) -> None:
    h, l = _w(second_low)
    lo, hi = float(l[1:6].min()), float(h[1:6].max())
    assert is_real_mw(h, l, 1, 5, lo, hi, sell=False) is ok


def test_end_of_the_drop_in_is_not_a_bottom() -> None:
    # 20 Nov 2026 loser (BUY 3566.73-3576.78): drop in ends, chop, ONE lower
    # low, rally. The old close check called it a W; the real rule doesn't.
    h, l = _hl([(3581.5, 3572.0),                   # end of the drop in
                (3575.0, 3572.6), (3575.6, 3572.9), (3576.8, 3573.0),
                (3575.0, 3569.0), (3570.5, 3566.7), (3576.0, 3568.4),
                (3575.5, 3572.0),                   # small bounce down, higher low
                (3577.5, 3572.5)])                  # leg out
    assert not is_real_mw(h, l, 1, 7, 3566.7, 3576.8, sell=False)


def test_lower_low_between_the_bottoms_kills_the_w() -> None:
    h, l = _hl([(24.0, 16.0), (17.0, 13.0), (19.0, 14.0), (16.0, 6.0),
                (19.5, 14.0), (17.5, 13.2), (26.0, 17.0)])
    # 13.0 and 13.2 are level, but 6.0 between them is lower (and more than
    # half the zone below either): a V, not a W
    assert not is_real_mw(h, l, 1, 5, 6.0, 19.5, sell=False)


def test_tiny_middle_top_is_not_a_w() -> None:
    h, l = _hl([(24.0, 16.0), (14.0, 13.0), (13.3, 13.1), (14.0, 13.05), (20.0, 14.0)])
    assert not is_real_mw(h, l, 1, 3, 13.0, 20.0, sell=False)   # 0.25 pt pullback on a 7 pt zone


def test_real_m_mirror_for_sells() -> None:
    h, l = _w(13.2)
    mh, ml = 100.0 - l, 100.0 - h          # flip the W into an M
    lo, hi = float(ml[1:6].min()), float(mh[1:6].max())
    assert is_real_mw(mh, ml, 1, 5, lo, hi, sell=True)
    h, l = _w(18.4)
    mh, ml = 100.0 - l, 100.0 - h
    lo, hi = float(ml[1:6].min()), float(mh[1:6].max())
    assert not is_real_mw(mh, ml, 1, 5, lo, hi, sell=True)


def test_real_mw_is_the_default_everywhere() -> None:
    assert VertexConfig().require_mw is True
    cfg = StrategyPipelineConfig()
    assert cfg.vertex_require_mw is True and cfg.vertex_mw_max_apart == 0.5
