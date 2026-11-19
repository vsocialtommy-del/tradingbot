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
FLAT = [(4014.0, 4014.6, 4013.6, 4014.2)] * 6
M_TOP = FLAT + [
    (4014.2, 4016.8, 4014.0, 4016.6),   # leg in (bull)
    (4016.6, 4018.6, 4014.8, 4018.4),   # leg in (bull)
    (4017.5, 4023.2, 4017.4, 4023.0),   # leg in (bull, big)
    (4023.0, 4025.3, 4020.0, 4020.4),   # vertex
    (4020.4, 4023.0, 4020.0, 4022.3),   # vertex
    (4022.3, 4022.7, 4019.9, 4020.2),   # vertex
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
    sells = [p for p in detect_vertex_patterns(frame(SPIKE_TOP)) if p.direction == "SELL"]
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
    assert len(got) > 20                  # the chart really has zones in it
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
