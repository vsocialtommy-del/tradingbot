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
