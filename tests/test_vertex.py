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
    (4015.6, 4016.2, 4012.0, 4013.5),
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
