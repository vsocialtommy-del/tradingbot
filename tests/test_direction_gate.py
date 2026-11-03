"""Tests for the M15 direction gate with the fresh-H1-zone exception."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.strategy import direction_gate as dg
from bot.strategy.direction_gate import (
    DirectionGateConfig,
    check_direction,
    resample_ohlc,
    trend_state,
)
from bot.strategy.structure import StructureState


def _m5_from_closes(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05T00:00:00Z", periods=len(closes), freq="5min", tz="UTC")
    c = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 0.3, "low": c - 0.3, "close": c, "volume": 1},
        index=idx,
    )


def _zigzag(points: list[float], bars_per_leg: int = 24) -> list[float]:
    out: list[float] = []
    for a, b in zip(points, points[1:]):
        out.extend(np.linspace(a, b, bars_per_leg, endpoint=False).tolist())
    out.append(points[-1])
    return out


UP = _zigzag([100, 110, 105, 115, 110, 120, 115, 125, 120, 130])
DOWN = _zigzag([130, 120, 125, 115, 120, 110, 115, 105, 110, 100])


class TestResample:
    def test_m5_to_m15_ohlc(self) -> None:
        df = _m5_from_closes([1, 2, 3, 4, 5, 6])
        out = resample_ohlc(df, "15min")
        assert len(out) == 2
        assert out.iloc[0]["open"] == 1 and out.iloc[0]["close"] == 3
        assert out.iloc[0]["high"] == pytest.approx(3.3)
        assert out.iloc[1]["low"] == pytest.approx(3.7)

    def test_drops_forming_bar(self) -> None:
        df = _m5_from_closes([1, 2, 3, 4])  # second M15 bar has only 1 of 3
        assert len(resample_ohlc(df, "15min")) == 1


class TestTrendState:
    def test_uptrend(self) -> None:
        assert trend_state(_m5_from_closes(UP), DirectionGateConfig()) == StructureState.UPTREND

    def test_downtrend(self) -> None:
        assert trend_state(_m5_from_closes(DOWN), DirectionGateConfig()) == StructureState.DOWNTREND


class TestCheckDirection:
    def test_with_trend_allowed(self) -> None:
        d = check_direction(direction="BUY", zone_bottom=118, zone_top=120,
                            df_m5=_m5_from_closes(UP))
        assert d.allowed and d.trend == StructureState.UPTREND

    def test_against_trend_blocked_without_htf_zone(self, monkeypatch) -> None:
        monkeypatch.setattr(dg, "fresh_htf_zones", lambda *a, **k: [])
        d = check_direction(direction="SELL", zone_bottom=128, zone_top=130,
                            df_m5=_m5_from_closes(UP))
        assert not d.allowed

    def test_against_trend_allowed_inside_fresh_h1_zone(self, monkeypatch) -> None:
        monkeypatch.setattr(dg, "fresh_htf_zones", lambda *a, **k: [("SELL", 127.0, 133.0)])
        d = check_direction(direction="SELL", zone_bottom=128, zone_top=130,
                            df_m5=_m5_from_closes(UP))
        assert d.allowed and d.htf_zone == (127.0, 133.0)

    def test_touching_h1_zone_counts(self, monkeypatch) -> None:
        monkeypatch.setattr(dg, "fresh_htf_zones", lambda *a, **k: [("BUY", 95.0, 98.0)])
        d = check_direction(direction="BUY", zone_bottom=98.0, zone_top=99.5,
                            df_m5=_m5_from_closes(DOWN))
        assert d.allowed

    def test_h1_zone_in_wrong_direction_does_not_help(self, monkeypatch) -> None:
        monkeypatch.setattr(dg, "fresh_htf_zones", lambda *a, **k: [("BUY", 127.0, 133.0)])
        d = check_direction(direction="SELL", zone_bottom=128, zone_top=130,
                            df_m5=_m5_from_closes(UP))
        assert not d.allowed

    def test_range_allows_both(self, monkeypatch) -> None:
        monkeypatch.setattr(dg, "trend_state", lambda *a, **k: StructureState.RANGE)
        for direction in ("BUY", "SELL"):
            d = check_direction(direction=direction, zone_bottom=1, zone_top=2,
                                df_m5=_m5_from_closes(UP))
            assert d.allowed

    def test_disabled_allows_everything(self) -> None:
        d = check_direction(direction="SELL", zone_bottom=128, zone_top=130,
                            df_m5=_m5_from_closes(UP),
                            config=DirectionGateConfig(enabled=False))
        assert d.allowed
