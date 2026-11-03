"""Tests for XAUTISM entry / SL / TP1 levels."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from bot.strategy.levels import LevelsConfig, compute_levels
from bot.strategy.pattern_detection import Base, Impulse, Pattern, PatternType
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


def _df(lows: list[float], highs: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-05T00:00:00Z", periods=len(lows), freq="5min", tz="UTC")
    mid = [(l + h) / 2 for l, h in zip(lows, highs)]
    return pd.DataFrame({"open": mid, "high": highs, "low": lows, "close": mid, "volume": 1}, index=idx)


def _zone(df: pd.DataFrame, *, direction: str, top: float, bottom: float,
          base_start: int, bos: float | None) -> ValidatedZone:
    t = df.index
    imp = Impulse(direction="DROP" if direction == "BUY" else "RALLY",
                  start_index=base_start - 1, end_index=base_start - 1,
                  start_time=t[base_start - 1], end_time=t[base_start - 1],
                  range_size=5, largest_body=5, candle_count=1)
    imp2 = replace(imp, direction="RALLY" if direction == "BUY" else "DROP",
                   start_index=base_start + 2, end_index=base_start + 2,
                   start_time=t[base_start + 2], end_time=t[base_start + 2])
    base = Base(start_index=base_start, end_index=base_start + 1, candle_count=2,
                top=top, bottom=bottom, range_size=top - bottom, largest_body=0.5)
    pt = PatternType.DBR if direction == "BUY" else PatternType.RBD
    pat = Pattern(pattern_type=pt, impulse_before=imp, base=base, impulse_after=imp2,
                  direction=direction, formed_at=t[base_start + 2])  # type: ignore[arg-type]
    z = Zone(direction=direction, top=top, bottom=bottom, formed_at=pat.formed_at,  # type: ignore[arg-type]
             source_pattern=pat)
    r = RefinedZone(direction=direction, top=top, bottom=bottom, formed_at=pat.formed_at,  # type: ignore[arg-type]
                    source_pattern=pat, is_tradeable=True, rejection_reason=None, original_zone=z)
    sw = None if bos is None else Swing(index=0, time=t[0], price=bos,
                                        kind="HIGH" if direction == "BUY" else "LOW")
    return ValidatedZone(direction=direction, top=top, bottom=bottom, formed_at=pat.formed_at,  # type: ignore[arg-type]
                         source_pattern=pat, refined_zone=r, is_strong_point=True,
                         validation_failures=[], broken_swing=sw, broken_at=None,
                         sl_anchor_swing=None)


N = 30
FLAT_LOW = [105.0] * N
FLAT_HIGH = [106.0] * N


class TestBuyLevels:
    def test_default_30pct_and_bos_tp(self) -> None:
        df = _df(FLAT_LOW, FLAT_HIGH)
        z = _zone(df, direction="BUY", top=104.0, bottom=100.0, base_start=20, bos=112.0)
        lv = compute_levels(z, df)
        assert lv.entry == 104.0
        assert lv.sl == pytest.approx(98.8)      # 100 - 0.3*4
        assert lv.sl_method == "PCT_30"
        assert lv.tp1 == pytest.approx(111.7)    # 112 - 0.3

    def test_swing_just_past_30pct_moves_sl_beyond_it(self) -> None:
        lows = FLAT_LOW.copy()
        lows[12] = 98.5   # swing low just below the 30% stop (98.8)
        df = _df(lows, FLAT_HIGH)
        z = _zone(df, direction="BUY", top=104.0, bottom=100.0, base_start=20, bos=112.0)
        lv = compute_levels(z, df)
        assert lv.sl_method == "SWING"
        assert lv.swing_price == 98.5
        assert lv.sl == pytest.approx(97.5)      # 1 point (10 pips) below

    def test_swing_inside_buffer_still_gets_one_point(self) -> None:
        lows = FLAT_LOW.copy()
        lows[12] = 99.5   # between zone bottom and the 30% stop
        df = _df(lows, FLAT_HIGH)
        z = _zone(df, direction="BUY", top=104.0, bottom=100.0, base_start=20, bos=112.0)
        lv = compute_levels(z, df)
        assert lv.sl == pytest.approx(98.5)      # 99.5 - 1.0, wider than 98.8

    def test_far_swing_ignored(self) -> None:
        lows = FLAT_LOW.copy()
        lows[12] = 90.0   # way beyond the window
        df = _df(lows, FLAT_HIGH)
        z = _zone(df, direction="BUY", top=104.0, bottom=100.0, base_start=20, bos=112.0)
        assert compute_levels(z, df).sl_method == "PCT_30"

    def test_no_bos_gives_no_tp(self) -> None:
        df = _df(FLAT_LOW, FLAT_HIGH)
        z = _zone(df, direction="BUY", top=104.0, bottom=100.0, base_start=20, bos=None)
        assert compute_levels(z, df).tp1 is None


class TestSellLevels:
    def test_mirror(self) -> None:
        highs = [99.0] * N
        highs[12] = 105.6  # swing high just above the 30% stop (105.2)
        df = _df([98.0] * N, highs)
        z = _zone(df, direction="SELL", top=104.0, bottom=100.0, base_start=20, bos=92.0)
        lv = compute_levels(z, df)
        assert lv.entry == 100.0
        assert lv.sl == pytest.approx(106.6)
        assert lv.sl_method == "SWING"
        assert lv.tp1 == pytest.approx(92.3)

    def test_custom_buffer(self) -> None:
        df = _df([98.0] * N, [99.0] * N)
        z = _zone(df, direction="SELL", top=104.0, bottom=100.0, base_start=20, bos=92.0)
        lv = compute_levels(z, df, LevelsConfig(tp_spread_buffer=0.5))
        assert lv.tp1 == pytest.approx(92.5)
