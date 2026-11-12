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
    tf_label,
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


class TestBosTrendRule:
    """Tommy's rule (12 Nov 2026): a close past the last swing = trend."""

    # Like 11-12 Nov: a higher low in the last swings, then a one-way drop
    # with no bounce big enough to make new swings.
    WATERFALL = _zigzag([4100, 4060, 4090, 4015, 4065, 4038, 4050, 3990], bars_per_leg=36)

    def test_old_rule_says_range_new_rule_sees_the_drop(self) -> None:
        df = _m5_from_closes(self.WATERFALL)
        old = DirectionGateConfig(trend_rule="swings")
        assert trend_state(df, old) == StructureState.RANGE
        assert trend_state(df, DirectionGateConfig()) == StructureState.DOWNTREND

    def test_buy_blocked_in_the_drop(self) -> None:
        df = _m5_from_closes(self.WATERFALL)
        cfg = DirectionGateConfig(allow_counter_trend_from_htf_zone=False)
        d = check_direction(direction="BUY", zone_bottom=3995, zone_top=3998, df_m5=df, config=cfg)
        assert not d.allowed and d.trend == StructureState.DOWNTREND
        assert "against M15 trend" in d.reason
        assert check_direction(direction="SELL", zone_bottom=4000, zone_top=4003, df_m5=df, config=cfg).allowed

    def test_close_above_last_swing_high_flips_to_uptrend(self) -> None:
        # The drop, then a rally that closes above the last swing high.
        df = _m5_from_closes(_zigzag([4100, 4060, 4090, 4040, 4070, 4020, 4045, 4030, 4080], bars_per_leg=36))
        assert trend_state(df, DirectionGateConfig()) == StructureState.UPTREND

    def test_m1_bot_reads_m5_and_says_so(self) -> None:
        df = _m5_from_closes(self.WATERFALL)
        cfg = DirectionGateConfig(trend_timeframe="5min", htf_timeframe="15min",
                                  allow_counter_trend_from_htf_zone=False)
        d = check_direction(direction="BUY", zone_bottom=3995, zone_top=3998, df_m5=df, config=cfg)
        assert "M5 trend" in d.reason and "M15 zone" in d.reason

    def test_labels(self) -> None:
        assert (tf_label("15min"), tf_label("5min"), tf_label("1h")) == ("M15", "M5", "H1")


class TestTrendSummary:
    """What the dashboard shows: state, since when, and the flip level."""

    def test_downtrend_with_flip_level(self) -> None:
        from bot.strategy.direction_gate import trend_summary

        df = _m5_from_closes(TestBosTrendRule.WATERFALL)
        s = trend_summary(df, DirectionGateConfig())
        assert s["state"] == "DOWNTREND" and s["timeframe"] == "M15" and s["rule"] == "bos"
        assert s["since"] is not None and s["down_below"] is None
        assert s["up_above"] is not None and s["up_above"] > df["close"].iloc[-1]

    def test_m1_bot_reads_m5(self) -> None:
        from bot.strategy.direction_gate import trend_summary

        df = _m5_from_closes(TestBosTrendRule.WATERFALL)
        s = trend_summary(df, DirectionGateConfig(trend_timeframe="5min"))
        assert s["timeframe"] == "M5" and s["state"] == "DOWNTREND"

    def test_m5_bot_reads_its_own_chart(self) -> None:
        """12 Nov 2026: M5 trend on M5 bars = the bars as they are."""
        from bot.strategy.direction_gate import trend_summary

        df = _m5_from_closes(TestBosTrendRule.WATERFALL)
        assert resample_ohlc(df, "5min")["close"].tolist() == df["close"].tolist()
        s = trend_summary(df, DirectionGateConfig(trend_timeframe="5min"))
        assert s["timeframe"] == "M5" and s["state"] == "DOWNTREND"

    def test_m1_bot_reads_its_own_chart(self) -> None:
        from bot.strategy.direction_gate import trend_summary

        df = _m5_from_closes(TestBosTrendRule.WATERFALL)
        df.index = pd.date_range("2026-01-05T00:00:00Z", periods=len(df), freq="1min", tz="UTC")
        assert len(resample_ohlc(df, "1min")) == len(df)
        s = trend_summary(df, DirectionGateConfig(trend_timeframe="1min"))
        assert s["timeframe"] == "M1" and s["state"] == "DOWNTREND"

    def test_too_little_data_is_range(self) -> None:
        from bot.strategy.direction_gate import trend_summary

        s = trend_summary(_m5_from_closes([100.0] * 6), DirectionGateConfig())
        assert s["state"] == "RANGE" and s["since"] is None


class TestResampleSourceLength:
    """The forming-bar check uses the real length of the source bars."""

    def _m1(self, n: int) -> pd.DataFrame:
        idx = pd.date_range("2026-11-12T09:00:00Z", periods=n, freq="1min", tz="UTC")
        c = np.arange(n, dtype=float) + 4000
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=idx)

    def test_m1_bars_into_m5_drop_the_unfinished_m5(self) -> None:
        # 09:00..09:06 closed M1 bars: the 09:05 M5 bar only has 2 of its 5.
        out = resample_ohlc(self._m1(7), "5min")
        assert list(out.index.strftime("%H:%M")) == ["09:00"]
        assert out["close"].iloc[-1] == 4004.0

    def test_m1_bars_into_m5_keep_a_finished_m5(self) -> None:
        out = resample_ohlc(self._m1(10), "5min")   # 09:00..09:09
        assert list(out.index.strftime("%H:%M")) == ["09:00", "09:05"]

    def test_weekend_gap_does_not_confuse_the_bar_length(self) -> None:
        a = self._m1(5)
        b = a.copy(); b.index = b.index + pd.Timedelta(days=2)
        assert dg.source_step(pd.concat([a, b])) == pd.Timedelta(minutes=1)
