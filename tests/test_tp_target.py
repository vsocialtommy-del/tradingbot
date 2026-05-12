"""Tests for ``bot.strategy.tp_target``.

Pure-logic coverage of the local-peak detector that drives the new
TP1 source in the loosened entry flow.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.tp_target import find_nearest_local_peak


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_df_from_highs_lows(
    highs: list[float], lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build an OHLC df where the only fields that matter are high/low.

    open/close set to the midpoint so the df is still a "real" candle
    set; we just don't read those columns here.
    """
    n = len(highs)
    if lows is None:
        lows = [h - 1.0 for h in highs]
    opens = [(h + lo) / 2.0 for h, lo in zip(highs, lows)]
    closes = list(opens)
    times = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [100] * n},
        index=times,
    )


# --------------------------------------------------------------------------- #
# Local-high detection — BUY side
# --------------------------------------------------------------------------- #


class TestBuyLocalHigh:
    def test_simple_single_peak(self) -> None:
        # Bar 2 is a strict local high.
        highs = [100.0, 101.0, 102.0, 101.0, 100.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=100.5, direction="BUY")
        assert result == 102.0

    def test_nearest_by_price_when_multiple_peaks(self) -> None:
        # Three peaks above entry: 102, 105, 110. Nearest by PRICE = 102.
        highs = [100.0, 102.0, 100.0, 105.0, 100.0, 110.0, 100.0, 100.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=101.0, direction="BUY")
        assert result == 102.0

    def test_peak_below_entry_excluded(self) -> None:
        # Bar 2 peak at 100.5; entry is 101 → peak is BELOW entry → excluded.
        highs = [100.0, 100.5, 99.0, 100.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=101.0, direction="BUY")
        assert result is None

    def test_peak_equal_to_entry_excluded(self) -> None:
        # Strict > inequality: peak high == entry doesn't count.
        highs = [100.0, 101.0, 100.0, 100.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=101.0, direction="BUY")
        assert result is None

    def test_last_bar_never_qualifies(self) -> None:
        # Last bar's high is highest in the series, but it has no
        # right shoulder so it can't be a local peak yet.
        highs = [100.0, 100.0, 100.0, 105.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=100.5, direction="BUY")
        assert result is None

    def test_first_bar_never_qualifies(self) -> None:
        highs = [105.0, 100.0, 100.0, 100.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=100.5, direction="BUY")
        assert result is None

    def test_flat_shoulders_not_a_peak(self) -> None:
        # 100, 101, 101, 101, 100 — middle bar's high tied with both
        # neighbours; strict > fails on both sides.
        highs = [100.0, 101.0, 101.0, 101.0, 100.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=100.5, direction="BUY")
        assert result is None

    def test_lookback_window_clips_old_peaks(self) -> None:
        # An old peak at bar 2 (high=105), a recent stretch of bars,
        # all flat. With lookback=3 the old peak is outside the
        # window and we get None; with lookback >= 15 we see it.
        n = 20
        highs = [100.0] * n
        highs[2] = 105.0
        df = make_df_from_highs_lows(highs)
        # last_candidate = n-2 = 18. window for lookback=3 is [16, 18];
        # bar 2 is outside.
        assert find_nearest_local_peak(
            df, entry_price=101.0, direction="BUY", lookback_bars=3,
        ) is None
        # window for lookback=20 is [1, 18]; bar 2 is inside.
        assert find_nearest_local_peak(
            df, entry_price=101.0, direction="BUY", lookback_bars=20,
        ) == 105.0

    def test_no_peak_in_window_returns_none(self) -> None:
        # Monotonic-up series: no bar has a higher high than its right
        # neighbour → no local peaks at all.
        highs = [100.0, 101.0, 102.0, 103.0, 104.0]
        df = make_df_from_highs_lows(highs)
        result = find_nearest_local_peak(df, entry_price=100.5, direction="BUY")
        assert result is None


# --------------------------------------------------------------------------- #
# Local-low detection — SELL side (mirror)
# --------------------------------------------------------------------------- #


class TestSellLocalLow:
    def test_simple_single_low(self) -> None:
        # Bar 2 is a strict local low.
        lows = [100.0, 99.0, 98.0, 99.0, 100.0]
        highs = [lo + 1.0 for lo in lows]
        df = make_df_from_highs_lows(highs, lows=lows)
        result = find_nearest_local_peak(df, entry_price=99.5, direction="SELL")
        assert result == 98.0

    def test_nearest_by_price_when_multiple_lows(self) -> None:
        # Three lows below entry: 98, 95, 90. Nearest by PRICE = 98
        # (closest to entry from below = highest among qualifying).
        lows = [100.0, 98.0, 100.0, 95.0, 100.0, 90.0, 100.0, 100.0]
        highs = [lo + 1.0 for lo in lows]
        df = make_df_from_highs_lows(highs, lows=lows)
        result = find_nearest_local_peak(df, entry_price=99.0, direction="SELL")
        assert result == 98.0

    def test_low_above_entry_excluded(self) -> None:
        lows = [100.0, 99.5, 100.0, 100.0]
        highs = [lo + 1.0 for lo in lows]
        df = make_df_from_highs_lows(highs, lows=lows)
        result = find_nearest_local_peak(df, entry_price=99.0, direction="SELL")
        assert result is None

    def test_low_equal_to_entry_excluded(self) -> None:
        lows = [100.0, 99.0, 100.0, 100.0]
        highs = [lo + 1.0 for lo in lows]
        df = make_df_from_highs_lows(highs, lows=lows)
        result = find_nearest_local_peak(df, entry_price=99.0, direction="SELL")
        assert result is None


# --------------------------------------------------------------------------- #
# Errors / edge cases
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_high_column_raises(self) -> None:
        df = pd.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="high"):
            find_nearest_local_peak(df, entry_price=100.0, direction="BUY")

    def test_short_df_returns_none(self) -> None:
        # Fewer than 3 bars → no possible shoulders.
        df = make_df_from_highs_lows([100.0, 105.0])
        result = find_nearest_local_peak(df, entry_price=100.0, direction="BUY")
        assert result is None

    def test_invalid_lookback_raises(self) -> None:
        df = make_df_from_highs_lows([100.0, 101.0, 100.0])
        with pytest.raises(ValueError, match="lookback_bars"):
            find_nearest_local_peak(
                df, entry_price=100.0, direction="BUY", lookback_bars=0,
            )
