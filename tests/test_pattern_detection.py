"""Tests for ``bot.strategy.pattern_detection``.

Each test follows the same pattern as ``test_structure.py``:
    1. Build a synthetic OHLC DataFrame from a list of close prices.
    2. Run a detector or helper.
    3. Assert specific patterns / fields / counts.

OHLC == close throughout because pattern detection only reads closes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.pattern_detection import (
    MPattern,
    PatternConfig,
    PatternType,
    WPattern,
    _highest_close_between,
    _lowest_close_between,
    _peak_clears_threshold,
    _trough_clears_threshold,
    _within_tolerance,
    detect_latest_m,
    detect_latest_w,
    detect_m_patterns,
    detect_w_patterns,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_df(closes: list[float], start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    times = pd.date_range(start=start, periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        },
        index=times,
    )


# Canonical patterns reused across tests. Strength=2 in all cases.

# Two lows of 14 with a peak of 22 between them. 13 bars.
CLEAN_W_CLOSES = [20, 18, 14, 16, 18, 20, 22, 20, 18, 14, 16, 18, 20]
# Mirror — two highs of 16 with a trough of 8 between them.
CLEAN_M_CLOSES = [10, 12, 16, 14, 12, 10, 8, 10, 12, 16, 14, 12, 10]


# --------------------------------------------------------------------------- #
# Clean W / M detection
# --------------------------------------------------------------------------- #


class TestCleanW:
    def test_textbook_w_detected(self) -> None:
        df = make_df(CLEAN_W_CLOSES)
        patterns = detect_w_patterns(df, PatternConfig(swing_strength=2))
        assert len(patterns) == 1
        w = patterns[0]
        assert w.low1.index == 2
        assert w.low1.price == 14.0
        assert w.low2.index == 9
        assert w.low2.price == 14.0
        assert w.peak_price == 22.0
        assert w.peak_index == 6
        assert w.completed is True
        assert w.pattern_type == PatternType.W
        assert w.formed_at == w.low2.time

    def test_detect_latest_w_returns_the_one(self) -> None:
        df = make_df(CLEAN_W_CLOSES)
        w = detect_latest_w(df, PatternConfig(swing_strength=2))
        assert w is not None
        assert w.low2.index == 9


class TestCleanM:
    def test_textbook_m_detected(self) -> None:
        df = make_df(CLEAN_M_CLOSES)
        patterns = detect_m_patterns(df, PatternConfig(swing_strength=2))
        assert len(patterns) == 1
        m = patterns[0]
        assert m.high1.index == 2
        assert m.high1.price == 16.0
        assert m.high2.index == 9
        assert m.high2.price == 16.0
        assert m.trough_price == 8.0
        assert m.trough_index == 6
        assert m.completed is True
        assert m.pattern_type == PatternType.M

    def test_detect_latest_m_returns_the_one(self) -> None:
        df = make_df(CLEAN_M_CLOSES)
        m = detect_latest_m(df, PatternConfig(swing_strength=2))
        assert m is not None
        assert m.high2.index == 9


# --------------------------------------------------------------------------- #
# Tolerance — out-of-tolerance pivots
# --------------------------------------------------------------------------- #


class TestToleranceFails:
    def test_w_with_lows_too_far_apart_not_detected(self) -> None:
        # Lows at 100.00 and 100.50 → diff_pct ≈ 0.499% > 0.1% default.
        closes = [
            105, 102, 100.00, 102, 105,
            108, 110, 108, 105,
            102, 100.50, 102, 105,
        ]
        df = make_df(closes)
        assert detect_w_patterns(df, PatternConfig(swing_strength=2)) == []

    def test_m_with_highs_too_far_apart_not_detected(self) -> None:
        closes = [
            95, 98, 100.00, 98, 95,
            92, 90, 92, 95,
            98, 100.50, 98, 95,
        ]
        df = make_df(closes)
        assert detect_m_patterns(df, PatternConfig(swing_strength=2)) == []


# --------------------------------------------------------------------------- #
# Tolerance — exactly at the boundary (just under / just over)
# --------------------------------------------------------------------------- #


class TestToleranceBoundary:
    def test_just_under_tolerance_detected(self) -> None:
        # Lows 100.00, 100.10. avg=100.05.
        # diff_pct = 0.10 / 100.05 * 100 ≈ 0.0999500%   < 0.1%.
        closes = [
            105, 102, 100.00, 102, 105,
            108, 110, 108, 105,
            102, 100.10, 102, 105,
        ]
        df = make_df(closes)
        ws = detect_w_patterns(df, PatternConfig(swing_strength=2))
        assert len(ws) == 1

    def test_just_over_tolerance_not_detected(self) -> None:
        # Lows 100.00, 100.30. avg=100.15.
        # diff_pct = 0.30 / 100.15 * 100 ≈ 0.2995%      > 0.1%.
        closes = [
            105, 102, 100.00, 102, 105,
            108, 110, 108, 105,
            102, 100.30, 102, 105,
        ]
        df = make_df(closes)
        assert detect_w_patterns(df, PatternConfig(swing_strength=2)) == []

    def test_widening_tolerance_admits_pattern(self) -> None:
        # Same data as just_over above, but with a wider tolerance.
        closes = [
            105, 102, 100.00, 102, 105,
            108, 110, 108, 105,
            102, 100.30, 102, 105,
        ]
        df = make_df(closes)
        # 0.5% tolerance comfortably accommodates 0.2995%.
        ws = detect_w_patterns(
            df, PatternConfig(swing_strength=2, pattern_tolerance_pct=0.5)
        )
        assert len(ws) == 1


# --------------------------------------------------------------------------- #
# Peak / trough threshold
# --------------------------------------------------------------------------- #


class TestPeakThreshold:
    def test_peak_too_low_not_detected(self) -> None:
        # Two lows at 100, peak between only ~0.1% above → fails 0.2% default.
        closes = [
            105, 102, 100, 100.05, 100.1, 100.05, 100, 102, 105,
        ]
        df = make_df(closes)
        assert detect_w_patterns(df, PatternConfig(swing_strength=2)) == []

    def test_peak_just_above_threshold_detected(self) -> None:
        # Lows at 100, peak at 100.30 → 0.3% above → passes 0.2% default.
        # We have to design a swing-detectable peak: closes must dip
        # before the peak and after, so the peak qualifies as a high
        # in its window. (Detection only cares about the highest close
        # *between* the lows, not whether it's a swing — so we don't
        # need a swing here, just a number.)
        closes = [
            105, 102, 100,
            100.10, 100.30, 100.10,  # peak at index 4
            100, 102, 105,
        ]
        df = make_df(closes)
        ws = detect_w_patterns(df, PatternConfig(swing_strength=2))
        assert len(ws) == 1
        assert ws[0].peak_price == pytest.approx(100.30)

    def test_trough_too_high_not_detected_for_m(self) -> None:
        # Two highs at 100, trough only 0.1% below → fails 0.2% default.
        closes = [
            95, 98, 100, 99.95, 99.9, 99.95, 100, 98, 95,
        ]
        df = make_df(closes)
        assert detect_m_patterns(df, PatternConfig(swing_strength=2)) == []


# --------------------------------------------------------------------------- #
# Lookback window
# --------------------------------------------------------------------------- #


class TestLookback:
    def test_w_older_than_lookback_not_returned(self) -> None:
        # 13 bars of clean W, then 60 bars of "noise" pushing it out of
        # the lookback=50 window. The W's low2 is at index 9; the latest
        # bar is index 72; so low2 is 63 bars old — outside the 50-bar
        # window.
        closes = list(CLEAN_W_CLOSES) + [
            22, 21, 20, 19, 21, 23, 22, 21, 20, 22, 24, 23, 22, 21, 23, 25,
            24, 23, 22, 24, 26, 25, 24, 23, 25, 27, 26, 25, 24, 26, 28, 27,
            26, 25, 27, 29, 28, 27, 26, 28, 30, 29, 28, 27, 29, 31, 30, 29,
            28, 30, 32, 31, 30, 29, 31, 33, 32, 31, 30, 32,
        ]
        df = make_df(closes)
        assert (
            detect_w_patterns(
                df, PatternConfig(swing_strength=2, lookback_bars=50)
            )
            == []
        )

    def test_widening_lookback_brings_old_pattern_back(self) -> None:
        # Same data, lookback=200 → the old W is now in scope.
        closes = list(CLEAN_W_CLOSES) + [
            22, 21, 20, 19, 21, 23, 22, 21, 20, 22, 24, 23, 22, 21, 23, 25,
            24, 23, 22, 24, 26, 25, 24, 23, 25, 27, 26, 25, 24, 26, 28, 27,
            26, 25, 27, 29, 28, 27, 26, 28, 30, 29, 28, 27, 29, 31, 30, 29,
            28, 30, 32, 31, 30, 29, 31, 33, 32, 31, 30, 32,
        ]
        df = make_df(closes)
        ws = detect_w_patterns(
            df, PatternConfig(swing_strength=2, lookback_bars=200)
        )
        assert len(ws) >= 1


# --------------------------------------------------------------------------- #
# Pattern still forming (not yet complete)
# --------------------------------------------------------------------------- #


class TestStillForming:
    def test_pattern_with_unconfirmed_second_low_not_returned(self) -> None:
        # Same shape as CLEAN_W_CLOSES truncated so the SECOND low
        # never gets its right-shoulder. With strength=2, swing detection
        # only checks bars [2..n-3]. If the would-be second low is at
        # index n-1 or n-2, it's never confirmed.
        # CLEAN_W_CLOSES has its second low at index 9. We truncate so
        # that bar 9 has only one bar after it (rather than the needed
        # two).
        closes = CLEAN_W_CLOSES[:11]  # bars 0..10; second low at idx 9 has only bar 10 after.
        df = make_df(closes)
        # detect_swings with strength=2 won't return a swing at idx 9
        # because the right shoulder is incomplete. Hence no W.
        assert detect_w_patterns(df, PatternConfig(swing_strength=2)) == []

    def test_pattern_with_only_one_low_not_returned(self) -> None:
        # Truncate even earlier — only the first low is confirmed.
        closes = CLEAN_W_CLOSES[:7]
        df = make_df(closes)
        assert detect_w_patterns(df, PatternConfig(swing_strength=2)) == []


# --------------------------------------------------------------------------- #
# Multiple patterns — most recent returned
# --------------------------------------------------------------------------- #


class TestMultiplePatterns:
    """Three identical W shapes back-to-back, sharing low prices."""

    # Three lows at price 100 at indices 2, 10, 18, with peaks of 110
    # at indices 6, 14. Pairs (2,10), (2,18), (10,18) all qualify as Ws.
    THREE_BOTTOMS = [
        105, 102, 100, 102, 105,
        108, 110, 108, 105, 102,
        100, 102, 105, 108, 110,
        108, 105, 102, 100, 102, 105,
    ]

    def test_all_pairs_detected(self) -> None:
        df = make_df(self.THREE_BOTTOMS)
        ws = detect_w_patterns(df, PatternConfig(swing_strength=2))
        # 3 lows → 3 unique pairs.
        assert len(ws) == 3
        pairs = {(w.low1.index, w.low2.index) for w in ws}
        assert pairs == {(2, 10), (2, 18), (10, 18)}

    def test_latest_w_picks_tightest_most_recent(self) -> None:
        df = make_df(self.THREE_BOTTOMS)
        latest = detect_latest_w(df, PatternConfig(swing_strength=2))
        # Both (2,18) and (10,18) share latest low2=18; tie-break picks
        # the one with the latest low1 → (10, 18).
        assert latest is not None
        assert latest.low1.index == 10
        assert latest.low2.index == 18


# --------------------------------------------------------------------------- #
# Insufficient data
# --------------------------------------------------------------------------- #


class TestInsufficientData:
    def test_empty_df(self) -> None:
        df = make_df([])
        assert detect_w_patterns(df) == []
        assert detect_m_patterns(df) == []
        assert detect_latest_w(df) is None
        assert detect_latest_m(df) is None

    def test_too_few_bars_for_any_swing(self) -> None:
        # 4 bars — strength=2 needs 5 minimum.
        df = make_df([10, 9, 8, 9])
        assert detect_w_patterns(df, PatternConfig(swing_strength=2)) == []

    def test_only_one_low_in_data(self) -> None:
        # Single V-shape — one low, no second pivot.
        df = make_df([10, 8, 6, 8, 10])
        assert detect_w_patterns(df, PatternConfig(swing_strength=2)) == []


# --------------------------------------------------------------------------- #
# M and W in the same dataset — independent detection
# --------------------------------------------------------------------------- #


class TestMixedPatterns:
    """Data that contains both an M (early) and a W (later)."""

    # Indices 0..12: M shape with highs at 100, trough at 90.
    # Indices 13..24: W shape with lows at 88, peak at 100.
    MIXED = [
        # M section (highs at 100, trough at 90)
        95, 98, 100, 98, 95, 92, 90, 92, 95, 98, 100, 98, 95,
        # W section continues; new lows at 88
        92, 88, 92, 95, 98, 100, 98, 95, 92, 88, 92, 95,
    ]

    def test_w_detected_independently(self) -> None:
        df = make_df(self.MIXED)
        ws = detect_w_patterns(df, PatternConfig(swing_strength=2))
        # Only one pair of equal-priced lows: those at 88, indices 14 and 22.
        assert len(ws) == 1
        assert ws[0].low1.price == 88.0
        assert ws[0].low2.price == 88.0

    def test_m_detected_independently(self) -> None:
        df = make_df(self.MIXED)
        ms = detect_m_patterns(df, PatternConfig(swing_strength=2))
        # Highs at 100 appear at indices 2, 10, 18 → three M pairs.
        # All have the equal-price highs at 100.
        assert len(ms) == 3
        latest = max(ms, key=lambda m: (m.high2.index, m.high1.index))
        assert latest.high2.index == 18

    def test_w_does_not_appear_in_m_results_and_vice_versa(self) -> None:
        df = make_df(self.MIXED)
        ws = detect_w_patterns(df, PatternConfig(swing_strength=2))
        ms = detect_m_patterns(df, PatternConfig(swing_strength=2))
        # Different pivot types, no overlap.
        w_idxs = {(w.low1.index, w.low2.index) for w in ws}
        m_idxs = {(m.high1.index, m.high2.index) for m in ms}
        assert w_idxs.isdisjoint(m_idxs)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


class TestInternalHelpers:
    def test_within_tolerance_inclusive(self) -> None:
        # Pick numbers that produce diff_pct exactly 0.1.
        # 100, 100.1001001... gives diff_pct ≈ 0.1 (close enough for floats).
        # We test inclusive at a clean value: 100 vs 100.10010010... is hard
        # to construct exactly; instead use a wider, exact case.
        # |200-202|/201 * 100 = 200/201 ≈ 0.9950 → use tolerance=1.0:
        assert _within_tolerance(200, 202, 1.0) is True
        # And just over:
        assert _within_tolerance(200, 204, 1.0) is False  # diff_pct ≈ 1.98%

    def test_within_tolerance_zero_average_returns_false(self) -> None:
        assert _within_tolerance(0, 0, 1.0) is False

    def test_highest_close_between_empty_interval(self) -> None:
        import numpy as np

        closes = np.array([10.0, 12.0, 14.0])
        # Adjacent indices → no bars strictly between.
        offset, price = _highest_close_between(closes, 0, 1)
        assert offset is None

    def test_lowest_close_between_with_data(self) -> None:
        import numpy as np

        closes = np.array([10.0, 12.0, 8.0, 14.0, 10.0])
        offset, price = _lowest_close_between(closes, 0, 4)
        # Between idx 0 and 4: closes[1:4] = [12, 8, 14]. min=8 at offset 1.
        assert offset == 1
        assert price == 8.0

    def test_peak_clears_threshold(self) -> None:
        # higher_low=100, threshold 1% → min_peak=101.
        assert _peak_clears_threshold(101.0, 100.0, 1.0) is True
        assert _peak_clears_threshold(100.99, 100.0, 1.0) is False

    def test_trough_clears_threshold(self) -> None:
        # lower_high=100, threshold 1% → max_trough=99.
        assert _trough_clears_threshold(99.0, 100.0, 1.0) is True
        assert _trough_clears_threshold(99.01, 100.0, 1.0) is False


# --------------------------------------------------------------------------- #
# Sanity: dataclass shape
# --------------------------------------------------------------------------- #


class TestDataclassShape:
    def test_w_pattern_fields(self) -> None:
        df = make_df(CLEAN_W_CLOSES)
        w = detect_latest_w(df, PatternConfig(swing_strength=2))
        assert isinstance(w, WPattern)
        # `formed_at` matches low2.time
        assert w.formed_at == w.low2.time
        # peak_time matches the indexed bar
        assert w.peak_time == df.index[w.peak_index]

    def test_m_pattern_fields(self) -> None:
        df = make_df(CLEAN_M_CLOSES)
        m = detect_latest_m(df, PatternConfig(swing_strength=2))
        assert isinstance(m, MPattern)
        assert m.formed_at == m.high2.time
        assert m.trough_time == df.index[m.trough_index]
