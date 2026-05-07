"""Tests for ``bot.strategy.zone_refinement``.

Tests build Zones manually (via the ``mark_zone_from_*`` helpers) so we
can construct exact body shapes for boundary-condition tests without
fighting the swing detector's strength parameters.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Zone, mark_zone_from_m, mark_zone_from_w
from bot.strategy.zone_refinement import (
    RefinedZone,
    RefinementConfig,
    refine_zone,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ohlc(
    closes: list[float],
    opens: list[float] | None = None,
    lows: list[float] | None = None,
    highs: list[float] | None = None,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    n = len(closes)
    if opens is None:
        opens = list(closes)
    if lows is None:
        lows = list(closes)
    if highs is None:
        highs = list(closes)
    times = pd.date_range(start=start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [100] * n,
        },
        index=times,
    )


def make_w_zone(
    df: pd.DataFrame, low1_idx: int, low2_idx: int, peak_idx: int
) -> Zone:
    """Construct a Zone wrapping a WPattern at the given bar indices."""
    low1 = Swing(
        index=low1_idx,
        time=df.index[low1_idx],
        price=float(df["close"].iloc[low1_idx]),
        kind="LOW",
    )
    low2 = Swing(
        index=low2_idx,
        time=df.index[low2_idx],
        price=float(df["close"].iloc[low2_idx]),
        kind="LOW",
    )
    pattern = WPattern(
        low1=low1,
        low2=low2,
        peak_index=peak_idx,
        peak_time=df.index[peak_idx],
        peak_price=float(df["close"].iloc[peak_idx]),
        formed_at=df.index[low2_idx],
        completed=True,
    )
    return mark_zone_from_w(pattern, df)


def make_m_zone(
    df: pd.DataFrame, high1_idx: int, high2_idx: int, trough_idx: int
) -> Zone:
    high1 = Swing(
        index=high1_idx,
        time=df.index[high1_idx],
        price=float(df["close"].iloc[high1_idx]),
        kind="HIGH",
    )
    high2 = Swing(
        index=high2_idx,
        time=df.index[high2_idx],
        price=float(df["close"].iloc[high2_idx]),
        kind="HIGH",
    )
    pattern = MPattern(
        high1=high1,
        high2=high2,
        trough_index=trough_idx,
        trough_time=df.index[trough_idx],
        trough_price=float(df["close"].iloc[trough_idx]),
        formed_at=df.index[high2_idx],
        completed=True,
    )
    return mark_zone_from_m(pattern, df)


# Reusable W skeleton: lows at idx 2 and 9, peak at idx 6.
W_CLOSES_GOLD = [
    1910, 1905, 1900,        # bars 0..2 (low1 at 2)
    1903, 1906, 1908,        # bars 3..5
    1910,                    # bar 6 (peak)
    1908, 1906,              # bars 7..8
    1900,                    # bar 9 (low2)
    1903, 1906, 1910,        # bars 10..12
]

# Reusable M skeleton: highs at idx 2 and 9, trough at idx 6.
M_CLOSES_GOLD = [
    1890, 1895, 1900,        # bars 0..2 (high1 at 2)
    1897, 1894, 1892,        # bars 3..5
    1890,                    # bar 6 (trough)
    1892, 1894,              # bars 7..8
    1900,                    # bar 9 (high2)
    1897, 1894, 1890,        # bars 10..12
]


# --------------------------------------------------------------------------- #
# W refinement — geometric correctness
# --------------------------------------------------------------------------- #


class TestWRefinementBasics:
    def test_refinement_strips_wicks_below_body(self) -> None:
        # Both swing-low bars: bullish, close=1900, open=1895, low=1880
        # (deep wick down to 1880).
        # Initial zone (from zone_marking): top=1900, bottom=1880 (wick).
        # Refined: top=1900, bottom=1895 — wick from 1895 down to 1880 is
        # excluded.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1895
        opens[9] = 1895
        lows = list(W_CLOSES_GOLD)
        lows[2] = 1880
        lows[9] = 1880
        df = make_ohlc(W_CLOSES_GOLD, opens=opens, lows=lows)
        zone = make_w_zone(df, low1_idx=2, low2_idx=9, peak_idx=6)

        # Initial includes the wick.
        assert zone.bottom == 1880.0
        assert zone.top - zone.bottom == 20.0

        refined = refine_zone(zone, df)
        # Body of swing-low bars: top=1900, bottom=1895 each.
        assert refined.top == 1900.0
        assert refined.bottom == 1895.0
        assert refined.top - refined.bottom == 5.0
        # Refinement actually narrowed the zone.
        assert (refined.top - refined.bottom) < (zone.top - zone.bottom)

    def test_bullish_swing_low_bars(self) -> None:
        # close > open at swing lows. body_top = close, body_bottom = open.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1895  # bullish: open=1895, close=1900
        opens[9] = 1898  # bullish: open=1898, close=1900
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top == 1900.0          # max(close@2, close@9) = 1900
        assert refined.bottom == 1895.0       # min(open@2, open@9) = 1895

    def test_bearish_swing_low_bars(self) -> None:
        # close < open at swing lows. body_top = open, body_bottom = close.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1908  # bearish: open=1908, close=1900
        opens[9] = 1905  # bearish: open=1905, close=1900
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        # body_top of each bar = open. Refined top = max(1908, 1905) = 1908.
        # body_bottom of each bar = close. Refined bottom = min(1900, 1900) = 1900.
        assert refined.top == 1908.0
        assert refined.bottom == 1900.0

    def test_mixed_bullish_bearish_swing_lows(self) -> None:
        # low1 bullish (open=1895, close=1900), low2 bearish (open=1905, close=1900).
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1895
        opens[9] = 1905
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        # Body extremes:
        #   low1: top=1900, bottom=1895
        #   low2: top=1905, bottom=1900
        assert refined.top == 1905.0
        assert refined.bottom == 1895.0
        assert refined.top - refined.bottom == 10.0


# --------------------------------------------------------------------------- #
# Refinement scope — key correctness test (PR #7 spec correction)
# --------------------------------------------------------------------------- #


class TestRefinementIgnoresPeak:
    """Verify refinement uses ONLY the swing-low bars, not the peak."""

    def test_high_peak_does_not_inflate_refined_top(self) -> None:
        # Peak bar at extremely high close (1950) — 50 points above lows.
        # If refinement naively included all bars in [low1.idx, low2.idx],
        # refined top would be 1950. With swing-bars-only, refined top
        # stays at the swing-low bars' bodies.
        closes = [
            1910, 1905, 1900,      # bars 0..2
            1925, 1940, 1950,      # bars 3..5
            1950,                  # bar 6 (peak)
            1950, 1925,            # bars 7..8
            1900,                  # bar 9 (low2)
            1903, 1906, 1910,      # bars 10..12
        ]
        df = make_ohlc(closes)  # all OHLC == close (dojis)
        zone = make_w_zone(df, 2, 9, 6)

        refined = refine_zone(zone, df)
        # Both swing-low bars are dojis at 1900. Refined zone is degenerate
        # at 1900 — explicitly NOT pulled up to the peak's 1950.
        assert refined.top == 1900.0
        assert refined.bottom == 1900.0
        # Width 0 → fails size filter.
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"


# --------------------------------------------------------------------------- #
# Size filter — boundary conditions
# --------------------------------------------------------------------------- #


class TestSizeFilterBoundaries:
    def test_width_at_minimum_threshold_is_tradeable(self) -> None:
        # Refined width exactly 5.0 points — inclusive lower bound.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1895  # body_bottom = 1895
        # opens[9] defaults to close (1900) — body_bottom = 1900
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top == 1900.0
        assert refined.bottom == 1895.0
        assert refined.top - refined.bottom == 5.0
        assert refined.is_tradeable is True
        assert refined.rejection_reason is None

    def test_width_just_below_minimum_rejected(self) -> None:
        # Refined width 4.99 — should fail size filter.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1895.01
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top - refined.bottom == pytest.approx(4.99)
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"

    def test_width_at_maximum_threshold_is_tradeable(self) -> None:
        # Refined width exactly 80.0 points — inclusive upper bound.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1820  # body_bottom = 1820, body_top = 1900 → width 80
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top - refined.bottom == 80.0
        assert refined.is_tradeable is True
        assert refined.rejection_reason is None

    def test_width_just_above_maximum_rejected(self) -> None:
        # Refined width 80.01 — should fail size filter.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1819.99
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top - refined.bottom == pytest.approx(80.01)
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_WIDE"


# --------------------------------------------------------------------------- #
# Doji-only zones
# --------------------------------------------------------------------------- #


class TestDojiOnly:
    def test_both_swing_lows_doji_at_same_close(self) -> None:
        # Both pivot bars are dojis (open == close) at the SAME close.
        # Refined zone has zero width → ZONE_TOO_NARROW.
        df = make_ohlc(W_CLOSES_GOLD)  # all opens default to closes
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top == 1900.0
        assert refined.bottom == 1900.0
        assert refined.top == refined.bottom
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"

    def test_dojis_at_slightly_different_closes(self) -> None:
        # Both dojis but closes differ by 0.05 (still within 0.1% tolerance).
        # Refined width = 0.05 → ZONE_TOO_NARROW (below 5 pts).
        closes = list(W_CLOSES_GOLD)
        closes[9] = 1900.05  # low2 close 0.05 above low1 close
        df = make_ohlc(closes)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top == pytest.approx(1900.05)
        assert refined.bottom == 1900.0
        assert refined.top - refined.bottom == pytest.approx(0.05)
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"


# --------------------------------------------------------------------------- #
# M refinement — mirror geometry
# --------------------------------------------------------------------------- #


class TestMRefinement:
    def test_strips_wicks_above_body(self) -> None:
        # Both swing-high bars: bearish, close=1900, open=1905, high=1920.
        # Initial zone: bottom=1900, top=1920 (wick).
        # Refined: bottom=1900, top=1905 — wick from 1905 to 1920 excluded.
        opens = list(M_CLOSES_GOLD)
        opens[2] = 1905
        opens[9] = 1905
        highs = list(M_CLOSES_GOLD)
        highs[2] = 1920
        highs[9] = 1920
        df = make_ohlc(M_CLOSES_GOLD, opens=opens, highs=highs)
        zone = make_m_zone(df, high1_idx=2, high2_idx=9, trough_idx=6)

        # Initial top includes the wick.
        assert zone.top == 1920.0

        refined = refine_zone(zone, df)
        # body_top = max(open, close) = 1905; body_bottom = 1900.
        assert refined.top == 1905.0
        assert refined.bottom == 1900.0
        assert refined.top - refined.bottom == 5.0
        assert (refined.top - refined.bottom) < (zone.top - zone.bottom)

    def test_size_filter_too_wide_for_m(self) -> None:
        # body_top = 1985 (open), body_bottom = 1900 (close) → width 85.
        opens = list(M_CLOSES_GOLD)
        opens[2] = 1985
        df = make_ohlc(M_CLOSES_GOLD, opens=opens)
        zone = make_m_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top - refined.bottom == 85.0
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_WIDE"

    def test_size_filter_tradeable_for_m(self) -> None:
        # body_top = 1920, body_bottom = 1900 → width 20.
        opens = list(M_CLOSES_GOLD)
        opens[2] = 1920
        df = make_ohlc(M_CLOSES_GOLD, opens=opens)
        zone = make_m_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.top - refined.bottom == 20.0
        assert refined.is_tradeable is True
        assert refined.direction == "SELL"


# --------------------------------------------------------------------------- #
# Metadata pass-through and original_zone preservation
# --------------------------------------------------------------------------- #


class TestMetadata:
    def test_original_zone_preserved_by_identity(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.original_zone is zone  # identity, not equality

    def test_direction_passed_through_for_w(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.direction == "BUY"

    def test_direction_passed_through_for_m(self) -> None:
        df = make_ohlc(M_CLOSES_GOLD)
        zone = make_m_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.direction == "SELL"

    def test_formed_at_preserved(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.formed_at == zone.formed_at
        assert refined.formed_at == df.index[9]  # low2 bar's time

    def test_source_pattern_preserved(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)
        assert refined.source_pattern is zone.source_pattern


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_default_config_used_when_none_passed(self) -> None:
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1895  # width 5 — at default min
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(zone, df)  # no config arg
        assert refined.is_tradeable is True

    def test_custom_min_threshold_respected(self) -> None:
        # Width 5 fails when min is raised to 6.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1895
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(
            zone, df, RefinementConfig(zone_min_size_points=6.0)
        )
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"

    def test_custom_max_threshold_respected(self) -> None:
        # Width 50 fails when max is lowered to 40.
        opens = list(W_CLOSES_GOLD)
        opens[2] = 1850  # body_bottom = 1850, width = 50
        df = make_ohlc(W_CLOSES_GOLD, opens=opens)
        zone = make_w_zone(df, 2, 9, 6)
        refined = refine_zone(
            zone, df, RefinementConfig(zone_max_size_points=40.0)
        )
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_WIDE"


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_open_column_rejected(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        zone = make_w_zone(df, 2, 9, 6)
        df_no_open = df.drop(columns=["open"])
        with pytest.raises(ValueError, match="'open' and 'close'"):
            refine_zone(zone, df_no_open)

    def test_missing_close_column_rejected(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        zone = make_w_zone(df, 2, 9, 6)
        df_no_close = df.drop(columns=["close"])
        with pytest.raises(ValueError, match="'open' and 'close'"):
            refine_zone(zone, df_no_close)

    def test_index_out_of_range_rejected(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        # Build a Zone whose pattern's low2 sits past the df.
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        bad_pattern = WPattern(
            low1=Swing(index=2, time=ts, price=1900.0, kind="LOW"),
            low2=Swing(index=999, time=ts, price=1900.0, kind="LOW"),
            peak_index=6,
            peak_time=ts,
            peak_price=1910.0,
            formed_at=ts,
            completed=True,
        )
        bad_zone = Zone(
            direction="BUY",
            top=1900.0,
            bottom=1900.0,
            formed_at=ts,
            source_pattern=bad_pattern,
        )
        with pytest.raises(ValueError, match="out of df range"):
            refine_zone(bad_zone, df)

    def test_unknown_pattern_type_rejected(self) -> None:
        df = make_ohlc(W_CLOSES_GOLD)
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        # Zone with a non-pattern source_pattern.
        bogus_zone = Zone(
            direction="BUY",
            top=1900.0,
            bottom=1900.0,
            formed_at=ts,
            source_pattern="not a pattern",  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="unsupported source_pattern"):
            refine_zone(bogus_zone, df)


# --------------------------------------------------------------------------- #
# End-to-end: zone_marking → zone_refinement
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_w_pipeline_with_realistic_wicks(self) -> None:
        # Realistic Gold scenario: dojis at swing lows with deep wicks.
        # Initial zone is wide (includes wick); refined zone is degenerate
        # (both bodies are flat at the swing close).
        opens = list(W_CLOSES_GOLD)  # all dojis
        lows = list(W_CLOSES_GOLD)
        lows[2] = 1885
        lows[9] = 1885
        df = make_ohlc(W_CLOSES_GOLD, opens=opens, lows=lows)
        zone = make_w_zone(df, 2, 9, 6)

        assert zone.top == 1900.0
        assert zone.bottom == 1885.0
        assert zone.top - zone.bottom == 15.0

        refined = refine_zone(zone, df)
        # Bodies are dojis at 1900 → degenerate refined zone.
        assert refined.top == 1900.0
        assert refined.bottom == 1900.0
        # Refined narrows even though it ends up degenerate.
        assert (refined.top - refined.bottom) <= (zone.top - zone.bottom)
        # Fails size filter — correct outcome.
        assert refined.is_tradeable is False
