"""Tests for ``bot.strategy.imbalance``.

Tests build ValidatedZones manually so we can drive each approach
scenario without dragging the full validation chain through every test.
The DataFrame's bars after the formation index are where the approach
state machine runs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.imbalance import (
    ApproachEvent,
    ImbalanceConfig,
    ImbalanceZone,
    track_imbalance,
)
from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.structure import Swing
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ohlc(
    closes: list[float],
    opens: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    n = len(closes)
    if opens is None:
        opens = list(closes)
    if highs is None:
        highs = list(closes)
    if lows is None:
        lows = list(closes)
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


def make_w_validated_zone(
    df: pd.DataFrame,
    *,
    top: float = 1900.0,
    bottom: float = 1895.0,
    is_strong_point: bool = True,
    low1_idx: int = 2,
    low2_idx: int = 9,
    peak_idx: int = 6,
) -> ValidatedZone:
    """Build a Strong-Point-validated W zone for imbalance tests."""
    pattern = WPattern(
        low1=Swing(
            index=low1_idx,
            time=df.index[low1_idx],
            price=float(df["close"].iloc[low1_idx]),
            kind="LOW",
        ),
        low2=Swing(
            index=low2_idx,
            time=df.index[low2_idx],
            price=float(df["close"].iloc[low2_idx]),
            kind="LOW",
        ),
        peak_index=peak_idx,
        peak_time=df.index[peak_idx],
        peak_price=float(df["close"].iloc[peak_idx]),
        formed_at=df.index[low2_idx],
        completed=True,
    )
    initial = Zone(
        direction="BUY",
        top=top,
        bottom=bottom,
        formed_at=df.index[low2_idx],
        source_pattern=pattern,
    )
    refined = RefinedZone(
        direction="BUY",
        top=top,
        bottom=bottom,
        formed_at=df.index[low2_idx],
        source_pattern=pattern,
        is_tradeable=True,
        rejection_reason=None,
        original_zone=initial,
    )
    return ValidatedZone(
        direction="BUY",
        top=top,
        bottom=bottom,
        formed_at=df.index[low2_idx],
        source_pattern=pattern,
        is_tradeable=True,
        rejection_reason=None,
        original_zone=initial,
        refined_zone=refined,
        is_strong_point=is_strong_point,
        validation_failures=[] if is_strong_point else ["IMPULSE_TOO_WEAK"],
        bos_event=None,
    )


def make_m_validated_zone(
    df: pd.DataFrame,
    *,
    top: float = 1905.0,
    bottom: float = 1900.0,
    is_strong_point: bool = True,
    high1_idx: int = 2,
    high2_idx: int = 9,
    trough_idx: int = 6,
) -> ValidatedZone:
    pattern = MPattern(
        high1=Swing(
            index=high1_idx,
            time=df.index[high1_idx],
            price=float(df["close"].iloc[high1_idx]),
            kind="HIGH",
        ),
        high2=Swing(
            index=high2_idx,
            time=df.index[high2_idx],
            price=float(df["close"].iloc[high2_idx]),
            kind="HIGH",
        ),
        trough_index=trough_idx,
        trough_time=df.index[trough_idx],
        trough_price=float(df["close"].iloc[trough_idx]),
        formed_at=df.index[high2_idx],
        completed=True,
    )
    initial = Zone(
        direction="SELL",
        top=top,
        bottom=bottom,
        formed_at=df.index[high2_idx],
        source_pattern=pattern,
    )
    refined = RefinedZone(
        direction="SELL",
        top=top,
        bottom=bottom,
        formed_at=df.index[high2_idx],
        source_pattern=pattern,
        is_tradeable=True,
        rejection_reason=None,
        original_zone=initial,
    )
    return ValidatedZone(
        direction="SELL",
        top=top,
        bottom=bottom,
        formed_at=df.index[high2_idx],
        source_pattern=pattern,
        is_tradeable=True,
        rejection_reason=None,
        original_zone=initial,
        refined_zone=refined,
        is_strong_point=is_strong_point,
        validation_failures=[] if is_strong_point else ["IMPULSE_TOO_WEAK"],
        bos_event=None,
    )


# A 12-bar synthetic prefix where the W is "set up". The interesting
# imbalance bars get appended in each test. Closes form a basic W with
# low1 at idx 2 and low2 at idx 9.
#
# IMPORTANT: bars 10 and 11 (after low2) are deliberately placed FAR
# from the zone — well past the retreat threshold — so the state
# machine stays IDLE through them and each test's tail bars start
# from a clean IDLE state. Otherwise the prefix would silently start
# an approach that the tail bars would complete or contaminate.
W_PREFIX_CLOSES = [1910, 1905, 1900, 1903, 1906, 1908, 1910, 1908, 1906, 1900, 1920, 1925]
M_PREFIX_CLOSES = [1890, 1895, 1900, 1897, 1894, 1892, 1890, 1892, 1894, 1900, 1880, 1875]


def make_w_df(*tail_bars: dict[str, float]) -> pd.DataFrame:
    """Build a W-prefix DataFrame plus extra bars described by dict.

    Each tail dict has keys ``open``, ``high``, ``low``, ``close``.
    """
    closes = list(W_PREFIX_CLOSES)
    opens = list(W_PREFIX_CLOSES)
    highs = list(W_PREFIX_CLOSES)
    lows = list(W_PREFIX_CLOSES)
    for bar in tail_bars:
        closes.append(bar["close"])
        opens.append(bar["open"])
        highs.append(bar["high"])
        lows.append(bar["low"])
    return make_ohlc(closes, opens=opens, highs=highs, lows=lows)


def make_m_df(*tail_bars: dict[str, float]) -> pd.DataFrame:
    closes = list(M_PREFIX_CLOSES)
    opens = list(M_PREFIX_CLOSES)
    highs = list(M_PREFIX_CLOSES)
    lows = list(M_PREFIX_CLOSES)
    for bar in tail_bars:
        closes.append(bar["close"])
        opens.append(bar["open"])
        highs.append(bar["high"])
        lows.append(bar["low"])
    return make_ohlc(closes, opens=opens, highs=highs, lows=lows)


# Convenience for crafting imbalance tail bars. For BUY zone with top=1900,
# approach_distance=7.5 → band (1900, 1907.5]; retreat threshold 1912.5.
def bar_in_approach_band_buy(low: float = 1905.0, high: float = 1910.0) -> dict:
    return {"open": high - 1, "high": high, "low": low, "close": high - 1}


def bar_retreated_buy(low: float = 1915.0, high: float = 1920.0) -> dict:
    return {"open": low + 1, "high": high, "low": low, "close": high - 1}


def bar_in_zone_buy(low: float = 1898.0, high: float = 1905.0) -> dict:
    return {"open": high - 1, "high": high, "low": low, "close": high - 1}


def bar_above_zone_buy(low: float = 1920.0, high: float = 1925.0) -> dict:
    return {"open": low + 1, "high": high, "low": low, "close": high - 1}


# --------------------------------------------------------------------------- #
# Clean Imbalance — two completed approaches without tap
# --------------------------------------------------------------------------- #


class TestCleanImbalance:
    def test_two_approaches_qualifies(self) -> None:
        df = make_w_df(
            # bar 12: enters approach band
            bar_in_approach_band_buy(low=1905, high=1908),
            # bar 13: retreats fully → approach 1 complete
            bar_retreated_buy(low=1915, high=1920),
            # bar 14: idle, above zone
            bar_above_zone_buy(low=1920, high=1925),
            # bar 15: enters approach band again
            bar_in_approach_band_buy(low=1903, high=1908),
            # bar 16: retreats → approach 2 complete → QUALIFIED
            bar_retreated_buy(low=1913, high=1918),
            # bar 17: idle
            bar_above_zone_buy(low=1920, high=1925),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)

        assert result.approach_count == 2
        assert result.is_imbalance is True
        assert result.is_tapped is False
        assert result.qualified_at == df.index[16]  # completion time of 2nd
        assert len(result.approach_events) == 2

    def test_three_approaches_increments_counter_keeps_imbalance(self) -> None:
        # Same as above plus a third approach.
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1908),
            bar_retreated_buy(low=1915, high=1920),
            bar_above_zone_buy(low=1920, high=1925),
            bar_in_approach_band_buy(low=1903, high=1908),
            bar_retreated_buy(low=1913, high=1918),
            bar_above_zone_buy(low=1920, high=1925),
            bar_in_approach_band_buy(low=1904, high=1909),
            bar_retreated_buy(low=1916, high=1922),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.approach_count == 3
        assert result.is_imbalance is True
        # qualified_at still set to the 2nd approach's completion (bar 16),
        # not the 3rd.
        assert result.qualified_at == df.index[16]


# --------------------------------------------------------------------------- #
# One approach only — not enough to qualify
# --------------------------------------------------------------------------- #


class TestSingleApproach:
    def test_one_approach_does_not_qualify(self) -> None:
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1908),
            bar_retreated_buy(low=1915, high=1920),
            bar_above_zone_buy(low=1922, high=1928),
            bar_above_zone_buy(low=1925, high=1930),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.approach_count == 1
        assert result.is_imbalance is False
        assert result.qualified_at is None
        assert result.is_tapped is False

    def test_approach_started_but_never_retreated_not_counted(self) -> None:
        # Bar enters band, next bars stay in band but never retreat fully.
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1908),
            bar_in_approach_band_buy(low=1903, high=1907),
            # high reaches 1912 — still BELOW retreat threshold of 1912.5
            {"open": 1907, "high": 1912, "low": 1907, "close": 1911},
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        # No completed approach since no bar.low >= 1912.5.
        assert result.approach_count == 0
        assert result.is_imbalance is False


# --------------------------------------------------------------------------- #
# Tap before any approaches
# --------------------------------------------------------------------------- #


class TestTappedImmediately:
    def test_zone_tapped_on_first_bar_no_approaches(self) -> None:
        df = make_w_df(
            bar_in_zone_buy(low=1895, high=1905),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.approach_count == 0
        assert result.is_imbalance is False
        assert result.is_tapped is True
        assert result.tapped_at == df.index[12]
        assert result.qualified_at is None

    def test_zone_tap_at_exact_top_counts_as_tap(self) -> None:
        # bar.low == zone.top exactly.
        df = make_w_df(
            {"open": 1910, "high": 1915, "low": 1900, "close": 1908},
        )
        zone = make_w_validated_zone(df, top=1900)
        result = track_imbalance(zone, df)
        assert result.is_tapped is True
        assert result.approach_count == 0


# --------------------------------------------------------------------------- #
# Tap after qualifying — was Imbalance, now disqualified
# --------------------------------------------------------------------------- #


class TestTapAfterQualifying:
    def test_tapped_after_two_approaches(self) -> None:
        df = make_w_df(
            # 2 approaches qualify
            bar_in_approach_band_buy(low=1905, high=1908),
            bar_retreated_buy(low=1915, high=1920),
            bar_above_zone_buy(low=1920, high=1925),
            bar_in_approach_band_buy(low=1903, high=1908),
            bar_retreated_buy(low=1913, high=1918),
            # Now tap.
            bar_in_zone_buy(low=1898, high=1905),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.approach_count == 2  # historical record preserved
        assert result.is_tapped is True
        assert result.is_imbalance is False  # tapped → disqualified
        # qualified_at is set: zone DID become Imbalance historically.
        assert result.qualified_at == df.index[16]
        assert result.tapped_at == df.index[17]


# --------------------------------------------------------------------------- #
# Boundary: approach exactly at threshold distance
# --------------------------------------------------------------------------- #


class TestBoundaryDistance:
    def test_approach_at_exact_outer_edge_inclusive(self) -> None:
        # bar.low == zone.top + approach_distance == 1907.5
        df = make_w_df(
            {"open": 1910, "high": 1912, "low": 1907.5, "close": 1908},
            bar_retreated_buy(low=1915, high=1920),
        )
        zone = make_w_validated_zone(df, top=1900)
        result = track_imbalance(zone, df)
        assert result.approach_count == 1  # boundary inclusive

    def test_approach_just_outside_outer_edge_excluded(self) -> None:
        # bar.low == 1907.51 — just outside band.
        df = make_w_df(
            {"open": 1910, "high": 1915, "low": 1907.51, "close": 1908},
            bar_retreated_buy(low=1915, high=1920),
        )
        zone = make_w_validated_zone(df, top=1900)
        result = track_imbalance(zone, df)
        assert result.approach_count == 0

    def test_retreat_at_exact_threshold_inclusive(self) -> None:
        # Retreat threshold for top=1900: 1900 + 7.5 + 5 = 1912.5.
        # bar.low == 1912.5 should complete the approach (inclusive).
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1910),
            {"open": 1913, "high": 1915, "low": 1912.5, "close": 1914},
        )
        zone = make_w_validated_zone(df, top=1900)
        result = track_imbalance(zone, df)
        assert result.approach_count == 1


# --------------------------------------------------------------------------- #
# Multiple approaches in same candle → count as 1
# --------------------------------------------------------------------------- #


class TestSameCandleSemantics:
    def test_single_bar_in_band_is_one_approach(self) -> None:
        # Bar enters band, then a separate bar retreats. One approach.
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1907),
            bar_retreated_buy(low=1916, high=1920),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.approach_count == 1

    def test_consecutive_bars_in_band_count_as_one_approach(self) -> None:
        # 3 bars in band, then retreat. Still 1 approach (oscillation
        # within an unbroken visit).
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1907),
            bar_in_approach_band_buy(low=1903, high=1906),
            bar_in_approach_band_buy(low=1904, high=1907),
            bar_retreated_buy(low=1916, high=1920),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.approach_count == 1
        # closest_price tracks the deepest within the visit.
        assert result.approach_events[0].closest_price == 1903.0


# --------------------------------------------------------------------------- #
# Gap-through-approach-into-zone → tap, not approach
# --------------------------------------------------------------------------- #


class TestGapThrough:
    def test_bar_high_above_band_low_in_zone_is_tap(self) -> None:
        # high=1925 (above retreat threshold), low=1898 (in zone).
        # Tap takes precedence over any band/retreat bookkeeping.
        df = make_w_df(
            {"open": 1920, "high": 1925, "low": 1898, "close": 1900},
        )
        zone = make_w_validated_zone(df, top=1900)
        result = track_imbalance(zone, df)
        assert result.is_tapped is True
        assert result.approach_count == 0
        assert result.tapped_at == df.index[12]


# --------------------------------------------------------------------------- #
# Skip when not Strong Point
# --------------------------------------------------------------------------- #


class TestNotStrongPoint:
    def test_non_strong_point_returns_no_imbalance(self) -> None:
        # Even with bars that would qualify, skip because is_strong_point=False.
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1908),
            bar_retreated_buy(low=1915, high=1920),
            bar_in_approach_band_buy(low=1903, high=1908),
            bar_retreated_buy(low=1913, high=1918),
        )
        zone = make_w_validated_zone(df, is_strong_point=False)
        result = track_imbalance(zone, df)
        assert result.is_imbalance is False
        assert result.approach_count == 0
        assert result.approach_events == []
        assert result.qualified_at is None


# --------------------------------------------------------------------------- #
# Mirror — SELL zone
# --------------------------------------------------------------------------- #


def bar_in_band_sell(high: float = 1895.0, low: float = 1890.0) -> dict:
    return {"open": low + 1, "high": high, "low": low, "close": low + 1}


def bar_retreated_sell(high: float = 1885.0, low: float = 1880.0) -> dict:
    return {"open": high - 1, "high": high, "low": low, "close": high - 1}


def bar_in_zone_sell(high: float = 1902.0, low: float = 1895.0) -> dict:
    return {"open": low + 1, "high": high, "low": low, "close": high - 1}


def bar_below_zone_sell(high: float = 1880.0, low: float = 1875.0) -> dict:
    return {"open": low + 1, "high": high, "low": low, "close": low + 1}


class TestSellZone:
    def test_clean_imbalance_for_sell(self) -> None:
        # SELL zone with bottom=1900, approach band [1892.5, 1900),
        # retreat threshold 1887.5.
        df = make_m_df(
            # approach 1
            bar_in_band_sell(high=1895, low=1893),
            bar_retreated_sell(high=1886, low=1880),
            # idle
            bar_below_zone_sell(high=1880, low=1875),
            # approach 2
            bar_in_band_sell(high=1897, low=1894),
            bar_retreated_sell(high=1885, low=1880),
        )
        zone = make_m_validated_zone(df, top=1905, bottom=1900)
        result = track_imbalance(zone, df)
        assert result.approach_count == 2
        assert result.is_imbalance is True
        assert result.is_tapped is False
        assert result.qualified_at == df.index[16]
        assert result.direction == "SELL"

    def test_sell_tap_takes_precedence(self) -> None:
        df = make_m_df(
            # bar high gaps above zone (tap from below)
            bar_in_zone_sell(high=1905, low=1895),
        )
        zone = make_m_validated_zone(df, top=1905, bottom=1900)
        result = track_imbalance(zone, df)
        assert result.is_tapped is True
        assert result.approach_count == 0

    def test_sell_boundary_at_outer_edge_inclusive(self) -> None:
        # bar.high == 1892.5 (= bottom - approach_distance) → in band.
        df = make_m_df(
            {"open": 1890, "high": 1892.5, "low": 1888, "close": 1889},
            bar_retreated_sell(high=1885, low=1880),
        )
        zone = make_m_validated_zone(df, top=1905, bottom=1900)
        result = track_imbalance(zone, df)
        assert result.approach_count == 1


# --------------------------------------------------------------------------- #
# Custom config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_higher_threshold_requires_more_approaches(self) -> None:
        # Two approaches happen; threshold raised to 3 → not qualified.
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1908),
            bar_retreated_buy(low=1915, high=1920),
            bar_in_approach_band_buy(low=1903, high=1908),
            bar_retreated_buy(low=1913, high=1918),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(
            zone, df, ImbalanceConfig(imbalance_approach_threshold=3)
        )
        assert result.approach_count == 2
        assert result.is_imbalance is False

    def test_smaller_approach_distance_misses_far_bars(self) -> None:
        # Bar at low=1905 is outside a 3-point approach band.
        df = make_w_df(
            bar_in_approach_band_buy(low=1905, high=1908),
            bar_retreated_buy(low=1915, high=1920),
        )
        zone = make_w_validated_zone(df)
        result = track_imbalance(
            zone, df, ImbalanceConfig(imbalance_approach_distance=3.0)
        )
        # 3-pt band: (1900, 1903]. low=1905 not in band.
        assert result.approach_count == 0


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_high_column_rejected(self) -> None:
        df = make_w_df(bar_in_approach_band_buy())
        zone = make_w_validated_zone(df)
        with pytest.raises(ValueError, match="'high'"):
            track_imbalance(zone, df.drop(columns=["high"]))

    def test_missing_low_column_rejected(self) -> None:
        df = make_w_df(bar_in_approach_band_buy())
        zone = make_w_validated_zone(df)
        with pytest.raises(ValueError, match="'low'"):
            track_imbalance(zone, df.drop(columns=["low"]))

    def test_formation_index_out_of_range_rejected(self) -> None:
        # Construct a zone with low2.index pointing past the df's end —
        # done manually here because the helper would raise IndexError
        # on df.index[999] before we could exercise the validation.
        df = make_w_df()
        ts = df.index[0]
        pattern = WPattern(
            low1=Swing(index=2, time=df.index[2], price=1900.0, kind="LOW"),
            low2=Swing(index=999, time=ts, price=1900.0, kind="LOW"),
            peak_index=6,
            peak_time=df.index[6],
            peak_price=1910.0,
            formed_at=ts,
            completed=True,
        )
        initial = Zone(
            direction="BUY",
            top=1900.0,
            bottom=1895.0,
            formed_at=ts,
            source_pattern=pattern,
        )
        refined = RefinedZone(
            direction="BUY",
            top=1900.0,
            bottom=1895.0,
            formed_at=ts,
            source_pattern=pattern,
            is_tradeable=True,
            rejection_reason=None,
            original_zone=initial,
        )
        bad_zone = ValidatedZone(
            direction="BUY",
            top=1900.0,
            bottom=1895.0,
            formed_at=ts,
            source_pattern=pattern,
            is_tradeable=True,
            rejection_reason=None,
            original_zone=initial,
            refined_zone=refined,
            is_strong_point=True,
            validation_failures=[],
            bos_event=None,
        )
        with pytest.raises(ValueError, match="out of df range"):
            track_imbalance(bad_zone, df)


# --------------------------------------------------------------------------- #
# Passthrough fields
# --------------------------------------------------------------------------- #


class TestPassthrough:
    def test_validated_zone_preserved_by_identity(self) -> None:
        df = make_w_df(bar_in_approach_band_buy())
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.validated_zone is zone

    def test_passthrough_fields_match_input(self) -> None:
        df = make_w_df(bar_in_approach_band_buy())
        zone = make_w_validated_zone(df)
        result = track_imbalance(zone, df)
        assert result.direction == zone.direction
        assert result.top == zone.top
        assert result.bottom == zone.bottom
        assert result.formed_at == zone.formed_at
        assert result.is_strong_point == zone.is_strong_point
        assert result.is_tradeable == zone.is_tradeable
        assert result.refined_zone is zone.refined_zone
