"""Tests for ``bot.strategy.zone_marking``.

Test data uses an ``make_ohlc`` helper that lets us specify ``low`` and
``high`` columns separately from ``close`` — necessary because zone
marking pulls the box bottom (top) from the wick column, not the close.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.pattern_detection import (
    MPattern,
    PatternConfig,
    WPattern,
    detect_latest_m,
    detect_latest_w,
)
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import (
    Zone,
    mark_zone,
    mark_zone_from_m,
    mark_zone_from_w,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ohlc(
    closes: list[float],
    lows: list[float] | None = None,
    highs: list[float] | None = None,
    opens: list[float] | None = None,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    """Build an OHLC DataFrame; missing columns default to ``closes``."""
    n = len(closes)
    if lows is None:
        lows = list(closes)
    if highs is None:
        highs = list(closes)
    if opens is None:
        opens = list(closes)
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


# Same close sequences as test_pattern_detection — they produce a
# textbook W and a textbook M with strength=2 and the default config.
W_CLOSES = [20, 18, 14, 16, 18, 20, 22, 20, 18, 14, 16, 18, 20]
M_CLOSES = [10, 12, 16, 14, 12, 10, 8, 10, 12, 16, 14, 12, 10]


# --------------------------------------------------------------------------- #
# Clean W → demand zone
# --------------------------------------------------------------------------- #


class TestWZone:
    def test_textbook_w_with_separated_wicks(self) -> None:
        # Closes give us the W; wicks at the swing-low bars dip below the
        # closes (12 and 13 vs close=14 at bars 2 and 9 respectively).
        # Box top should be the higher of the two close-based lows (14);
        # box bottom should be the deepest wick in [bar2 .. bar9] = 12.
        lows = [20, 18, 12, 16, 18, 20, 22, 20, 18, 13, 16, 18, 20]
        df = make_ohlc(W_CLOSES, lows=lows)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        assert pattern is not None

        zone = mark_zone_from_w(pattern, df)
        assert zone.direction == "BUY"
        assert zone.top == 14.0
        assert zone.bottom == 12.0
        assert zone.formed_at == pattern.formed_at
        assert zone.source_pattern is pattern

    def test_box_height_is_positive_for_wicked_w(self) -> None:
        lows = [20, 18, 12, 16, 18, 20, 22, 20, 18, 13, 16, 18, 20]
        df = make_ohlc(W_CLOSES, lows=lows)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_w(pattern, df)
        assert zone.top - zone.bottom == 2.0

    def test_w_with_ohlc_equal_to_closes_yields_degenerate_zone(self) -> None:
        # When wicks == closes (synthetic data), bottom == max(low1, low2)
        # because the lowest "low" in the range IS one of the close-based
        # lows. Result: top == bottom (zero height). That's expected and
        # OK — refinement / size filter handle this downstream.
        df = make_ohlc(W_CLOSES)  # lows default to closes
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_w(pattern, df)
        assert zone.top == 14.0
        assert zone.bottom == 14.0
        assert zone.top == zone.bottom  # degenerate but not inverted


# --------------------------------------------------------------------------- #
# Clean M → supply zone
# --------------------------------------------------------------------------- #


class TestMZone:
    def test_textbook_m_with_separated_wicks(self) -> None:
        # Wicks at the swing-high bars stick above the closes
        # (18 and 17 vs close=16 at bars 2 and 9).
        highs = [10, 12, 18, 14, 12, 10, 8, 10, 12, 17, 14, 12, 10]
        df = make_ohlc(M_CLOSES, highs=highs)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        assert pattern is not None

        zone = mark_zone_from_m(pattern, df)
        assert zone.direction == "SELL"
        assert zone.bottom == 16.0  # min(16, 16) = 16
        assert zone.top == 18.0  # max wick in range
        assert zone.formed_at == pattern.formed_at
        assert zone.source_pattern is pattern

    def test_box_height_is_positive_for_wicked_m(self) -> None:
        highs = [10, 12, 18, 14, 12, 10, 8, 10, 12, 17, 14, 12, 10]
        df = make_ohlc(M_CLOSES, highs=highs)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_m(pattern, df)
        assert zone.top - zone.bottom == 2.0


# --------------------------------------------------------------------------- #
# Direction semantics
# --------------------------------------------------------------------------- #


class TestDirection:
    def test_w_produces_buy_zone(self) -> None:
        lows = [20, 18, 12, 16, 18, 20, 22, 20, 18, 13, 16, 18, 20]
        df = make_ohlc(W_CLOSES, lows=lows)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone(pattern, df)  # dispatch path
        assert zone.direction == "BUY"

    def test_m_produces_sell_zone(self) -> None:
        highs = [10, 12, 18, 14, 12, 10, 8, 10, 12, 17, 14, 12, 10]
        df = make_ohlc(M_CLOSES, highs=highs)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        zone = mark_zone(pattern, df)
        assert zone.direction == "SELL"


# --------------------------------------------------------------------------- #
# Edge cases requested in the spec
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    def test_very_tight_lows_produces_small_but_valid_zone(self) -> None:
        # Lows differ by 0.005 (within 0.1% tolerance: diff_pct ≈ 0.0357%).
        # The bar at low1 has a slightly deeper wick (13.99) so the box
        # has a tiny positive height.
        closes = [20, 18, 14.000, 16, 18, 20, 22, 20, 18, 14.005, 16, 18, 20]
        lows = [20, 18, 13.990, 16, 18, 20, 22, 20, 18, 14.005, 16, 18, 20]
        df = make_ohlc(closes, lows=lows)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        assert pattern is not None
        zone = mark_zone_from_w(pattern, df)
        # top = max(low1.close, low2.close) = max(14.000, 14.005) = 14.005
        assert zone.top == pytest.approx(14.005)
        # bottom = min(low column in [2..9]) = 13.990
        assert zone.bottom == pytest.approx(13.990)
        assert zone.top - zone.bottom == pytest.approx(0.015)

    def test_deep_wick_makes_zone_wide(self) -> None:
        # Big wick at low1 (close=14, low=8 — six points of wick).
        # Demonstrates the "wide initial zone" behaviour the spec
        # accepts: refinement will narrow it.
        lows = [20, 18, 8, 16, 18, 20, 22, 20, 18, 14, 16, 18, 20]
        df = make_ohlc(W_CLOSES, lows=lows)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_w(pattern, df)
        assert zone.top == 14.0
        assert zone.bottom == 8.0
        assert zone.top - zone.bottom == 6.0  # the full wick

    def test_deep_wick_above_for_m_makes_zone_wide(self) -> None:
        # Symmetric case for M.
        highs = [10, 12, 24, 14, 12, 10, 8, 10, 12, 16, 14, 12, 10]
        df = make_ohlc(M_CLOSES, highs=highs)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_m(pattern, df)
        assert zone.bottom == 16.0
        assert zone.top == 24.0
        assert zone.top - zone.bottom == 8.0


# --------------------------------------------------------------------------- #
# formed_at and source_pattern
# --------------------------------------------------------------------------- #


class TestZoneMetadata:
    def test_formed_at_matches_low2_time_for_w(self) -> None:
        df = make_ohlc(W_CLOSES)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_w(pattern, df)
        assert zone.formed_at == pattern.low2.time

    def test_formed_at_matches_high2_time_for_m(self) -> None:
        df = make_ohlc(M_CLOSES)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_m(pattern, df)
        assert zone.formed_at == pattern.high2.time

    def test_source_pattern_is_the_input_pattern(self) -> None:
        df = make_ohlc(W_CLOSES)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone_from_w(pattern, df)
        # Identity, not equality — preserve the actual reference.
        assert zone.source_pattern is pattern


# --------------------------------------------------------------------------- #
# Dispatch + error handling
# --------------------------------------------------------------------------- #


class TestDispatchAndErrors:
    def test_mark_zone_dispatches_w(self) -> None:
        df = make_ohlc(W_CLOSES)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone(pattern, df)
        assert zone.direction == "BUY"

    def test_mark_zone_dispatches_m(self) -> None:
        df = make_ohlc(M_CLOSES)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        zone = mark_zone(pattern, df)
        assert zone.direction == "SELL"

    def test_mark_zone_rejects_unknown_pattern_type(self) -> None:
        with pytest.raises(TypeError, match="unsupported pattern type"):
            mark_zone("not a pattern", make_ohlc(W_CLOSES))  # type: ignore[arg-type]

    def test_w_marker_requires_low_column(self) -> None:
        df = make_ohlc(W_CLOSES)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        df_no_low = df.drop(columns=["low"])
        with pytest.raises(ValueError, match="'low' column"):
            mark_zone_from_w(pattern, df_no_low)

    def test_m_marker_requires_high_column(self) -> None:
        df = make_ohlc(M_CLOSES)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        df_no_high = df.drop(columns=["high"])
        with pytest.raises(ValueError, match="'high' column"):
            mark_zone_from_m(pattern, df_no_high)

    def test_indices_out_of_range_rejected(self) -> None:
        df = make_ohlc(W_CLOSES)
        # Build a fake pattern whose low2 sits past the end of the df.
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        bad_low1 = Swing(index=2, time=ts, price=14.0, kind="LOW")
        bad_low2 = Swing(index=999, time=ts, price=14.0, kind="LOW")
        bad_pattern = WPattern(
            low1=bad_low1,
            low2=bad_low2,
            peak_index=6,
            peak_time=ts,
            peak_price=22.0,
            formed_at=ts,
            completed=True,
        )
        with pytest.raises(ValueError, match="out of df range"):
            mark_zone_from_w(bad_pattern, df)

    def test_indices_in_wrong_order_rejected(self) -> None:
        df = make_ohlc(W_CLOSES)
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        # low1 chronologically AFTER low2 — caller error.
        bad_low1 = Swing(index=9, time=ts, price=14.0, kind="LOW")
        bad_low2 = Swing(index=2, time=ts, price=14.0, kind="LOW")
        bad_pattern = WPattern(
            low1=bad_low1,
            low2=bad_low2,
            peak_index=6,
            peak_time=ts,
            peak_price=22.0,
            formed_at=ts,
            completed=True,
        )
        with pytest.raises(ValueError, match="wrong order"):
            mark_zone_from_w(bad_pattern, df)


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


class TestInvariants:
    def test_top_never_below_bottom_for_w(self) -> None:
        # Try several wick configurations.
        for lows in [
            [20, 18, 14, 16, 18, 20, 22, 20, 18, 14, 16, 18, 20],   # equal
            [20, 18, 12, 16, 18, 20, 22, 20, 18, 13, 16, 18, 20],   # wicked
            [20, 18, 8, 16, 18, 20, 22, 20, 18, 9, 16, 18, 20],     # deeply wicked
        ]:
            df = make_ohlc(W_CLOSES, lows=lows)
            pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
            zone = mark_zone_from_w(pattern, df)
            assert zone.top >= zone.bottom, (
                f"invariant violated: top={zone.top} bottom={zone.bottom} "
                f"lows={lows}"
            )

    def test_top_never_below_bottom_for_m(self) -> None:
        for highs in [
            [10, 12, 16, 14, 12, 10, 8, 10, 12, 16, 14, 12, 10],
            [10, 12, 18, 14, 12, 10, 8, 10, 12, 17, 14, 12, 10],
            [10, 12, 22, 14, 12, 10, 8, 10, 12, 21, 14, 12, 10],
        ]:
            df = make_ohlc(M_CLOSES, highs=highs)
            pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
            zone = mark_zone_from_m(pattern, df)
            assert zone.top >= zone.bottom


# --------------------------------------------------------------------------- #
# End-to-end integration
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_w_pattern_to_zone_pipeline(self) -> None:
        # Realistic flow: OHLC → pattern detection → zone marking.
        lows = [20, 18, 13.5, 16, 18, 20, 22, 20, 18, 13.7, 16, 18, 20]
        df = make_ohlc(W_CLOSES, lows=lows)
        pattern = detect_latest_w(df, PatternConfig(swing_strength=2))
        zone = mark_zone(pattern, df)

        assert isinstance(zone, Zone)
        assert zone.direction == "BUY"
        assert zone.top == 14.0  # higher of the two equal close-based lows
        assert zone.bottom == 13.5  # deepest wick
        assert zone.top > zone.bottom
        assert zone.formed_at == df.index[pattern.low2.index]

    def test_m_pattern_to_zone_pipeline(self) -> None:
        highs = [10, 12, 16.5, 14, 12, 10, 8, 10, 12, 16.3, 14, 12, 10]
        df = make_ohlc(M_CLOSES, highs=highs)
        pattern = detect_latest_m(df, PatternConfig(swing_strength=2))
        zone = mark_zone(pattern, df)

        assert isinstance(zone, Zone)
        assert zone.direction == "SELL"
        assert zone.bottom == 16.0
        assert zone.top == 16.5
        assert zone.top > zone.bottom
        assert zone.formed_at == df.index[pattern.high2.index]
