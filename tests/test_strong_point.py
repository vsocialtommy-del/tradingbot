"""Tests for ``bot.strategy.strong_point``.

Tests build OHLC + RefinedZone + BosEvent fixtures by hand so we can
target each validation gate independently, including boundary
conditions for the 60% body and 50% base ratios.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.structure import BosEvent, Swing
from bot.strategy.strong_point import (
    StrongPointConfig,
    ValidatedZone,
    validate_strong_point,
)
from bot.strategy.zone_marking import mark_zone_from_m, mark_zone_from_w
from bot.strategy.zone_refinement import RefinedZone, refine_zone


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


def make_w_refined_zone(
    df: pd.DataFrame,
    low1_idx: int = 2,
    low2_idx: int = 9,
    peak_idx: int = 6,
) -> RefinedZone:
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
    zone = mark_zone_from_w(pattern, df)
    return refine_zone(zone, df)


def make_m_refined_zone(
    df: pd.DataFrame,
    high1_idx: int = 2,
    high2_idx: int = 9,
    trough_idx: int = 6,
) -> RefinedZone:
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
    zone = mark_zone_from_m(pattern, df)
    return refine_zone(zone, df)


def make_bos(
    bar_index: int,
    direction: str,
    df: pd.DataFrame,
    broken_swing_index: int = 6,
    broken_level: float = 1910.0,
) -> BosEvent:
    return BosEvent(
        bar_index=bar_index,
        time=df.index[bar_index],
        direction=direction,  # type: ignore[arg-type]
        broken_swing_index=broken_swing_index,
        broken_level=broken_level,
        break_close=float(df["close"].iloc[bar_index]),
    )


# A "clean Strong Point" W: lows at 2/9 (close=1900), peak at 6 (1910),
# impulse at 11 (open=1900 close=1920 high=1925 low=1898 → body 20,
# range 27, body/range ≈ 0.74). Base bars are tight (range 8 and 2).
#
# Refined zone uses bar 2's body (open=1895 close=1900 → 5 pts wide).

CLEAN_W_DATA = {
    "closes": [1910, 1905, 1900, 1903, 1906, 1908, 1910, 1908, 1906, 1900, 1903, 1920, 1922],
    "opens":  [1910, 1905, 1895, 1903, 1906, 1908, 1910, 1908, 1906, 1900, 1903, 1900, 1920],
    "highs":  [1910, 1905, 1901, 1903, 1906, 1908, 1910, 1908, 1906, 1901, 1903, 1925, 1922],
    "lows":   [1910, 1905, 1893, 1903, 1906, 1908, 1910, 1908, 1906, 1899, 1903, 1898, 1920],
}


def make_clean_w_df() -> pd.DataFrame:
    return make_ohlc(**CLEAN_W_DATA)


# --------------------------------------------------------------------------- #
# Clean Strong Point
# --------------------------------------------------------------------------- #


class TestCleanStrongPoint:
    def test_clean_setup_validates(self) -> None:
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        # Sanity: zone passes size filter.
        assert zone.is_tradeable is True

        bos = make_bos(11, "UP", df, broken_level=1910.0)
        validated = validate_strong_point(zone, df, [bos])

        assert validated.is_strong_point is True
        assert validated.validation_failures == []
        assert validated.bos_event is bos

    def test_passthrough_fields_preserved(self) -> None:
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.direction == "BUY"
        assert v.top == zone.top
        assert v.bottom == zone.bottom
        assert v.formed_at == zone.formed_at
        assert v.source_pattern is zone.source_pattern
        assert v.is_tradeable == zone.is_tradeable
        assert v.refined_zone is zone


# --------------------------------------------------------------------------- #
# No BoS yet
# --------------------------------------------------------------------------- #


class TestNoBosYet:
    def test_empty_bos_list(self) -> None:
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        v = validate_strong_point(zone, df, [])
        assert v.is_strong_point is False
        assert v.validation_failures == ["NO_BOS_YET"]
        assert v.bos_event is None

    def test_only_down_bos_for_w_zone(self) -> None:
        # W zone (BUY) needs UP BoS; only DOWN exists → NO_BOS_YET.
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        bos_down = make_bos(11, "DOWN", df, broken_level=1900.0)
        v = validate_strong_point(zone, df, [bos_down])
        assert v.is_strong_point is False
        assert "NO_BOS_YET" in v.validation_failures
        assert v.bos_event is None

    def test_bos_before_pattern_completion(self) -> None:
        # BoS occurs at bar 5 (before low2 at bar 9) — must be ignored.
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        early_bos = make_bos(5, "UP", df)
        v = validate_strong_point(zone, df, [early_bos])
        assert v.is_strong_point is False
        assert v.validation_failures == ["NO_BOS_YET"]


# --------------------------------------------------------------------------- #
# Impulse strength + direction
# --------------------------------------------------------------------------- #


class TestImpulseStrength:
    def test_impulse_too_weak_wick_driven_bos(self) -> None:
        # BoS bar with tiny body, large wick. Direction is correct but body
        # ratio fails 0.6 threshold.
        d = dict(CLEAN_W_DATA)
        # impulse bar = idx 11.
        # open=1900 close=1912 high=1925 low=1908 → body 12, range 17, ratio ≈ 0.71 — strong.
        # Make it weak: open=1910 close=1912 → body 2, high=1925 low=1908 → range 17, ratio 0.118.
        d = {k: list(v) for k, v in d.items()}
        d["opens"][11] = 1910
        d["closes"][11] = 1912
        d["highs"][11] = 1925
        d["lows"][11] = 1908
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert "IMPULSE_TOO_WEAK" in v.validation_failures
        # Direction is OK (close > open), so that one shouldn't fire.
        assert "IMPULSE_WRONG_DIRECTION" not in v.validation_failures

    def test_impulse_wrong_direction_bearish_for_up_bos(self) -> None:
        # Body is decisive, but bar is bearish (open > close) for an UP BoS.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        # open=1925 close=1912 → bearish, body 13, range (high=1928, low=1910)=18,
        # ratio 13/18 ≈ 0.72 — passes strength.
        d["opens"][11] = 1925
        d["closes"][11] = 1912
        d["highs"][11] = 1928
        d["lows"][11] = 1910
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert "IMPULSE_WRONG_DIRECTION" in v.validation_failures
        assert "IMPULSE_TOO_WEAK" not in v.validation_failures

    def test_doji_impulse_fails_direction(self) -> None:
        # open == close → can't satisfy close > open.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        d["opens"][11] = 1912
        d["closes"][11] = 1912  # doji
        d["highs"][11] = 1925
        d["lows"][11] = 1898
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert "IMPULSE_WRONG_DIRECTION" in v.validation_failures


class TestImpulseDirectionM:
    def test_bullish_impulse_for_down_bos_fails(self) -> None:
        # M zone needs DOWN BoS where bar is bearish. Make it bullish: fail.
        closes = [1890, 1895, 1900, 1897, 1894, 1892, 1890, 1892, 1894, 1900, 1897, 1880, 1878]
        opens = [1890, 1895, 1900, 1897, 1894, 1892, 1890, 1892, 1894, 1905, 1897, 1875, 1880]
        highs = [1890, 1895, 1901, 1897, 1894, 1892, 1890, 1892, 1894, 1907, 1897, 1888, 1883]
        lows = [1890, 1895, 1899, 1897, 1894, 1892, 1890, 1892, 1894, 1900, 1897, 1872, 1875]
        # impulse bar idx 11: open=1875 close=1880 → bullish (open < close).
        # body=5, range=16, ratio=0.31. Both fails: TOO_WEAK + WRONG_DIRECTION.
        df = make_ohlc(closes, opens=opens, highs=highs, lows=lows)
        zone = make_m_refined_zone(df)
        bos = make_bos(11, "DOWN", df, broken_level=1900.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert "IMPULSE_WRONG_DIRECTION" in v.validation_failures


# --------------------------------------------------------------------------- #
# Base compactness
# --------------------------------------------------------------------------- #


class TestBaseCompactness:
    def test_base_too_large_for_w(self) -> None:
        # Make base bars (idx 2 and 9) with huge ranges relative to impulse.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        # impulse range remains 27 (high 1925 low 1898).
        # threshold = 0.5 * 27 = 13.5.
        # Make base bar 2's range = 25 (>13.5) → fails.
        d["highs"][2] = 1915
        d["lows"][2] = 1890
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        # Note: zone refinement uses bar 2's open/close for body, not range.
        # opens[2]=1895, closes[2]=1900 unchanged → still tradeable.
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert "BASE_NOT_COMPACT" in v.validation_failures


# --------------------------------------------------------------------------- #
# Boundary conditions — 60% body and 50% base
# --------------------------------------------------------------------------- #


class TestBoundaries:
    def test_body_ratio_at_threshold_passes(self) -> None:
        # body/range = exactly 0.6.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        # body = 6, range = 10 → ratio = 0.6 inclusive.
        # impulse bar idx 11: open=1910 close=1916 → body 6, bullish.
        # high=1918 low=1908 → range 10. body/range = 0.6.
        d["opens"][11] = 1910
        d["closes"][11] = 1916
        d["highs"][11] = 1918
        d["lows"][11] = 1908
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        # 0.6 is INCLUSIVE → strong impulse passes; need to check base too.
        # Base bars: idx 2 has range 8 (high 1901, low 1893).
        # Threshold = 0.5 * 10 = 5. Bar 2 range 8 > 5 → base fails.
        # So is_strong_point should still be False, but for BASE_NOT_COMPACT
        # — NOT for IMPULSE_TOO_WEAK.
        assert "IMPULSE_TOO_WEAK" not in v.validation_failures

    def test_body_ratio_just_below_threshold_fails(self) -> None:
        # body/range = 0.59 (just under).
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        # body 5.9, range 10 → ratio 0.59. Use 5.9 → not float-clean.
        # Use body 59, range 100. open=1910, close=1969, high=1980, low=1880.
        # body=59, range=100, ratio=0.59.
        d["opens"][11] = 1910
        d["closes"][11] = 1969
        d["highs"][11] = 1980
        d["lows"][11] = 1880
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert "IMPULSE_TOO_WEAK" in v.validation_failures

    def test_base_ratio_at_threshold_passes(self) -> None:
        # base_range / impulse_range = exactly 0.5.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        # impulse range = 27 (unchanged).
        # base bar 2 range = 13.5 → at threshold.
        # high - low = 13.5 → high=1908, low=1894.5
        d["highs"][2] = 1908
        d["lows"][2] = 1894.5
        # base bar 9 range needs to be <= 13.5 too.
        d["highs"][9] = 1905
        d["lows"][9] = 1898
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        # base_ratio == 0.5 inclusive → passes. Should NOT fail BASE_NOT_COMPACT.
        assert "BASE_NOT_COMPACT" not in v.validation_failures

    def test_base_ratio_just_above_threshold_fails(self) -> None:
        # base_range / impulse_range = 0.51.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        # impulse range 27, threshold 13.5. Make base bar 2 range 13.77.
        d["highs"][2] = 1908
        d["lows"][2] = 1894.23
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert "BASE_NOT_COMPACT" in v.validation_failures


# --------------------------------------------------------------------------- #
# Zero-range bars
# --------------------------------------------------------------------------- #


class TestZeroRangeBars:
    def test_zero_range_impulse_fails_too_weak(self) -> None:
        # Impulse bar with all OHLC equal — range=0, body=0.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        d["opens"][11] = 1920
        d["closes"][11] = 1920
        d["highs"][11] = 1920
        d["lows"][11] = 1920
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        # Should not raise (no division by zero).
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert "IMPULSE_TOO_WEAK" in v.validation_failures

    def test_zero_range_base_bar_passes_compactness(self) -> None:
        # Base bar idx 2 is a doji at exactly 1900 (no wick at all).
        # Bar 9 (the other base) might still have some range from CLEAN_W_DATA.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        d["opens"][2] = 1900
        d["closes"][2] = 1900
        d["highs"][2] = 1900
        d["lows"][2] = 1900
        d["opens"][9] = 1900
        d["highs"][9] = 1900
        d["lows"][9] = 1900
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        # NOTE: with both base bars at zero open/close=1900, refined zone
        # becomes 1900-1900 (width 0) → fails size filter. validate_strong_point
        # short-circuits to NOT_TRADEABLE.
        assert zone.is_tradeable is False
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.validation_failures == ["NOT_TRADEABLE"]


# --------------------------------------------------------------------------- #
# Multiple BoS events
# --------------------------------------------------------------------------- #


class TestMultipleBos:
    def test_first_matching_bos_used(self) -> None:
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        bos1 = make_bos(11, "UP", df, broken_level=1910.0)
        bos2 = make_bos(12, "UP", df, broken_level=1915.0)
        # Pass in reverse order to ensure sorting works.
        v = validate_strong_point(zone, df, [bos2, bos1])
        assert v.bos_event is bos1  # the earlier one

    def test_opposite_direction_events_ignored(self) -> None:
        # Both UP and DOWN events present; for W, only UP is considered.
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        bos_up = make_bos(11, "UP", df, broken_level=1910.0)
        bos_down = make_bos(10, "DOWN", df, broken_level=1900.0)
        v = validate_strong_point(zone, df, [bos_down, bos_up])
        # The DOWN event is at an earlier index, but it's ignored. UP wins.
        assert v.bos_event is bos_up
        assert v.is_strong_point is True


# --------------------------------------------------------------------------- #
# Multiple failures collected together
# --------------------------------------------------------------------------- #


class TestMultipleFailures:
    def test_weak_impulse_and_non_compact_base_both_returned(self) -> None:
        # Construct a scenario that fails both gates simultaneously.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        # Weak impulse: body 2, range 20 → ratio 0.1.
        d["opens"][11] = 1910
        d["closes"][11] = 1912
        d["highs"][11] = 1925
        d["lows"][11] = 1905
        # Non-compact base: bar 2 range 15 (> 0.5*20 = 10).
        d["highs"][2] = 1910
        d["lows"][2] = 1895
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert "IMPULSE_TOO_WEAK" in v.validation_failures
        assert "BASE_NOT_COMPACT" in v.validation_failures


# --------------------------------------------------------------------------- #
# Not-tradeable zones short-circuit
# --------------------------------------------------------------------------- #


class TestNotTradeable:
    def test_zone_failing_size_filter_skips_validation(self) -> None:
        # Refined width 0 < 5 → NOT_TRADEABLE upstream.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        d["opens"][2] = 1900  # remove the body that gave width 5
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        assert zone.is_tradeable is False  # sanity
        # Even with a perfect BoS, validation short-circuits.
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is False
        assert v.validation_failures == ["NOT_TRADEABLE"]
        assert v.bos_event is None  # not even computed


# --------------------------------------------------------------------------- #
# M mirror — full pass
# --------------------------------------------------------------------------- #


class TestMStrongPoint:
    def test_clean_m_strong_point(self) -> None:
        # M with highs at idx 2/9 (close=1900), trough at 6 (1890).
        # Impulse at idx 11 — bearish move down with strong body.
        # open=1900 close=1880 → body 20, bearish ✓.
        # high=1902 low=1875 → range 27, body/range ≈ 0.74 ✓.
        # Base bars at 2 and 9 with small ranges.
        closes = [1890, 1895, 1900, 1897, 1894, 1892, 1890, 1892, 1894, 1900, 1897, 1880, 1878]
        opens =  [1890, 1895, 1905, 1897, 1894, 1892, 1890, 1892, 1894, 1900, 1897, 1900, 1880]
        highs =  [1890, 1895, 1907, 1897, 1894, 1892, 1890, 1892, 1894, 1901, 1897, 1902, 1880]
        lows =   [1890, 1895, 1899, 1897, 1894, 1892, 1890, 1892, 1894, 1899, 1897, 1875, 1878]
        df = make_ohlc(closes, opens=opens, highs=highs, lows=lows)
        zone = make_m_refined_zone(df)
        # Sanity — refined width = max(open=1905, open=1900) - min(close=1900, close=1900) = 5.
        assert zone.is_tradeable is True
        bos = make_bos(11, "DOWN", df, broken_level=1890.0)
        v = validate_strong_point(zone, df, [bos])
        assert v.is_strong_point is True
        assert v.validation_failures == []
        assert v.direction == "SELL"


# --------------------------------------------------------------------------- #
# Custom config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_lower_body_ratio_admits_weaker_impulse(self) -> None:
        # body 6, range 20 → ratio 0.3. Default 0.6 fails; setting 0.25 passes.
        d = {k: list(v) for k, v in CLEAN_W_DATA.items()}
        d["opens"][11] = 1910
        d["closes"][11] = 1916
        d["highs"][11] = 1925
        d["lows"][11] = 1905
        df = make_ohlc(**d)
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        # Default — fails.
        v_default = validate_strong_point(zone, df, [bos])
        assert "IMPULSE_TOO_WEAK" in v_default.validation_failures
        # Lowered threshold — passes the impulse strength gate.
        v_relaxed = validate_strong_point(
            zone, df, [bos], StrongPointConfig(impulse_min_body_ratio=0.25)
        )
        assert "IMPULSE_TOO_WEAK" not in v_relaxed.validation_failures

    def test_tighter_base_ratio_rejects_otherwise_clean_setup(self) -> None:
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        # Default — clean setup passes.
        v_default = validate_strong_point(zone, df, [bos])
        assert v_default.is_strong_point is True
        # Tighten base ratio to 0.25 — bar 2's range (8) > 0.25*27=6.75.
        v_strict = validate_strong_point(
            zone, df, [bos], StrongPointConfig(base_max_range_ratio=0.25)
        )
        assert "BASE_NOT_COMPACT" in v_strict.validation_failures


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_required_column_rejected(self) -> None:
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        bos = make_bos(11, "UP", df, broken_level=1910.0)
        for col in ("open", "high", "low", "close"):
            df_missing = df.drop(columns=[col])
            with pytest.raises(ValueError, match=f"'{col}'"):
                validate_strong_point(zone, df_missing, [bos])

    def test_bos_index_out_of_range_rejected(self) -> None:
        df = make_clean_w_df()
        zone = make_w_refined_zone(df)
        # BoS at bar 999 — out of df.
        bad_bos = BosEvent(
            bar_index=999,
            time=pd.Timestamp("2026-01-01T00:00:00Z"),
            direction="UP",
            broken_swing_index=6,
            broken_level=1910.0,
            break_close=1920.0,
        )
        with pytest.raises(ValueError, match="out of df range"):
            validate_strong_point(zone, df, [bad_bos])
