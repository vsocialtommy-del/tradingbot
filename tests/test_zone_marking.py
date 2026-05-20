"""Tests for ``bot.strategy.zone_marking``.

Constructs Pattern objects directly (bypassing detection) so the
geometry math can be tested in isolation from the detection cascade.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.pattern_detection import (
    Base,
    Impulse,
    Pattern,
    PatternType,
)
from bot.strategy.zone_marking import Zone, mark_zone


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ohlc(
    opens: list[float],
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    n = len(opens)
    closes = list(closes) if closes is not None else list(opens)
    highs = list(highs) if highs is not None else [max(o, c) + 0.2 for o, c in zip(opens, closes)]
    lows = list(lows) if lows is not None else [min(o, c) - 0.2 for o, c in zip(opens, closes)]
    times = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [100] * n},
        index=times,
    )


def make_impulse(
    direction: str, start_index: int, end_index: int,
    df: pd.DataFrame, range_size: float = 5.0, largest_body: float = 5.0,
) -> Impulse:
    return Impulse(
        direction=direction,  # type: ignore[arg-type]
        start_index=start_index, end_index=end_index,
        start_time=df.index[start_index],
        end_time=df.index[end_index],
        range_size=range_size,
        largest_body=largest_body,
        candle_count=end_index - start_index + 1,
    )


def make_base(start_index: int, end_index: int, df: pd.DataFrame) -> Base:
    """Build a Base matching production semantics: top/bottom are wick-inclusive."""
    o = df["open"].iloc[start_index : end_index + 1].to_numpy()
    h = df["high"].iloc[start_index : end_index + 1].to_numpy()
    lo = df["low"].iloc[start_index : end_index + 1].to_numpy()
    c = df["close"].iloc[start_index : end_index + 1].to_numpy()
    top = float(h.max())
    bottom = float(lo.min())
    bodies = [abs(ci - oi) for oi, ci in zip(o, c)]
    return Base(
        start_index=start_index, end_index=end_index,
        candle_count=end_index - start_index + 1,
        top=top, bottom=bottom,
        range_size=top - bottom,
        largest_body=float(max(bodies)),
    )


def make_pattern(
    pattern_type: PatternType,
    df: pd.DataFrame,
    *,
    imp_before_indices: tuple[int, int],
    base_indices: tuple[int, int],
    imp_after_indices: tuple[int, int],
) -> Pattern:
    direction_map = {
        PatternType.RBR: ("RALLY", "RALLY", "BUY"),
        PatternType.DBD: ("DROP", "DROP", "SELL"),
        PatternType.DBR: ("DROP", "RALLY", "BUY"),
        PatternType.RBD: ("RALLY", "DROP", "SELL"),
    }
    d_before, d_after, zone_dir = direction_map[pattern_type]
    return Pattern(
        pattern_type=pattern_type,
        impulse_before=make_impulse(d_before, *imp_before_indices, df),
        base=make_base(*base_indices, df),
        impulse_after=make_impulse(d_after, *imp_after_indices, df),
        direction=zone_dir,  # type: ignore[arg-type]
        formed_at=df.index[imp_after_indices[1]],
    )


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


class TestZoneMarkingGeometry:
    def test_single_bullish_base_candle(self) -> None:
        opens = [100.0, 100.0, 105.0]
        closes = [100.0, 100.5, 105.0]
        # Explicit highs/lows so the test isn't sensitive to the
        # ``make_ohlc`` defaults (which auto-add a ±0.2 wick).
        highs = [100.0, 100.8, 105.0]
        lows = [100.0, 99.7, 105.0]
        df = make_ohlc(opens, closes=closes, highs=highs, lows=lows)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        # Wick-inclusive: top = high (100.8), bottom = low (99.7).
        assert zone.top == 100.8
        assert zone.bottom == 99.7
        assert zone.direction == "BUY"

    def test_single_bearish_base_candle(self) -> None:
        opens = [100.0, 100.5, 95.0]
        closes = [100.0, 100.0, 95.0]
        highs = [100.0, 100.9, 95.0]
        lows = [100.0, 99.5, 95.0]
        df = make_ohlc(opens, closes=closes, highs=highs, lows=lows)
        pattern = make_pattern(
            PatternType.DBD, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        # Wick-inclusive: top = high (100.9), bottom = low (99.5).
        assert zone.top == 100.9
        assert zone.bottom == 99.5
        assert zone.direction == "SELL"

    def test_multi_base_mixed_bullish_bearish(self) -> None:
        opens = [
            100.0,                # imp_before
            100.5, 100.0, 100.7,   # base
            105.0,                # imp_after
        ]
        closes = [
            100.0,
            100.0, 100.7, 100.3,
            105.0,
        ]
        # Explicit highs/lows on base bars so wick extents are predictable.
        highs = [100.0, 100.9, 100.8, 100.7, 105.0]
        lows  = [100.0,  99.8,  99.9,  99.6, 105.0]
        df = make_ohlc(opens, closes=closes, highs=highs, lows=lows)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 3),
            imp_after_indices=(4, 4),
        )
        zone = mark_zone(pattern, df)
        # Wick-inclusive: top = max(highs[1:4]) = 100.9
        #                 bottom = min(lows[1:4]) = 99.6
        assert zone.top == 100.9
        assert zone.bottom == 99.6

    def test_doji_base_with_flat_wicks_produces_zero_height(self) -> None:
        # If high == low == open == close on every base bar, the zone
        # has zero height. The size filter rejects these downstream.
        opens = [100.0, 100.0, 100.0, 100.0, 105.0]
        closes = [100.0, 100.0, 100.0, 100.0, 105.0]
        highs = [100.0, 100.0, 100.0, 100.0, 105.0]
        lows = [100.0, 100.0, 100.0, 100.0, 105.0]
        df = make_ohlc(opens, closes=closes, highs=highs, lows=lows)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 3),
            imp_after_indices=(4, 4),
        )
        zone = mark_zone(pattern, df)
        assert zone.top == zone.bottom == 100.0

    def test_wicks_included(self) -> None:
        # A long-wick rejection on the base bar should widen the zone
        # to encompass the full rejection range — that's the whole
        # point of wick-inclusive marking.
        opens = [100.0, 100.0, 105.0]
        closes = [100.0, 100.5, 105.0]
        highs = [100.0, 110.0, 105.0]   # tall upper wick on base bar
        lows = [100.0, 90.0, 105.0]      # tall lower wick
        df = make_ohlc(opens, closes=closes, highs=highs, lows=lows)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        # Wicks INCLUDED: zone spans the rejection range 90-110.
        assert zone.top == 110.0
        assert zone.bottom == 90.0


# --------------------------------------------------------------------------- #
# Direction propagation across all 4 pattern types
# --------------------------------------------------------------------------- #


class TestDirectionPropagation:
    def test_rbr_is_buy(self) -> None:
        df = make_ohlc([100.0, 100.0, 105.0])
        p = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        assert mark_zone(p, df).direction == "BUY"

    def test_dbd_is_sell(self) -> None:
        df = make_ohlc([100.0, 100.0, 95.0])
        p = make_pattern(
            PatternType.DBD, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        assert mark_zone(p, df).direction == "SELL"

    def test_dbr_is_buy(self) -> None:
        df = make_ohlc([100.0, 95.0, 100.0])
        p = make_pattern(
            PatternType.DBR, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        assert mark_zone(p, df).direction == "BUY"

    def test_rbd_is_sell(self) -> None:
        df = make_ohlc([100.0, 105.0, 100.0])
        p = make_pattern(
            PatternType.RBD, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        assert mark_zone(p, df).direction == "SELL"


# --------------------------------------------------------------------------- #
# Errors / metadata
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_open_column_rejected(self) -> None:
        df = pd.DataFrame({"close": [100.0]})
        with pytest.raises(ValueError, match="open"):
            mark_zone(_minimal_pattern(), df)


def _minimal_pattern() -> Pattern:
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    impulse = Impulse(
        direction="RALLY",
        start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=5.0, largest_body=5.0, candle_count=1,
    )
    base = Base(
        start_index=0, end_index=0, candle_count=1,
        top=100.0, bottom=100.0, range_size=0.0, largest_body=0.0,
    )
    return Pattern(
        pattern_type=PatternType.RBR,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction="BUY",
        formed_at=ts,
    )


class TestMetadata:
    def test_formed_at_from_pattern(self) -> None:
        df = make_ohlc([100.0, 100.0, 105.0])
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        assert zone.formed_at == pattern.formed_at

    def test_source_pattern_identity(self) -> None:
        df = make_ohlc([100.0, 100.0, 105.0])
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        assert zone.source_pattern is pattern


# --------------------------------------------------------------------------- #
# PR #57: rejection-wick extension. Widens the zone's top/bottom by ±N bars
# of the base to capture institutional rejection wicks that border the base
# but are technically classified as impulse bars by detect_bases.
# --------------------------------------------------------------------------- #


class TestWickExtension:
    """``wick_extend_bars`` widens the zone to include border-bar wicks."""

    def test_extend_zero_matches_strict_base(self) -> None:
        # ``wick_extend_bars=0`` (the pre-PR-#57 default) returns the
        # base's own wick-inclusive envelope, ignoring border bars.
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0, 102.0, 105.0],
            closes=[100.0, 101.0, 100.5, 105.0, 110.0],
            highs=[101.0, 102.0, 101.5, 106.0, 111.0],
            lows=[94.0, 99.0, 99.5, 101.0, 104.0],
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),       # base = bars 1 and 2
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=0)
        # base.top = max(highs[1], highs[2]) = max(102.0, 101.5) = 102.0
        # base.bottom = min(lows[1], lows[2]) = min(99.0, 99.5) = 99.0
        assert zone.top == 102.0
        assert zone.bottom == 99.0

    def test_extend_one_captures_border_wicks(self) -> None:
        # Border bars (index 0 and 3) have wicks extending BEYOND the
        # base's range — those wicks should now be in the zone.
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0, 102.0, 105.0],
            closes=[100.0, 101.0, 100.5, 105.0, 110.0],
            highs=[101.0, 102.0, 101.5, 108.0, 111.0],  # bar 3 high = 108.0
            lows=[94.0, 99.0, 99.5, 101.0, 104.0],       # bar 0 low = 94.0
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # Now scans bars 0..3: top = max(101, 102, 101.5, 108) = 108
        # bottom = min(94, 99, 99.5, 101) = 94
        assert zone.top == 108.0
        assert zone.bottom == 94.0

    def test_border_wicks_inside_zone_leave_unchanged(self) -> None:
        # Border bars exist but don't extend past the base's range —
        # the zone is unchanged from the strict-base result.
        df = make_ohlc(
            opens=[101.0, 100.0, 101.0, 102.0, 105.0],
            closes=[101.5, 101.0, 100.5, 105.0, 110.0],
            highs=[101.8, 102.0, 101.5, 101.9, 111.0],  # borders <= base.top
            lows=[100.5, 99.0, 99.5, 99.8, 104.0],       # borders >= base.bottom
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # Same as wick_extend_bars=0 — no border wick is outside base bounds.
        assert zone.top == 102.0
        assert zone.bottom == 99.0

    def test_extend_two_captures_further_wicks(self) -> None:
        # ``wick_extend_bars=2`` reaches one bar further on each side.
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0, 102.0, 102.5, 105.0],
            closes=[100.0, 101.0, 100.5, 102.5, 105.0, 110.0],
            highs=[101.0, 102.0, 101.5, 102.8, 115.0, 111.0],  # bar 4 high
            lows=[94.0, 99.0, 99.5, 101.5, 102.0, 80.0],       # bar 5 low (out of reach for N=2)
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(2, 3),       # base = bars 2 and 3
            imp_after_indices=(4, 5),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=2)
        # Scans bars 0..5: top = max(101, 102, 101.5, 102.8, 115, 111) = 115
        # bottom = min(94, 99, 99.5, 101.5, 102, 80) = 80
        assert zone.top == 115.0
        assert zone.bottom == 80.0

    def test_extend_clips_to_df_start(self) -> None:
        # Base at the start of df — extension can't walk off the left.
        df = make_ohlc(
            opens=[100.0, 101.0, 105.0],
            closes=[101.0, 100.5, 110.0],
            highs=[102.0, 101.5, 111.0],
            lows=[99.0, 99.5, 104.0],
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),  # impulse_before = bar 0
            base_indices=(0, 1),         # base starts at bar 0
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # ext_start = max(0, 0 - 1) = 0. ext_end = min(2, 1 + 1) = 2.
        # Scan bars 0..2: top = max(102, 101.5, 111) = 111, bottom = 99.
        assert zone.top == 111.0
        assert zone.bottom == 99.0

    def test_extend_clips_to_df_end(self) -> None:
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0],
            closes=[100.0, 101.0, 100.5],
            highs=[101.0, 102.0, 101.5],
            lows=[94.0, 99.0, 99.5],
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),         # base ends at last bar
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # ext_end clipped to df.len - 1 = 2. ext_start = 0.
        # Scan bars 0..2: top = max(101, 102, 101.5) = 102, bottom = min = 94.
        assert zone.top == 102.0
        assert zone.bottom == 94.0

    def test_negative_extend_raises(self) -> None:
        df = make_ohlc([100.0, 100.0, 105.0])
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        with pytest.raises(ValueError, match="wick_extend_bars must be >= 0"):
            mark_zone(pattern, df, wick_extend_bars=-1)

    def test_sell_zone_captures_rejection_wick_above_base(self) -> None:
        # Operator's production case: SELL (RBD). The first bar of
        # impulse_after has a high wick reaching ABOVE the base's
        # high — that's the rejection wick that should be in the zone.
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0, 100.5, 95.0],
            closes=[101.0, 101.0, 100.5, 100.0, 90.0],
            highs=[102.0, 102.0, 101.5, 105.0, 95.0],   # bar 3 wicks UP to 105
            lows=[94.0, 99.0, 99.5, 99.5, 89.0],
        )
        pattern = make_pattern(
            PatternType.RBD, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # Pre-PR-#57 (strict base): top would be 102.0 (base only).
        # Post-PR-#57 (extend=1): top is 105.0, including the bar-3 wick.
        assert zone.top == 105.0
        # The original SELL setup's entry would now be 105 instead of
        # 102 — which is what the operator wants (the rejection wick
        # is part of the institutional defence of the level).
