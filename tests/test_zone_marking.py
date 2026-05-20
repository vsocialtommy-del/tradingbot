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
    """``wick_extend_bars`` widens the zone in the **rejection direction only**.

    PR #60: BUY zones widen ``bottom`` (lower rejection wicks); SELL
    zones widen ``top`` (upper rejection wicks). The opposite side is
    the impulse itself, not a rejection wick, and is left at base.
    """

    def test_extend_zero_matches_strict_base(self) -> None:
        # ``wick_extend_bars=0`` returns the base's own wick-inclusive
        # envelope, ignoring border bars (in either direction).
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

    def test_buy_extend_one_widens_bottom_only(self) -> None:
        # BUY (RBR) — bar 0 (impulse_before's last rally bar) has a
        # low wick at 94, BELOW base.bottom. That's a real rejection
        # wick: demand defended that level. bar 3 (impulse_after's
        # first rally bar) has a high wick at 108, ABOVE base.top —
        # but that's the rally taking off, NOT a rejection wick, so
        # it must NOT pull ``top`` up.
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
        # top stays at base.top (102) — the high wick at 108 on the
        # impulse_after bar is part of the RALLY, not a rejection.
        # bottom widens to min(99.0, 94.0) = 94.0 — real rejection wick.
        assert zone.top == 102.0
        assert zone.bottom == 94.0

    def test_sell_extend_one_widens_top_only(self) -> None:
        # SELL (RBD) — bar 3 (impulse_after's first drop bar) has a
        # high wick at 105, ABOVE base.top. That's a real rejection
        # wick: supply defended that level. bar 0 (impulse_before's
        # last rally bar) has a low wick at 94, BELOW base.bottom —
        # but that's a body of the prior rally, NOT a rejection wick,
        # so it must NOT pull ``bottom`` down.
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0, 100.5, 95.0],
            closes=[101.0, 101.0, 100.5, 100.0, 90.0],
            highs=[102.0, 102.0, 101.5, 105.0, 95.0],   # bar 3 wicks UP to 105
            lows=[94.0, 99.0, 99.5, 99.5, 89.0],         # bar 0 wicks down to 94
        )
        pattern = make_pattern(
            PatternType.RBD, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # top widens to max(102.0, 105.0) = 105.0 — real rejection wick.
        # bottom stays at base.bottom (99) — the low wick at 94 is on
        # the rally-up bar, not a rejection.
        assert zone.top == 105.0
        assert zone.bottom == 99.0

    def test_buy_border_wicks_inside_zone_leave_unchanged(self) -> None:
        # BUY with border bars but no lower wick below base.bottom —
        # zone is unchanged from strict-base.
        df = make_ohlc(
            opens=[101.0, 100.0, 101.0, 102.0, 105.0],
            closes=[101.5, 101.0, 100.5, 105.0, 110.0],
            highs=[101.8, 102.0, 101.5, 101.9, 111.0],
            lows=[100.5, 99.0, 99.5, 99.8, 104.0],       # all borders >= base.bottom
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        assert zone.top == 102.0
        assert zone.bottom == 99.0

    def test_buy_extend_two_reaches_further_lower_wick(self) -> None:
        # BUY (RBR), N=2 reaches one bar further on each side. Only
        # lower wicks matter; the high outlier at bar 4 must be
        # ignored (it's in impulse_after).
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0, 102.0, 102.5, 105.0],
            closes=[100.0, 101.0, 100.5, 102.5, 105.0, 110.0],
            highs=[101.0, 102.0, 101.5, 102.8, 115.0, 111.0],  # bar 4 high (ignored)
            lows=[80.0, 99.0, 99.5, 101.5, 102.0, 104.0],       # bar 0 low = 80
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(2, 3),       # base = bars 2 and 3
            imp_after_indices=(4, 5),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=2)
        # Scans bars 0..5. top stays at base.top = max(101.5, 102.8) = 102.8.
        # bottom widens to min(99.5, 101.5, 80, 99, 102, 104) = 80.
        assert zone.top == 102.8
        assert zone.bottom == 80.0

    def test_buy_extend_clips_to_df_start(self) -> None:
        # Base at the start of df — extension can't walk off the left.
        df = make_ohlc(
            opens=[100.0, 101.0, 105.0],
            closes=[101.0, 100.5, 110.0],
            highs=[102.0, 101.5, 111.0],
            lows=[99.0, 95.0, 104.0],                    # bar 1 low = 95
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(0, 1),         # base starts at bar 0
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # ext_start = max(0, 0-1) = 0. ext_end = min(2, 1+1) = 2.
        # BUY: top stays at base.top = max(102, 101.5) = 102.
        # bottom widens to min(99, 95, 104) = 95.
        assert zone.top == 102.0
        assert zone.bottom == 95.0

    def test_sell_extend_clips_to_df_end(self) -> None:
        df = make_ohlc(
            opens=[105.0, 100.0, 101.0],
            closes=[100.0, 100.5, 99.5],
            highs=[106.0, 102.0, 103.0],                 # bar 2 high = 103
            lows=[99.0, 99.5, 99.0],
        )
        pattern = make_pattern(
            PatternType.RBD, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),         # base ends at last bar
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # ext_end clipped to df.len-1 = 2. ext_start = 0.
        # SELL: top widens to max(102, 103, 106) = 106.
        # bottom stays at base.bottom = min(99.5, 99) = 99.
        assert zone.top == 106.0
        assert zone.bottom == 99.0

    def test_negative_extend_raises(self) -> None:
        df = make_ohlc([100.0, 100.0, 105.0])
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0), base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        with pytest.raises(ValueError, match="wick_extend_bars must be >= 0"):
            mark_zone(pattern, df, wick_extend_bars=-1)

    def test_buy_zone_ignores_upper_border_wick(self) -> None:
        # Regression for the bug PR #60 fixes: a BUY zone where the
        # impulse_after's first bar has a tall upper wick (the rally
        # taking off) must NOT widen the zone top. Pre-PR-#60 the
        # symmetric extension pulled top up to the rally's peak.
        df = make_ohlc(
            opens=[100.0, 100.0, 101.0, 102.0, 105.0],
            closes=[101.0, 101.0, 100.5, 105.0, 110.0],
            highs=[101.5, 102.0, 101.5, 120.0, 111.0],   # bar 3 high = 120 (rally peak)
            lows=[100.0, 99.0, 99.5, 101.0, 104.0],
        )
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # Direction-aware: top stays at base.top = 102.0, NOT 120.0.
        assert zone.top == 102.0

    def test_sell_zone_ignores_lower_border_wick(self) -> None:
        # Mirror: SELL zone where the impulse_after's first bar has a
        # deep lower wick (the drop taking off) must NOT widen the
        # zone bottom.
        df = make_ohlc(
            opens=[95.0, 100.0, 101.0, 100.5, 95.0],
            closes=[101.0, 101.0, 100.5, 100.0, 90.0],
            highs=[102.0, 102.0, 101.5, 100.8, 95.0],
            lows=[94.0, 99.0, 99.5, 80.0, 89.0],         # bar 3 low = 80 (drop trough)
        )
        pattern = make_pattern(
            PatternType.RBD, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 2),
            imp_after_indices=(3, 4),
        )
        zone = mark_zone(pattern, df, wick_extend_bars=1)
        # Direction-aware: bottom stays at base.bottom = 99.0, NOT 80.0.
        assert zone.bottom == 99.0
