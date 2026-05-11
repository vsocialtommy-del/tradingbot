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
    o = df["open"].iloc[start_index : end_index + 1].to_numpy()
    c = df["close"].iloc[start_index : end_index + 1].to_numpy()
    body_tops = [max(oi, ci) for oi, ci in zip(o, c)]
    body_bottoms = [min(oi, ci) for oi, ci in zip(o, c)]
    bodies = [abs(ci - oi) for oi, ci in zip(o, c)]
    top = max(body_tops)
    bottom = min(body_bottoms)
    return Base(
        start_index=start_index, end_index=end_index,
        candle_count=end_index - start_index + 1,
        top=float(top), bottom=float(bottom),
        range_size=float(top - bottom),
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
        df = make_ohlc(opens, closes=closes)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        assert zone.top == 100.5
        assert zone.bottom == 100.0
        assert zone.direction == "BUY"

    def test_single_bearish_base_candle(self) -> None:
        opens = [100.0, 100.5, 95.0]
        closes = [100.0, 100.0, 95.0]
        df = make_ohlc(opens, closes=closes)
        pattern = make_pattern(
            PatternType.DBD, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        # bearish bar: body_top = open = 100.5; body_bottom = close = 100.0
        assert zone.top == 100.5
        assert zone.bottom == 100.0
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
        df = make_ohlc(opens, closes=closes)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 3),
            imp_after_indices=(4, 4),
        )
        zone = mark_zone(pattern, df)
        # body_tops:   max(100.5, 100.7, 100.7) = 100.7
        # body_bottoms: min(100.0, 100.0, 100.3) = 100.0
        assert zone.top == 100.7
        assert zone.bottom == 100.0

    def test_all_doji_base_produces_zero_height(self) -> None:
        opens = [100.0, 100.0, 100.0, 100.0, 105.0]
        closes = [100.0, 100.0, 100.0, 100.0, 105.0]
        df = make_ohlc(opens, closes=closes)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 3),
            imp_after_indices=(4, 4),
        )
        zone = mark_zone(pattern, df)
        assert zone.top == zone.bottom == 100.0

    def test_wicks_excluded(self) -> None:
        opens = [100.0, 100.0, 105.0]
        closes = [100.0, 100.5, 105.0]
        highs = [100.0, 110.0, 105.0]   # massive wick on base bar
        lows = [100.0, 90.0, 105.0]
        df = make_ohlc(opens, closes=closes, highs=highs, lows=lows)
        pattern = make_pattern(
            PatternType.RBR, df,
            imp_before_indices=(0, 0),
            base_indices=(1, 1),
            imp_after_indices=(2, 2),
        )
        zone = mark_zone(pattern, df)
        # Wick from 90 to 110 ignored; body envelope is 100-100.5.
        assert zone.top == 100.5
        assert zone.bottom == 100.0


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
