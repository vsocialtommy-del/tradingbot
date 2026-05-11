"""Tests for ``bot.strategy.strong_point`` — break-and-close validation.

Constructs RefinedZone + Pattern fixtures directly so we can test
each branch of validation in isolation. Covers BUY and SELL paths
symmetrically.
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
from bot.strategy.strong_point import (
    StrongPointConfig,
    ValidatedZone,
    compute_sl_price,
    validate_strong_point,
)
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_df(closes: list[float], start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    n = len(closes)
    times = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": list(closes),
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": list(closes),
            "volume": [100] * n,
        },
        index=times,
    )


def make_pattern(
    *,
    pattern_type: PatternType,
    direction: str,
    df: pd.DataFrame,
    base_start: int = 5,
    base_end: int = 7,
    impulse_after_end: int = 10,
    base_top: float = 100.5,
    base_bottom: float = 100.0,
) -> Pattern:
    ts = df.index[base_start]
    impulse_before = Impulse(
        direction="RALLY" if pattern_type in (PatternType.RBR, PatternType.RBD) else "DROP",
        start_index=0, end_index=base_start - 1,
        start_time=df.index[0], end_time=df.index[base_start - 1],
        range_size=5.0, largest_body=5.0, candle_count=base_start,
    )
    impulse_after = Impulse(
        direction="RALLY" if pattern_type in (PatternType.RBR, PatternType.DBR) else "DROP",
        start_index=base_end + 1, end_index=impulse_after_end,
        start_time=df.index[base_end + 1], end_time=df.index[impulse_after_end],
        range_size=5.0, largest_body=5.0,
        candle_count=impulse_after_end - base_end,
    )
    base = Base(
        start_index=base_start, end_index=base_end,
        candle_count=base_end - base_start + 1,
        top=base_top, bottom=base_bottom,
        range_size=base_top - base_bottom, largest_body=0.5,
    )
    return Pattern(
        pattern_type=pattern_type,
        impulse_before=impulse_before,
        base=base,
        impulse_after=impulse_after,
        direction=direction,  # type: ignore[arg-type]
        formed_at=df.index[impulse_after_end],
    )


def make_refined(
    *,
    direction: str,
    top: float = 100.5,
    bottom: float = 100.0,
    is_tradeable: bool = True,
    rejection_reason: str | None = None,
    pattern: Pattern | None = None,
    df: pd.DataFrame | None = None,
) -> RefinedZone:
    if df is None:
        df = make_df([100.0] * 20)
    if pattern is None:
        pt = PatternType.RBR if direction == "BUY" else PatternType.DBD
        pattern = make_pattern(
            pattern_type=pt, direction=direction, df=df,
            base_top=top, base_bottom=bottom,
        )
    zone = Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom,
        formed_at=pattern.formed_at,
        source_pattern=pattern,
    )
    return RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom,
        formed_at=pattern.formed_at,
        source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=zone,
    )


def make_swing(idx: int, df: pd.DataFrame, price: float, kind: str) -> Swing:
    return Swing(
        index=idx, time=df.index[idx], price=price,
        kind=kind,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# NOT_TRADEABLE short-circuit
# --------------------------------------------------------------------------- #


class TestNotTradeable:
    def test_zone_failing_size_filter_skips_validation(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(
            direction="BUY",
            is_tradeable=False, rejection_reason="ZONE_TOO_NARROW",
            df=df,
        )
        v = validate_strong_point(refined, df, swings=[])
        assert v.is_strong_point is False
        assert v.validation_failures == ["NOT_TRADEABLE"]
        # SL anchor + broken swing not even computed.
        assert v.sl_anchor_swing is None
        assert v.broken_swing is None


# --------------------------------------------------------------------------- #
# Missing structural swings
# --------------------------------------------------------------------------- #


class TestMissingSwings:
    def test_buy_zone_no_swing_above(self) -> None:
        # Zone at 100-100.5; no swing high exists above it in the swing list.
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", df=df)
        swings = [make_swing(2, df, price=99.0, kind="LOW")]  # only a low below
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is False
        assert "NO_SWING_ABOVE" in v.validation_failures
        # SL anchor was found (the low below).
        assert v.sl_anchor_swing is not None
        assert v.sl_anchor_swing.price == 99.0

    def test_sell_zone_no_swing_below(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="SELL", df=df)
        swings = [make_swing(2, df, price=101.0, kind="HIGH")]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is False
        assert "NO_SWING_BELOW" in v.validation_failures

    def test_no_sl_anchor(self) -> None:
        # BUY zone with no low swing below → cannot pin SL.
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", df=df)
        # Only swing highs above the zone, no lows below.
        swings = [make_swing(2, df, price=101.0, kind="HIGH")]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is False
        assert v.validation_failures == ["NO_SL_ANCHOR"]


# --------------------------------------------------------------------------- #
# Break-and-close (validation success)
# --------------------------------------------------------------------------- #


class TestBreakAndClose:
    def test_buy_break_above_nearest_high_with_body_close(self) -> None:
        # 20 quiet bars at 100 + pattern formed by bar 10 +
        # bar 15 closes above 101 (the nearest swing high). → SP
        closes = [100.0] * 14 + [102.0] + [100.0] * 5
        df = make_df(closes)
        refined = make_refined(direction="BUY", df=df)
        swings = [
            make_swing(8, df, price=99.0, kind="LOW"),    # SL anchor
            make_swing(9, df, price=101.0, kind="HIGH"),  # break target
        ]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is True
        assert v.validation_failures == []
        assert v.broken_swing is not None
        assert v.broken_swing.price == 101.0
        assert v.broken_at == df.index[14]
        assert v.sl_anchor_swing is not None
        assert v.sl_anchor_swing.price == 99.0

    def test_sell_break_below_nearest_low_with_body_close(self) -> None:
        closes = [100.0] * 14 + [98.0] + [100.0] * 5
        df = make_df(closes)
        refined = make_refined(direction="SELL", df=df)
        swings = [
            make_swing(8, df, price=101.0, kind="HIGH"),  # SL anchor
            make_swing(9, df, price=99.0, kind="LOW"),    # break target
        ]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is True
        assert v.broken_swing.price == 99.0
        assert v.sl_anchor_swing.price == 101.0


class TestNearestSwingSelection:
    def test_buy_uses_lowest_priced_high_above(self) -> None:
        # Multiple highs above zone — should pick the LOWEST (closest to zone top).
        closes = [100.0] * 14 + [101.0] + [100.0] * 5  # body close at 101 — clears 100.8 but not 105
        df = make_df(closes)
        refined = make_refined(direction="BUY", df=df)
        swings = [
            make_swing(8, df, price=99.0, kind="LOW"),
            make_swing(9, df, price=100.8, kind="HIGH"),  # nearest (lowest)
            make_swing(9, df, price=105.0, kind="HIGH"),  # higher up
        ]
        v = validate_strong_point(refined, df, swings=swings)
        # The 101.0 close clears 100.8 (broken_swing) so SP confirmed.
        assert v.is_strong_point is True
        assert v.broken_swing.price == 100.8

    def test_sell_uses_highest_priced_low_below(self) -> None:
        closes = [100.0] * 14 + [99.5] + [100.0] * 5  # clears 99.7 not 90
        df = make_df(closes)
        refined = make_refined(direction="SELL", df=df)
        swings = [
            make_swing(8, df, price=101.0, kind="HIGH"),
            make_swing(9, df, price=99.7, kind="LOW"),   # nearest (highest)
            make_swing(9, df, price=90.0, kind="LOW"),   # lower
        ]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is True
        assert v.broken_swing.price == 99.7

    def test_buy_sl_anchor_is_highest_low_below(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", df=df)
        swings = [
            make_swing(8, df, price=95.0, kind="LOW"),   # further below
            make_swing(9, df, price=99.0, kind="LOW"),   # nearest (highest)
            make_swing(10, df, price=101.0, kind="HIGH"),
        ]
        v = validate_strong_point(refined, df, swings=swings)
        # SL anchor should be 99.0 (the nearest low — highest-priced below).
        assert v.sl_anchor_swing.price == 99.0


# --------------------------------------------------------------------------- #
# NO_BREAK_YET vs INVALIDATED
# --------------------------------------------------------------------------- #


class TestPendingVsInvalidated:
    def test_no_break_yet_when_no_post_pattern_break(self) -> None:
        # Zone formed, no body close past 101.
        closes = [100.0] * 20
        df = make_df(closes)
        refined = make_refined(direction="BUY", df=df)
        swings = [
            make_swing(8, df, price=99.0, kind="LOW"),
            make_swing(9, df, price=101.0, kind="HIGH"),
        ]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is False
        assert v.validation_failures == ["NO_BREAK_YET"]
        # SL anchor still populated (we found it before the scan).
        assert v.sl_anchor_swing is not None

    def test_buy_invalidated_by_body_close_below_zone_bottom(self) -> None:
        # Zone 100-100.5. After pattern, bar 14 closes at 99.5 → invalidated.
        closes = [100.0] * 14 + [99.5] + [100.0] * 5
        df = make_df(closes)
        refined = make_refined(direction="BUY", df=df)
        swings = [
            make_swing(8, df, price=99.0, kind="LOW"),
            make_swing(9, df, price=101.0, kind="HIGH"),
        ]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is False
        assert "INVALIDATED" in v.validation_failures
        assert v.broken_at is None

    def test_sell_invalidated_by_body_close_above_zone_top(self) -> None:
        closes = [100.0] * 14 + [101.0] + [100.0] * 5
        df = make_df(closes)
        refined = make_refined(direction="SELL", df=df)
        swings = [
            make_swing(8, df, price=101.5, kind="HIGH"),  # well above zone
            make_swing(9, df, price=99.0, kind="LOW"),
        ]
        v = validate_strong_point(refined, df, swings=swings)
        assert v.is_strong_point is False
        assert "INVALIDATED" in v.validation_failures


# --------------------------------------------------------------------------- #
# compute_sl_price
# --------------------------------------------------------------------------- #


class TestComputeSlPrice:
    def test_buy_sl_below_anchor(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", df=df)
        anchor = make_swing(8, df, price=99.0, kind="LOW")
        validated = ValidatedZone(
            direction="BUY", top=100.5, bottom=100.0,
            formed_at=refined.formed_at,
            source_pattern=refined.source_pattern,
            refined_zone=refined,
            is_strong_point=True,
            validation_failures=[],
            broken_swing=make_swing(9, df, price=101.0, kind="HIGH"),
            broken_at=df.index[14],
            sl_anchor_swing=anchor,
        )
        sl = compute_sl_price(validated, StrongPointConfig(sl_buffer_points=17.5))
        # BUY: SL = anchor - buffer = 99 - 17.5 = 81.5.
        assert sl == 81.5

    def test_sell_sl_above_anchor(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="SELL", df=df)
        anchor = make_swing(8, df, price=101.0, kind="HIGH")
        validated = ValidatedZone(
            direction="SELL", top=100.5, bottom=100.0,
            formed_at=refined.formed_at,
            source_pattern=refined.source_pattern,
            refined_zone=refined,
            is_strong_point=True,
            validation_failures=[],
            broken_swing=make_swing(9, df, price=99.0, kind="LOW"),
            broken_at=df.index[14],
            sl_anchor_swing=anchor,
        )
        sl = compute_sl_price(validated, StrongPointConfig(sl_buffer_points=17.5))
        # SELL: SL = anchor + buffer = 101 + 17.5 = 118.5.
        assert sl == 118.5

    def test_no_anchor_raises(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", df=df)
        validated = ValidatedZone(
            direction="BUY", top=100.5, bottom=100.0,
            formed_at=refined.formed_at,
            source_pattern=refined.source_pattern,
            refined_zone=refined,
            is_strong_point=False,
            validation_failures=["NO_SL_ANCHOR"],
            broken_swing=None, broken_at=None,
            sl_anchor_swing=None,
        )
        with pytest.raises(ValueError, match="sl_anchor_swing"):
            compute_sl_price(validated)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


class TestDefaults:
    def test_strong_point_config_defaults(self) -> None:
        c = StrongPointConfig()
        assert c.sl_buffer_points == 17.5
