"""Tests for ``bot.strategy.strong_point`` — loosened entry rules.

Post May-2026 refinement: validator is a passthrough that returns
``is_strong_point=True`` for any size-filter-passing zone that hasn't
been body-broken since pattern formation. ``compute_sl_price`` returns
``zone bound ± buffer``.
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
    direction: str,
    df: pd.DataFrame,
    base_start: int = 5,
    base_end: int = 7,
    impulse_after_end: int = 10,
    base_top: float = 100.5,
    base_bottom: float = 100.0,
) -> Pattern:
    pt = PatternType.RBR if direction == "BUY" else PatternType.DBD
    impulse_before = Impulse(
        direction="RALLY" if pt == PatternType.RBR else "DROP",
        start_index=0, end_index=base_start - 1,
        start_time=df.index[0], end_time=df.index[base_start - 1],
        range_size=5.0, largest_body=5.0, candle_count=base_start,
    )
    impulse_after = Impulse(
        direction="RALLY" if pt == PatternType.RBR else "DROP",
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
        pattern_type=pt,
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
    df: pd.DataFrame | None = None,
    impulse_after_end: int = 10,
) -> RefinedZone:
    if df is None:
        df = make_df([100.0] * 20)
    pattern = make_pattern(
        direction=direction, df=df,
        base_top=top, base_bottom=bottom,
        impulse_after_end=impulse_after_end,
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


# --------------------------------------------------------------------------- #
# validate_strong_point — passthrough behaviour
# --------------------------------------------------------------------------- #


class TestPassthrough:
    def test_tradeable_buy_zone_is_strong_point(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", df=df)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is True
        assert v.validation_failures == []
        # Loosened-flow shape stability: break-and-close fields are
        # permanently None now.
        assert v.broken_swing is None
        assert v.broken_at is None
        assert v.sl_anchor_swing is None

    def test_tradeable_sell_zone_is_strong_point(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="SELL", df=df)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is True

    def test_not_tradeable_short_circuits(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(
            direction="BUY",
            is_tradeable=False, rejection_reason="ZONE_TOO_NARROW",
            df=df,
        )
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is False
        assert v.validation_failures == ["NOT_TRADEABLE"]


# --------------------------------------------------------------------------- #
# Body-break safety check — ZONE_VIOLATED_BEFORE_RETEST
# --------------------------------------------------------------------------- #


class TestBodyBreakSafety:
    """A freshly-detected pattern whose zone has already been body-broken
    since formation is rejected. The lifecycle dedup (PR #35) can't
    catch this because the zone isn't persisted yet."""

    def test_buy_zone_body_close_below_bottom_rejects(self) -> None:
        # Pattern formed at bar 10 (impulse_after.end_index). Bar 14
        # body-closes at 99.5, which is below zone.bottom=100.0.
        closes = [100.0] * 14 + [99.5] + [100.0] * 5
        df = make_df(closes)
        refined = make_refined(direction="BUY", df=df, impulse_after_end=10)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is False
        assert v.validation_failures == ["ZONE_VIOLATED_BEFORE_RETEST"]

    def test_sell_zone_body_close_above_top_rejects(self) -> None:
        # Zone 100-100.5. Body close at 101 = above zone.top.
        closes = [100.0] * 14 + [101.0] + [100.0] * 5
        df = make_df(closes)
        refined = make_refined(direction="SELL", df=df, impulse_after_end=10)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is False
        assert v.validation_failures == ["ZONE_VIOLATED_BEFORE_RETEST"]

    def test_wick_only_through_zone_does_not_reject(self) -> None:
        # The body-break check is on close, not low/high. A wick poking
        # below zone.bottom while the body closes back inside doesn't
        # invalidate. Mirrors ``zone_lifecycle.check_violation``.
        closes = [100.0] * 14 + [100.2] + [100.0] * 5
        df = make_df(closes)
        # Manually widen the wick at bar 14: low=99.0 (deep poke),
        # close stays 100.2 — inside the 100-100.5 zone.
        df.loc[df.index[14], "low"] = 99.0
        refined = make_refined(direction="BUY", df=df, impulse_after_end=10)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is True

    def test_close_at_zone_bottom_does_not_reject_buy(self) -> None:
        # Strict inequality — close == zone.bottom is NOT a body break.
        closes = [100.0] * 14 + [100.0] + [100.0] * 5
        df = make_df(closes)
        refined = make_refined(direction="BUY", df=df, impulse_after_end=10)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is True

    def test_break_before_formation_is_ignored(self) -> None:
        # The scan only looks at bars AFTER impulse_after.end_index.
        # A close < zone.bottom on bar 3 (before pattern formed at
        # bar 10) shouldn't reject the zone — by formation time the
        # pattern detector already considered that context.
        closes = [100.0] * 20
        closes[3] = 90.0
        df = make_df(closes)
        refined = make_refined(direction="BUY", df=df, impulse_after_end=10)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is True

    def test_no_bars_after_formation(self) -> None:
        # Pattern formed on the very last bar — no scan range. Should
        # be tradeable.
        df = make_df([100.0] * 11)
        refined = make_refined(direction="BUY", df=df, impulse_after_end=10)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is True


# --------------------------------------------------------------------------- #
# compute_sl_price — zone-bound formula
# --------------------------------------------------------------------------- #


class TestComputeSlPrice:
    def test_buy_sl_is_zone_bottom_minus_buffer(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", top=100.5, bottom=100.0, df=df)
        v = validate_strong_point(refined, df)
        sl = compute_sl_price(v)  # default buffer 17.5
        assert sl == pytest.approx(100.0 - 17.5)

    def test_sell_sl_is_zone_top_plus_buffer(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="SELL", top=100.5, bottom=100.0, df=df)
        v = validate_strong_point(refined, df)
        sl = compute_sl_price(v)
        assert sl == pytest.approx(100.5 + 17.5)

    def test_buffer_is_tunable(self) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", top=100.5, bottom=100.0, df=df)
        v = validate_strong_point(refined, df)
        sl = compute_sl_price(v, StrongPointConfig(sl_buffer_points=10.0))
        assert sl == pytest.approx(100.0 - 10.0)

    def test_sl_does_not_read_sl_anchor_swing(self) -> None:
        # Even if sl_anchor_swing happens to be populated by some
        # exotic code path, compute_sl_price ignores it under the new
        # rules. Smoke-test that the formula stays zone-bound.
        df = make_df([100.0] * 20)
        refined = make_refined(direction="BUY", top=100.5, bottom=100.0, df=df)
        v = ValidatedZone(
            direction=refined.direction, top=refined.top, bottom=refined.bottom,
            formed_at=refined.formed_at,
            source_pattern=refined.source_pattern,
            refined_zone=refined,
            is_strong_point=True, validation_failures=[],
            broken_swing=None, broken_at=None,
            sl_anchor_swing=None,
        )
        assert compute_sl_price(v) == pytest.approx(100.0 - 17.5)


# --------------------------------------------------------------------------- #
# Direction symmetry
# --------------------------------------------------------------------------- #


class TestDirectionSymmetry:
    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_validator_and_sl_round_trip(self, direction: str) -> None:
        df = make_df([100.0] * 20)
        refined = make_refined(direction=direction, df=df)
        v = validate_strong_point(refined, df)
        assert v.is_strong_point is True
        sl = compute_sl_price(v)
        if direction == "BUY":
            assert sl < v.bottom
        else:
            assert sl > v.top
