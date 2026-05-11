"""Tests for ``bot.strategy.zone_refinement`` — size-filter verdict.

Post-PR #31, refinement is just a passthrough + size filter. The
body-extreme math moved into ``zone_marking`` (which reads the
pattern's base directly). These tests pin down the size filter only.
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
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import (
    RefinedZone,
    RefinementConfig,
    refine_zone,
)


def make_df(n: int = 5) -> pd.DataFrame:
    times = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * n, "high": [100.5] * n,
            "low": [99.5] * n, "close": [100.0] * n,
            "volume": [100] * n,
        },
        index=times,
    )


def make_pattern() -> Pattern:
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    impulse = Impulse(
        direction="RALLY",
        start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=5.0, largest_body=5.0, candle_count=1,
    )
    base = Base(
        start_index=1, end_index=1, candle_count=1,
        top=100.5, bottom=100.0, range_size=0.5, largest_body=0.5,
    )
    return Pattern(
        pattern_type=PatternType.RBR,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction="BUY",
        formed_at=ts,
    )


def make_zone(top: float, bottom: float, direction: str = "BUY") -> Zone:
    return Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom,
        formed_at=pd.Timestamp("2026-01-01", tz="UTC"),
        source_pattern=make_pattern(),
    )


# --------------------------------------------------------------------------- #
# Passthrough
# --------------------------------------------------------------------------- #


class TestPassthrough:
    def test_refined_top_bottom_match_input(self) -> None:
        # Refinement no longer mutates top/bottom — it passes them
        # through unchanged and just attaches a tradeability verdict.
        zone = make_zone(top=105.5, bottom=100.0)
        df = make_df()
        refined = refine_zone(zone, df)
        assert refined.top == 105.5
        assert refined.bottom == 100.0

    def test_direction_passthrough(self) -> None:
        zone = make_zone(top=100.5, bottom=100.0, direction="SELL")
        refined = refine_zone(zone, make_df())
        assert refined.direction == "SELL"

    def test_original_zone_identity_preserved(self) -> None:
        zone = make_zone(top=100.5, bottom=100.0)
        refined = refine_zone(zone, make_df())
        assert refined.original_zone is zone


# --------------------------------------------------------------------------- #
# Size filter — boundaries
# --------------------------------------------------------------------------- #


class TestSizeFilter:
    def test_width_at_minimum_is_tradeable(self) -> None:
        # width = 5 — exact lower bound (inclusive).
        zone = make_zone(top=105.0, bottom=100.0)
        refined = refine_zone(zone, make_df())
        assert refined.is_tradeable is True
        assert refined.rejection_reason is None

    def test_width_just_below_minimum_rejected(self) -> None:
        zone = make_zone(top=104.99, bottom=100.0)
        refined = refine_zone(zone, make_df())
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"

    def test_width_at_maximum_is_tradeable(self) -> None:
        zone = make_zone(top=180.0, bottom=100.0)  # width 80
        refined = refine_zone(zone, make_df())
        assert refined.is_tradeable is True

    def test_width_just_above_maximum_rejected(self) -> None:
        zone = make_zone(top=180.01, bottom=100.0)
        refined = refine_zone(zone, make_df())
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_WIDE"

    def test_zero_width_rejected(self) -> None:
        zone = make_zone(top=100.0, bottom=100.0)
        refined = refine_zone(zone, make_df())
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_custom_min_threshold(self) -> None:
        zone = make_zone(top=105.0, bottom=100.0)  # width 5
        cfg = RefinementConfig(zone_min_size_points=6.0)
        refined = refine_zone(zone, make_df(), cfg)
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_NARROW"

    def test_custom_max_threshold(self) -> None:
        zone = make_zone(top=150.0, bottom=100.0)  # width 50
        cfg = RefinementConfig(zone_max_size_points=40.0)
        refined = refine_zone(zone, make_df(), cfg)
        assert refined.is_tradeable is False
        assert refined.rejection_reason == "ZONE_TOO_WIDE"

    def test_default_config_when_none_passed(self) -> None:
        zone = make_zone(top=110.0, bottom=100.0)
        refined = refine_zone(zone, make_df())
        assert refined.is_tradeable is True


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TestErrors:
    def test_missing_open_column_rejected(self) -> None:
        zone = make_zone(top=105.0, bottom=100.0)
        df = make_df().drop(columns=["open"])
        with pytest.raises(ValueError, match="open"):
            refine_zone(zone, df)

    def test_inverted_zone_rejected(self) -> None:
        zone = Zone(
            direction="BUY",
            top=100.0, bottom=105.0,
            formed_at=pd.Timestamp("2026-01-01", tz="UTC"),
            source_pattern=make_pattern(),
        )
        with pytest.raises(ValueError, match="inverted"):
            refine_zone(zone, make_df())
