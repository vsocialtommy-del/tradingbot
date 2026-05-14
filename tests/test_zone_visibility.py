"""Tests for ``bot.strategy.zone_visibility`` — body-cross count.

PR #48: the visibility rule says a zone is dead once enough candles
have body-overlapped its price range since formation. The pipeline
gate uses ``count == 0`` as the visible threshold; existing-setup
cancellation uses ``count >= 1``. This module's only job is to compute
the count correctly.

Covered cases:
* No post-formation bars → count = 0
* Candle fully above the zone (body) → not counted
* Candle fully below the zone (body) → not counted
* Candle body spanning the zone (open > top, close < bottom) → 1
* Candle body starting above and ending inside → 1
* Candle body starting inside and ending below → 1
* Wick into zone but body fully outside → not counted (bodies only)
* Multiple candles → counts sum
* Bar with timestamp ≤ ``since_time`` → excluded
* Empty df → 0
"""

from __future__ import annotations

import pandas as pd

from bot.strategy.zone_visibility import (
    count_bodies_through_zone,
    is_zone_visible_for_pipeline,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_df(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """Build a tiny OHLC df. Each row: (iso_time, open, high, low, close)."""
    times = pd.DatetimeIndex(
        [pd.Timestamp(r[0]).tz_convert("UTC") if pd.Timestamp(r[0]).tzinfo
         else pd.Timestamp(r[0]).tz_localize("UTC")
         for r in rows]
    )
    return pd.DataFrame(
        {
            "open":  [r[1] for r in rows],
            "high":  [r[2] for r in rows],
            "low":   [r[3] for r in rows],
            "close": [r[4] for r in rows],
        },
        index=times,
    )


ZONE_TOP = 4694.0
ZONE_BOTTOM = 4689.0
FORMED_AT = pd.Timestamp("2026-05-13T19:00:00Z")


# --------------------------------------------------------------------------- #
# count_bodies_through_zone
# --------------------------------------------------------------------------- #


class TestCountBodies:
    def test_empty_df_returns_zero(self) -> None:
        df = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": []},
            index=pd.DatetimeIndex([], tz="UTC"),
        )
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 0

    def test_no_bars_after_since_time(self) -> None:
        # All bars are AT or BEFORE since_time; nothing counts.
        df = make_df([
            ("2026-05-13T18:55:00Z", 4690.0, 4691.0, 4689.0, 4690.5),
            ("2026-05-13T19:00:00Z", 4690.5, 4691.0, 4690.0, 4690.5),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 0

    def test_candle_fully_above_zone_not_counted(self) -> None:
        # Body 4695 → 4700, fully above zone (4689-4694). Wick down
        # to 4693 would touch the zone but body doesn't.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4695.0, 4700.0, 4693.0, 4700.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 0

    def test_candle_fully_below_zone_not_counted(self) -> None:
        # Body 4685 → 4680, fully below zone.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4685.0, 4686.0, 4680.0, 4680.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 0

    def test_body_fully_spans_zone(self) -> None:
        # Big bearish bar: open 4700, close 4685. Body fully spans
        # the 4689-4694 zone.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4700.0, 4700.0, 4684.0, 4685.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 1

    def test_body_starts_above_ends_inside(self) -> None:
        # Open 4696 (above zone), close 4691 (inside zone). Body
        # overlaps from zone.top down to 4691.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4696.0, 4696.0, 4690.0, 4691.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 1

    def test_body_starts_inside_ends_below(self) -> None:
        # Open 4692 (inside zone), close 4687 (below zone). Body
        # overlaps from 4692 down to zone.bottom.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4692.0, 4692.5, 4686.0, 4687.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 1

    def test_body_fully_inside_zone(self) -> None:
        # Both open and close inside the zone — body fully inside.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4692.0, 4693.5, 4690.5, 4691.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 1

    def test_wick_into_zone_body_outside_not_counted(self) -> None:
        # Wick down to 4690 (into zone) but body 4696 → 4695 stays
        # fully above zone.top. Wicks don't count.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4696.0, 4696.5, 4690.0, 4695.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 0

    def test_body_exactly_at_zone_top_boundary_not_counted(self) -> None:
        # Open == close == zone.top. Body has zero range, max(o,c)
        # = top, so body_top > zone_bottom holds but body_bottom <
        # zone_top fails (4694 < 4694 is False). Boundary case → not
        # counted. Good: prevents zero-range bars from triggering.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4694.0, 4694.0, 4694.0, 4694.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 0

    def test_multiple_bars_count_sums(self) -> None:
        # The spam scenario: 3 bars all bodying through the zone.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4696.0, 4696.0, 4690.0, 4691.0),  # in
            ("2026-05-13T19:10:00Z", 4691.0, 4691.5, 4687.0, 4688.0),  # in
            ("2026-05-13T19:15:00Z", 4688.0, 4688.5, 4684.0, 4685.0),  # below
        ])
        # First two bodies overlap zone; third is fully below.
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 2

    def test_only_post_formation_bars_count(self) -> None:
        # Bar BEFORE formed_at (formation candle itself) doesn't count
        # even if its body is in the zone — by construction, base
        # candles ARE in the zone.
        df = make_df([
            # Before formed_at — base candle, body in zone, excluded.
            ("2026-05-13T18:55:00Z", 4692.0, 4694.0, 4690.0, 4691.0),
            # After formed_at — body in zone, counted.
            ("2026-05-13T19:05:00Z", 4695.0, 4695.5, 4690.0, 4691.0),
        ])
        assert count_bodies_through_zone(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            since_time=FORMED_AT,
        ) == 1


# --------------------------------------------------------------------------- #
# is_zone_visible_for_pipeline — convenience wrapper
# --------------------------------------------------------------------------- #


class TestIsVisibleForPipeline:
    def test_zero_bodies_through_visible(self) -> None:
        # Price rallied away after formation, never bodied back into zone.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4695.0, 4700.0, 4694.5, 4699.0),
            ("2026-05-13T19:10:00Z", 4699.0, 4705.0, 4698.0, 4703.0),
            ("2026-05-13T19:15:00Z", 4703.0, 4708.0, 4701.0, 4706.0),
        ])
        assert is_zone_visible_for_pipeline(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            formed_at=FORMED_AT,
        ) is True

    def test_one_body_through_not_visible_to_pipeline(self) -> None:
        # A single candle bodied through. Pipeline path treats this as
        # not visible (zone has flipped or in transition). The flipped
        # path has its own check; pipeline can't use this zone.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4700.0, 4700.0, 4684.0, 4685.0),
        ])
        assert is_zone_visible_for_pipeline(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            formed_at=FORMED_AT,
        ) is False

    def test_many_bodies_through_not_visible(self) -> None:
        # Definitely dead.
        df = make_df([
            ("2026-05-13T19:05:00Z", 4695.0, 4695.5, 4690.0, 4691.0),
            ("2026-05-13T19:10:00Z", 4691.0, 4691.5, 4687.0, 4688.0),
            ("2026-05-13T19:15:00Z", 4688.0, 4690.0, 4685.0, 4686.0),
        ])
        assert is_zone_visible_for_pipeline(
            df, zone_top=ZONE_TOP, zone_bottom=ZONE_BOTTOM,
            formed_at=FORMED_AT,
        ) is False
