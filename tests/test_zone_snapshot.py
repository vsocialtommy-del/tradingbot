"""Tests for ``bot.visualization.zone_snapshot`` — MT5 chart CSV writer.

PR #49: bot writes a CSV of current zones; companion MQL5 EA reads it
and draws rectangles on the chart. This module's job is the CSV side.

Covers:
* No-op mode when ``MT5_FILES_DIR`` unset
* No-op mode when dir doesn't exist
* CSV format: header row, columns in expected order, types right
* Filters: status, distance, age, visibility (PR #48)
* Atomic write: target file always valid (never half-written)
* Bad zone rows (FLIPPED with null flipped_direction) defensively dropped
* Empty zone list → CSV with header only
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID, uuid4

import pandas as pd
import pytest

from bot.logging.supabase_logger import Zone
from bot.visualization.zone_snapshot import (
    RENDERED_STATUSES,
    ZoneSnapshotConfig,
    ZoneSnapshotWriter,
)


NOW = datetime(2026, 5, 14, 22, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_zone(
    *,
    zone_id: UUID | None = None,
    direction: str = "BUY",
    status: str = "CONFIRMED",
    flipped_direction: str | None = None,
    top: float = 4691.0,
    bottom: float = 4685.0,
    formed_at: datetime | None = None,
    pattern_type: str = "RBR",
) -> Zone:
    return Zone(
        id=zone_id or uuid4(),
        symbol="XAUUSD",
        direction=direction,  # type: ignore[arg-type]
        zone_type="STRONG_POINT",
        pattern_type=pattern_type,  # type: ignore[arg-type]
        top=Decimal(str(top)),
        bottom=Decimal(str(bottom)),
        approach_count=0,
        formed_at=formed_at or NOW - timedelta(hours=2),
        invalidated_at=None,
        last_evaluation_result=None,
        status=status,  # type: ignore[arg-type]
        consumed_at=None,
        violated_at=None,
        flipped_at=None,
        flipped_direction=flipped_direction,  # type: ignore[arg-type]
        created_at=NOW - timedelta(hours=2),
        updated_at=NOW,
    )


def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV file → (header, list-of-rows)."""
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    dict_rows = [dict(zip(header, r, strict=False)) for r in rows[1:]]
    return header, dict_rows


# --------------------------------------------------------------------------- #
# No-op mode
# --------------------------------------------------------------------------- #


class TestNoOpMode:
    def test_unset_env_disables_writer(self) -> None:
        w = ZoneSnapshotWriter(None)
        assert w.enabled is False
        # write should silently return 0.
        n = w.write([make_zone()], current_price=4690.0)
        assert n == 0

    def test_empty_string_disables_writer(self) -> None:
        w = ZoneSnapshotWriter("")
        assert w.enabled is False

    def test_nonexistent_dir_disables_writer(self, tmp_path) -> None:
        nonexistent = str(tmp_path / "does_not_exist")
        w = ZoneSnapshotWriter(nonexistent)
        assert w.enabled is False
        # No file should be created.
        assert not os.path.exists(nonexistent)


# --------------------------------------------------------------------------- #
# CSV format
# --------------------------------------------------------------------------- #


class TestCsvFormat:
    def test_header_row(self, tmp_path) -> None:
        w = ZoneSnapshotWriter(str(tmp_path))
        w.write([], current_price=4690.0, now=NOW)
        header, _ = read_csv(str(tmp_path / "tradingbot_zones.csv"))
        assert header == [
            "zone_id", "direction", "status", "flipped_direction",
            "top", "bottom", "formed_at_unix",
        ]

    def test_one_zone_round_trips(self, tmp_path) -> None:
        zid = uuid4()
        formed = NOW - timedelta(hours=1)
        z = make_zone(
            zone_id=zid, direction="BUY", status="CONFIRMED",
            top=4691.0, bottom=4685.0, formed_at=formed,
        )
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([z], current_price=4690.0, now=NOW)
        assert n == 1
        _, rows = read_csv(str(tmp_path / "tradingbot_zones.csv"))
        assert len(rows) == 1
        r = rows[0]
        assert r["zone_id"] == str(zid)
        assert r["direction"] == "BUY"
        assert r["status"] == "CONFIRMED"
        assert r["flipped_direction"] == ""
        assert float(r["top"]) == pytest.approx(4691.0)
        assert float(r["bottom"]) == pytest.approx(4685.0)
        assert int(r["formed_at_unix"]) == int(formed.timestamp())

    def test_flipped_zone_emits_flipped_direction(self, tmp_path) -> None:
        z = make_zone(
            direction="BUY", status="FLIPPED",
            flipped_direction="SELL",
            top=4691.0, bottom=4685.0,
        )
        w = ZoneSnapshotWriter(str(tmp_path))
        w.write([z], current_price=4690.0, now=NOW)
        _, rows = read_csv(str(tmp_path / "tradingbot_zones.csv"))
        assert rows[0]["flipped_direction"] == "SELL"

    def test_empty_zones_writes_header_only(self, tmp_path) -> None:
        w = ZoneSnapshotWriter(str(tmp_path))
        w.write([], current_price=4690.0, now=NOW)
        header, rows = read_csv(str(tmp_path / "tradingbot_zones.csv"))
        assert len(header) == 7
        assert rows == []


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


class TestFilters:
    def test_status_filter_drops_consumed(self, tmp_path) -> None:
        zones = [
            make_zone(status="CONFIRMED"),
            make_zone(status="CONSUMED"),
            make_zone(status="VIOLATED"),
            make_zone(status="ACTIVE"),
            make_zone(status="FLIPPED", flipped_direction="SELL"),
        ]
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write(zones, current_price=4690.0, now=NOW)
        # Only CONFIRMED + ACTIVE + FLIPPED survive.
        assert n == 3
        _, rows = read_csv(str(tmp_path / "tradingbot_zones.csv"))
        statuses = {r["status"] for r in rows}
        assert statuses == {"CONFIRMED", "ACTIVE", "FLIPPED"}

    def test_distance_filter_drops_far_zones(self, tmp_path) -> None:
        near = make_zone(top=4691.0, bottom=4685.0)        # near 4690
        far_above = make_zone(top=4800.0, bottom=4794.0)   # +110, outside ±50
        far_below = make_zone(top=4600.0, bottom=4594.0)   # -90,  outside ±50
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([near, far_above, far_below],
                    current_price=4690.0, now=NOW)
        assert n == 1
        _, rows = read_csv(str(tmp_path / "tradingbot_zones.csv"))
        assert float(rows[0]["top"]) == pytest.approx(4691.0)

    def test_distance_filter_keeps_zone_containing_current_price(
        self, tmp_path,
    ) -> None:
        # If current price is INSIDE the zone, it's definitely close.
        z = make_zone(top=4691.0, bottom=4685.0)
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([z], current_price=4688.0, now=NOW)
        assert n == 1

    def test_age_filter_drops_old_zones(self, tmp_path) -> None:
        recent = make_zone(formed_at=NOW - timedelta(days=2))
        ancient = make_zone(formed_at=NOW - timedelta(days=10))
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([recent, ancient], current_price=4690.0, now=NOW)
        assert n == 1

    def test_age_cutoff_is_configurable(self, tmp_path) -> None:
        # Tight 1-day window, both zones older than that → drop both.
        cfg = ZoneSnapshotConfig(draw_age_days=1)
        recent = make_zone(formed_at=NOW - timedelta(days=2))
        w = ZoneSnapshotWriter(str(tmp_path), config=cfg)
        n = w.write([recent], current_price=4690.0, now=NOW)
        assert n == 0

    def test_flipped_with_null_flipped_direction_dropped(
        self, tmp_path,
    ) -> None:
        # Malformed row (shouldn't exist under the CHECK constraint
        # but be defensive — bad DB rows shouldn't crash the EA).
        z = make_zone(status="FLIPPED", flipped_direction=None)
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([z], current_price=4690.0, now=NOW)
        assert n == 0


# --------------------------------------------------------------------------- #
# Visibility filter (PR #48 integration)
# --------------------------------------------------------------------------- #


def make_ohlc_with_bodies_through(
    *,
    zone_top: float, zone_bottom: float,
    formed_at: datetime, n_bodies_after: int,
) -> pd.DataFrame:
    rows = []
    pre = pd.date_range(end=formed_at, periods=20, freq="5min", tz="UTC")
    for _ in pre:
        rows.append((zone_top + 5, zone_top + 6, zone_top + 4, zone_top + 5))
    post = pd.date_range(
        start=formed_at + pd.Timedelta(minutes=5),
        periods=n_bodies_after, freq="5min", tz="UTC",
    )
    mid = (zone_top + zone_bottom) / 2.0
    for _ in post:
        rows.append((zone_top - 0.1, zone_top + 0.5,
                     zone_bottom - 0.5, mid))
    index = pre.union(post)
    return pd.DataFrame(
        {
            "open":  [r[0] for r in rows],
            "high":  [r[1] for r in rows],
            "low":   [r[2] for r in rows],
            "close": [r[3] for r in rows],
        },
        index=index,
    )


class TestVisibilityFilter:
    def test_obscured_zone_dropped_when_df_provided(self, tmp_path) -> None:
        formed = NOW - timedelta(hours=1)
        z = make_zone(top=4691.0, bottom=4685.0, formed_at=formed)
        # 3 post-formation bodies through → obscured (count >= 2).
        df = make_ohlc_with_bodies_through(
            zone_top=4691.0, zone_bottom=4685.0,
            formed_at=formed, n_bodies_after=3,
        )
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([z], current_price=4690.0, df=df, now=NOW)
        assert n == 0

    def test_visible_zone_kept_when_df_provided(self, tmp_path) -> None:
        formed = NOW - timedelta(hours=1)
        z = make_zone(top=4691.0, bottom=4685.0, formed_at=formed)
        df = make_ohlc_with_bodies_through(
            zone_top=4691.0, zone_bottom=4685.0,
            formed_at=formed, n_bodies_after=0,
        )
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([z], current_price=4690.0, df=df, now=NOW)
        assert n == 1

    def test_visibility_filter_skipped_when_df_missing(
        self, tmp_path,
    ) -> None:
        # Without df, the visibility filter can't run. Zone should
        # be emitted (other filters permitting).
        z = make_zone(top=4691.0, bottom=4685.0)
        w = ZoneSnapshotWriter(str(tmp_path))
        n = w.write([z], current_price=4690.0, df=None, now=NOW)
        assert n == 1

    def test_visibility_filter_can_be_disabled(self, tmp_path) -> None:
        cfg = ZoneSnapshotConfig(apply_visibility_filter=False)
        formed = NOW - timedelta(hours=1)
        z = make_zone(top=4691.0, bottom=4685.0, formed_at=formed)
        df = make_ohlc_with_bodies_through(
            zone_top=4691.0, zone_bottom=4685.0,
            formed_at=formed, n_bodies_after=3,
        )
        w = ZoneSnapshotWriter(str(tmp_path), config=cfg)
        # With filter disabled, the obscured zone STILL gets written.
        n = w.write([z], current_price=4690.0, df=df, now=NOW)
        assert n == 1


# --------------------------------------------------------------------------- #
# Atomic write
# --------------------------------------------------------------------------- #


class TestAtomicWrite:
    def test_no_partial_files_left_behind(self, tmp_path) -> None:
        w = ZoneSnapshotWriter(str(tmp_path))
        w.write([make_zone() for _ in range(5)],
                current_price=4690.0, now=NOW)
        # Only the target CSV should exist; no stray .tmp files.
        files = os.listdir(tmp_path)
        assert files == ["tradingbot_zones.csv"]

    def test_overwrites_previous_content(self, tmp_path) -> None:
        w = ZoneSnapshotWriter(str(tmp_path))
        # First write: 3 zones.
        w.write([make_zone() for _ in range(3)],
                current_price=4690.0, now=NOW)
        # Second write: 1 zone. Should fully replace, not append.
        w.write([make_zone()], current_price=4690.0, now=NOW)
        _, rows = read_csv(str(tmp_path / "tradingbot_zones.csv"))
        assert len(rows) == 1


# --------------------------------------------------------------------------- #
# Sanity: the rendered-status set is what we expect
# --------------------------------------------------------------------------- #


def test_rendered_statuses_set() -> None:
    assert RENDERED_STATUSES == frozenset({"CONFIRMED", "ACTIVE", "FLIPPED"})
