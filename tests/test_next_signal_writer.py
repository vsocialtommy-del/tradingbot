"""Tests for ``bot.signals.next_signal_writer`` — dashboard signal feeder.

PR #51: the bot maintains a small ``signals`` table holding the
closest pending BUY + SELL zone, refreshed on every M5 close. The
Vercel dashboard reads it.

Covers:
* No qualifying zones → all active rows in that direction deactivated.
* One BUY zone in range → upserted with full trade levels.
* Both BUY + SELL zones → both upserted independently.
* Distance filter excludes zones too far from current price.
* Status filter excludes CONSUMED / VIOLATED rows.
* Direction-correct gate: BUY zones below current price, SELL above
  (the side that allows a still-pending retest).
* FLIPPED zone with ``flipped_direction`` matching counts.
* Closest-by-entry-distance picked when multiple qualify.
* Failure on UPSERT → caught, logged, doesn't crash; the other
  direction still processes.
* Failure on get_zones → both directions report error.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.logging.supabase_logger import SignalInput, SupabaseLogger, Zone
from bot.signals.next_signal_writer import (
    NextSignalConfig,
    NextSignalWriter,
    _pick_closest_zone,
    _zone_is_pending_retest,
    _zone_matches_direction,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


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


def make_df(
    *,
    n_bars: int = 60,
    base_price: float = 4700.0,
) -> pd.DataFrame:
    """Build an OHLC df with synthetic local highs and lows so the TP
    chain has something to find."""
    times = pd.date_range(
        end=NOW, periods=n_bars, freq="5min", tz="UTC",
    )
    # Plant a clean stair-step pattern of local highs above and lows
    # below the base price so find_nearest_local_peak has anchors.
    highs = [base_price + 5.0] * n_bars
    lows = [base_price - 5.0] * n_bars
    # Local highs at indices 10, 20, 30, 40 climbing.
    for i, idx in enumerate((10, 20, 30, 40)):
        highs[idx] = base_price + 10.0 + (i + 1) * 5.0  # 4715, 4720, 4725, 4730
    # Local lows mirroring downward.
    for i, idx in enumerate((10, 20, 30, 40)):
        lows[idx] = base_price - 10.0 - (i + 1) * 5.0  # 4685, 4680, 4675, 4670
    return pd.DataFrame(
        {
            "open": [base_price] * n_bars,
            "high": highs,
            "low": lows,
            "close": [base_price] * n_bars,
        },
        index=times,
    )


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=SupabaseLogger)


@pytest.fixture
def writer(mock_supabase: MagicMock) -> NextSignalWriter:
    return NextSignalWriter(mock_supabase)


# --------------------------------------------------------------------------- #
# Direction + retest helpers
# --------------------------------------------------------------------------- #


class TestZoneMatchesDirection:
    def test_confirmed_buy_matches_buy(self) -> None:
        z = make_zone(direction="BUY", status="CONFIRMED")
        assert _zone_matches_direction(z, "BUY") is True
        assert _zone_matches_direction(z, "SELL") is False

    def test_active_sell_matches_sell(self) -> None:
        z = make_zone(direction="SELL", status="ACTIVE")
        assert _zone_matches_direction(z, "SELL") is True
        assert _zone_matches_direction(z, "BUY") is False

    def test_flipped_uses_flipped_direction(self) -> None:
        # Original SELL zone, now FLIPPED to BUY.
        z = make_zone(
            direction="SELL", status="FLIPPED", flipped_direction="BUY",
        )
        assert _zone_matches_direction(z, "BUY") is True
        assert _zone_matches_direction(z, "SELL") is False


class TestZoneIsPendingRetest:
    def test_buy_above_zone(self) -> None:
        # Price above zone.top → pending retest down to entry.
        z = make_zone(direction="BUY", top=4691.0, bottom=4685.0)
        assert _zone_is_pending_retest(z, "BUY", bid=4700.0, ask=4700.1)

    def test_buy_inside_zone_not_pending(self) -> None:
        z = make_zone(direction="BUY", top=4691.0, bottom=4685.0)
        # bid inside the zone — retest already happened.
        assert not _zone_is_pending_retest(z, "BUY", bid=4688.0, ask=4688.1)

    def test_sell_below_zone(self) -> None:
        z = make_zone(direction="SELL", top=4711.0, bottom=4705.0)
        assert _zone_is_pending_retest(z, "SELL", bid=4699.9, ask=4700.0)

    def test_sell_inside_zone_not_pending(self) -> None:
        z = make_zone(direction="SELL", top=4711.0, bottom=4705.0)
        assert not _zone_is_pending_retest(
            z, "SELL", bid=4707.9, ask=4708.0,
        )


# --------------------------------------------------------------------------- #
# _pick_closest_zone
# --------------------------------------------------------------------------- #


class TestPickClosestZone:
    def test_picks_nearest_buy_zone(self) -> None:
        far = make_zone(direction="BUY", top=4670.0, bottom=4664.0)
        near = make_zone(direction="BUY", top=4690.0, bottom=4684.0)
        # Current bid 4700 → near.top is 10 away, far.top is 30.
        result = _pick_closest_zone(
            direction="BUY",
            zones=[far, near],
            bid=4700.0, ask=4700.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is near

    def test_skips_wrong_direction(self) -> None:
        sell = make_zone(direction="SELL", top=4690.0, bottom=4684.0)
        result = _pick_closest_zone(
            direction="BUY",
            zones=[sell],
            bid=4700.0, ask=4700.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is None

    def test_skips_consumed_status(self) -> None:
        z = make_zone(
            direction="BUY", status="CONSUMED",
            top=4690.0, bottom=4684.0,
        )
        result = _pick_closest_zone(
            direction="BUY",
            zones=[z],
            bid=4700.0, ask=4700.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is None

    def test_skips_violated_status(self) -> None:
        z = make_zone(
            direction="BUY", status="VIOLATED",
            top=4690.0, bottom=4684.0,
        )
        result = _pick_closest_zone(
            direction="BUY",
            zones=[z],
            bid=4700.0, ask=4700.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is None

    def test_includes_flipped_with_matching_flipped_direction(self) -> None:
        # Original SELL, now FLIPPED → BUY.
        flipped = make_zone(
            direction="SELL", status="FLIPPED",
            flipped_direction="BUY",
            top=4690.0, bottom=4684.0,
        )
        result = _pick_closest_zone(
            direction="BUY",
            zones=[flipped],
            bid=4700.0, ask=4700.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is flipped

    def test_distance_filter_excludes_far_zones(self) -> None:
        far = make_zone(direction="BUY", top=4640.0, bottom=4634.0)
        result = _pick_closest_zone(
            direction="BUY",
            zones=[far],
            bid=4700.0, ask=4700.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is None

    def test_age_filter_excludes_old_zones(self) -> None:
        old = make_zone(
            direction="BUY", top=4690.0, bottom=4684.0,
            formed_at=NOW - timedelta(days=10),
        )
        result = _pick_closest_zone(
            direction="BUY",
            zones=[old],
            bid=4700.0, ask=4700.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is None

    def test_pending_retest_gate_excludes_inside_zone(self) -> None:
        # bid inside the zone → not a pending retest.
        z = make_zone(direction="BUY", top=4691.0, bottom=4685.0)
        result = _pick_closest_zone(
            direction="BUY",
            zones=[z],
            bid=4688.0, ask=4688.1, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is None

    def test_picks_nearest_sell_zone(self) -> None:
        far = make_zone(direction="SELL", top=4730.0, bottom=4724.0)
        near = make_zone(direction="SELL", top=4710.0, bottom=4704.0)
        # Current ask 4700, SELL nearest entry is bottom of near = 4704.
        result = _pick_closest_zone(
            direction="SELL",
            zones=[far, near],
            bid=4699.9, ask=4700.0, now=NOW,
            max_distance_points=50.0, max_age_days=7,
        )
        assert result is near


# --------------------------------------------------------------------------- #
# NextSignalWriter.write — end-to-end with the mocked Supabase
# --------------------------------------------------------------------------- #


class TestWrite:
    def test_no_zones_deactivates_both_directions(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        mock_supabase.get_zones_by_status.return_value = []
        df = make_df()

        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)

        assert outcome == {"BUY": "deactivated", "SELL": "deactivated"}
        assert mock_supabase.deactivate_signals_for_direction.call_count == 2
        mock_supabase.upsert_signal_for_direction.assert_not_called()

    def test_one_buy_zone_writes_buy_signal(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        z = make_zone(
            direction="BUY", top=4690.0, bottom=4684.0,
            pattern_type="RBR",
        )
        mock_supabase.get_zones_by_status.return_value = [z]
        df = make_df()

        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)

        assert outcome["BUY"] == "wrote"
        assert outcome["SELL"] == "deactivated"
        mock_supabase.upsert_signal_for_direction.assert_called_once()
        signal: SignalInput = (
            mock_supabase.upsert_signal_for_direction.call_args.args[0]
        )
        assert signal.direction == "BUY"
        assert signal.zone_id == z.id
        assert float(signal.entry_price) == 4690.0  # zone.top for BUY
        # SL = zone_bottom - 17.5 = 4684 - 17.5 = 4666.5
        assert float(signal.sl_price) == pytest.approx(4666.5)
        # current_price = bid for BUY
        assert float(signal.current_price) == 4700.0
        # distance = bid - entry = 4700 - 4690 = 10
        assert float(signal.distance_dollars) == pytest.approx(10.0)
        assert signal.pattern_type == "RBR"
        assert signal.zone_status == "CONFIRMED"
        # TPs computed from local highs in the synthetic df.
        assert signal.tp1_price is not None

    def test_both_directions_processed(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        buy = make_zone(direction="BUY", top=4690.0, bottom=4684.0)
        sell = make_zone(direction="SELL", top=4716.0, bottom=4710.0)
        mock_supabase.get_zones_by_status.return_value = [buy, sell]
        df = make_df()

        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)

        assert outcome == {"BUY": "wrote", "SELL": "wrote"}
        assert mock_supabase.upsert_signal_for_direction.call_count == 2
        directions = {
            c.args[0].direction
            for c in mock_supabase.upsert_signal_for_direction.call_args_list
        }
        assert directions == {"BUY", "SELL"}

    def test_sell_signal_uses_ask_and_top_buffer(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        sell = make_zone(direction="SELL", top=4716.0, bottom=4710.0)
        mock_supabase.get_zones_by_status.return_value = [sell]
        df = make_df()

        writer.write(df, bid=4699.9, ask=4700.0, now=NOW)

        signal: SignalInput = (
            mock_supabase.upsert_signal_for_direction.call_args.args[0]
        )
        # entry = zone.bottom for SELL
        assert float(signal.entry_price) == 4710.0
        # SL = zone_top + 17.5 = 4716 + 17.5 = 4733.5
        assert float(signal.sl_price) == pytest.approx(4733.5)
        # current_price = ask for SELL
        assert float(signal.current_price) == 4700.0
        # distance = entry - ask = 4710 - 4700 = 10
        assert float(signal.distance_dollars) == pytest.approx(10.0)

    def test_zones_too_far_deactivate(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        # +60 away (outside the 50pt distance filter).
        far = make_zone(direction="BUY", top=4640.0, bottom=4634.0)
        mock_supabase.get_zones_by_status.return_value = [far]
        df = make_df()

        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        assert outcome["BUY"] == "deactivated"
        mock_supabase.upsert_signal_for_direction.assert_not_called()

    def test_consumed_zones_excluded(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        # Even if Supabase happens to return CONSUMED rows, we filter
        # them out defensively.
        z = make_zone(
            direction="BUY", status="CONSUMED",
            top=4690.0, bottom=4684.0,
        )
        mock_supabase.get_zones_by_status.return_value = [z]
        df = make_df()

        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        assert outcome["BUY"] == "deactivated"

    def test_get_zones_failure_marks_both_error(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        mock_supabase.get_zones_by_status.side_effect = RuntimeError(
            "supabase 500"
        )
        df = make_df()
        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        assert outcome == {"BUY": "error", "SELL": "error"}
        mock_supabase.upsert_signal_for_direction.assert_not_called()
        mock_supabase.deactivate_signals_for_direction.assert_not_called()

    def test_upsert_failure_isolated_to_direction(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        # BUY upsert fails, SELL still has no zones → SELL deactivated.
        buy = make_zone(direction="BUY", top=4690.0, bottom=4684.0)
        mock_supabase.get_zones_by_status.return_value = [buy]
        mock_supabase.upsert_signal_for_direction.side_effect = (
            RuntimeError("upsert failed")
        )
        df = make_df()

        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        assert outcome["BUY"] == "error"
        assert outcome["SELL"] == "deactivated"

    def test_empty_df_skips(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        df = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": []}
        )
        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        assert outcome == {"BUY": "skipped", "SELL": "skipped"}
        mock_supabase.get_zones_by_status.assert_not_called()

    def test_flipped_zone_writes_signal(
        self, writer: NextSignalWriter, mock_supabase: MagicMock,
    ) -> None:
        # Original SELL zone now flipped to BUY direction.
        flipped = make_zone(
            direction="SELL", status="FLIPPED",
            flipped_direction="BUY",
            top=4690.0, bottom=4684.0,
        )
        mock_supabase.get_zones_by_status.return_value = [flipped]
        df = make_df()

        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        assert outcome["BUY"] == "wrote"
        signal: SignalInput = (
            mock_supabase.upsert_signal_for_direction.call_args.args[0]
        )
        assert signal.direction == "BUY"
        assert signal.zone_status == "FLIPPED"


# --------------------------------------------------------------------------- #
# Custom config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_custom_distance_filter(
        self, mock_supabase: MagicMock,
    ) -> None:
        # Tighten distance to 5 → near zone (10 away) gets excluded.
        writer = NextSignalWriter(
            mock_supabase,
            config=NextSignalConfig(max_distance_points=5.0),
        )
        z = make_zone(direction="BUY", top=4690.0, bottom=4684.0)
        mock_supabase.get_zones_by_status.return_value = [z]
        df = make_df()
        outcome = writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        assert outcome["BUY"] == "deactivated"

    def test_custom_sl_buffer(
        self, mock_supabase: MagicMock,
    ) -> None:
        writer = NextSignalWriter(
            mock_supabase,
            config=NextSignalConfig(sl_buffer_points=10.0),
        )
        z = make_zone(direction="BUY", top=4690.0, bottom=4684.0)
        mock_supabase.get_zones_by_status.return_value = [z]
        df = make_df()
        writer.write(df, bid=4700.0, ask=4700.1, now=NOW)
        signal: SignalInput = (
            mock_supabase.upsert_signal_for_direction.call_args.args[0]
        )
        # SL = zone_bottom - 10.0 = 4684 - 10 = 4674
        assert float(signal.sl_price) == pytest.approx(4674.0)
