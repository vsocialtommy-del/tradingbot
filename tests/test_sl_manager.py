"""Tests for ``bot.exits.sl_manager``.

PR #31 refactor: ``calculate_initial_sl(zone, anchor_swing)`` —
caller passes the anchor swing (Strong Point picks it). No more
lookback heuristic inside sl_manager. ``validate_sl_distance`` and
``apply_sl_to_setup`` are unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.execution.mt5_connector import MT5Connector
from bot.exits.sl_manager import (
    SLCalculation,
    SLManager,
    SLManagerConfig,
    SLValidation,
)
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade
from bot.strategy.pattern_detection import (
    Base,
    Impulse,
    Pattern,
    PatternType,
)
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_setup(
    *,
    id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
) -> Setup:
    return Setup(
        id=id or uuid4(),
        zone_id=uuid4(),
        direction=direction,  # type: ignore[arg-type]
        entry_mode="STRONG_POINT_FIRST_TOUCH",
        planned_layer1_price=Decimal("1900"),
        planned_layer2_price=Decimal("1897.5"),
        planned_layer3_price=Decimal("1895"),
        planned_sl_price=Decimal("1880"),
        planned_tp1_price=Decimal("1907"),
        status=status,  # type: ignore[arg-type]
        skip_reason=None,
        activated_at=NOW,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_trade(
    *,
    id: UUID | None = None,
    setup_id: UUID | None = None,
    layer_number: int = 1,
    status: str = "FILLED",
    mt5_ticket: int | None = 11111,
) -> Trade:
    return Trade(
        id=id or uuid4(),
        setup_id=setup_id or uuid4(),
        layer_number=layer_number,
        direction="BUY",
        order_type="MARKET",
        mt5_ticket=mt5_ticket,
        entry_price=Decimal("1900.0"),
        exit_price=None,
        lot_size=Decimal("0.01"),
        sl_price=Decimal("1880.0"),
        tp_price=None,
        status=status,  # type: ignore[arg-type]
        pnl=None,
        commission=Decimal("0"),
        swap=Decimal("0"),
        close_reason=None,
        filled_at=NOW,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_validated_zone(
    *,
    direction: str = "BUY",
    top: float = 1900.0,
    bottom: float = 1895.0,
) -> ValidatedZone:
    """Minimal ValidatedZone — only direction is read by sl_manager."""
    ts = pd.Timestamp("2026-05-08T12:00:00Z")
    impulse = Impulse(
        direction="RALLY", start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=5.0, largest_body=5.0, candle_count=1,
    )
    base = Base(
        start_index=1, end_index=1, candle_count=1,
        top=top, bottom=bottom, range_size=top - bottom, largest_body=0.5,
    )
    pattern = Pattern(
        pattern_type=PatternType.RBR,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction=direction,  # type: ignore[arg-type]
        formed_at=ts,
    )
    zone = Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
    )
    refined = RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=True, rejection_reason=None, original_zone=zone,
    )
    return ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        refined_zone=refined,
        is_strong_point=True, validation_failures=[],
        broken_swing=None, broken_at=None, sl_anchor_swing=None,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=MT5Connector)


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.get_trades_for_setup.return_value = []
    return m


@pytest.fixture
def manager(
    mock_mt5: MagicMock, mock_supabase: MagicMock,
) -> SLManager:
    return SLManager(mt5=mock_mt5, supabase=mock_supabase)


# --------------------------------------------------------------------------- #
# calculate_initial_sl
# --------------------------------------------------------------------------- #


class TestCalculateInitialSL:
    def test_buy_applies_buffer_below_anchor_low(
        self, manager: SLManager,
    ) -> None:
        zone = make_validated_zone(direction="BUY")
        anchor = Swing(
            index=5, time=pd.Timestamp("2026-05-08T11:00:00Z"),
            price=1882.0, kind="LOW",
        )
        result = manager.calculate_initial_sl(zone, anchor)
        # BUY: SL = anchor.price - sl_buffer_points = 1882 - 17.5 = 1864.5
        assert result.sl_price == 1864.5
        assert result.reference_swing_price == 1882.0
        assert result.buffer_used == 17.5
        assert result.direction == "BUY"
        assert result.fallback_used is False

    def test_sell_applies_buffer_above_anchor_high(
        self, manager: SLManager,
    ) -> None:
        zone = make_validated_zone(direction="SELL")
        anchor = Swing(
            index=5, time=pd.Timestamp("2026-05-08T11:00:00Z"),
            price=1918.0, kind="HIGH",
        )
        result = manager.calculate_initial_sl(zone, anchor)
        # SELL: SL = anchor.price + sl_buffer_points = 1918 + 17.5 = 1935.5
        assert result.sl_price == 1935.5
        assert result.reference_swing_price == 1918.0

    def test_buy_with_high_anchor_rejected(
        self, manager: SLManager,
    ) -> None:
        # BUY zones need a LOW anchor; HIGH is wrong-side.
        zone = make_validated_zone(direction="BUY")
        anchor = Swing(
            index=5, time=pd.Timestamp("2026-05-08T11:00:00Z"),
            price=1918.0, kind="HIGH",
        )
        with pytest.raises(ValueError, match="BUY zone needs a LOW"):
            manager.calculate_initial_sl(zone, anchor)

    def test_sell_with_low_anchor_rejected(
        self, manager: SLManager,
    ) -> None:
        zone = make_validated_zone(direction="SELL")
        anchor = Swing(
            index=5, time=pd.Timestamp("2026-05-08T11:00:00Z"),
            price=1882.0, kind="LOW",
        )
        with pytest.raises(ValueError, match="SELL zone needs a HIGH"):
            manager.calculate_initial_sl(zone, anchor)

    def test_custom_buffer_via_config(
        self, mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        mgr = SLManager(
            mt5=mock_mt5, supabase=mock_supabase,
            config=SLManagerConfig(sl_buffer_points=10.0),
        )
        zone = make_validated_zone(direction="BUY")
        anchor = Swing(
            index=5, time=pd.Timestamp("2026-05-08T11:00:00Z"),
            price=1900.0, kind="LOW",
        )
        result = mgr.calculate_initial_sl(zone, anchor)
        assert result.buffer_used == 10.0
        assert result.sl_price == 1890.0


# --------------------------------------------------------------------------- #
# validate_sl_distance
# --------------------------------------------------------------------------- #


class TestValidateSLDistance:
    def test_buy_sl_below_entry_within_band_valid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4520.0, sl_price=4500.0, direction="BUY",
        )
        assert result.is_valid is True
        assert result.distance_points == pytest.approx(20.0)
        assert result.is_too_close is False
        assert result.is_too_far is False
        assert result.error is None

    def test_sell_sl_above_entry_within_band_valid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4480.0, sl_price=4500.0, direction="SELL",
        )
        assert result.is_valid is True
        assert result.distance_points == pytest.approx(20.0)

    def test_buy_sl_at_or_above_entry_invalid(
        self, manager: SLManager,
    ) -> None:
        equal = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4500.0, direction="BUY",
        )
        assert equal.is_valid is False
        assert equal.error is not None
        assert "must be below" in equal.error

        above = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4510.0, direction="BUY",
        )
        assert above.is_valid is False

    def test_sell_sl_at_or_below_entry_invalid(
        self, manager: SLManager,
    ) -> None:
        equal = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4500.0, direction="SELL",
        )
        assert equal.is_valid is False
        assert "must be above" in (equal.error or "")

        below = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4490.0, direction="SELL",
        )
        assert below.is_valid is False

    def test_too_close_flagged(self, manager: SLManager) -> None:
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4497.0, direction="BUY",
        )
        assert result.is_valid is False
        assert result.is_too_close is True
        assert result.distance_points == pytest.approx(3.0)

    def test_too_far_flagged(self, manager: SLManager) -> None:
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4250.0, direction="BUY",
        )
        assert result.is_valid is False
        assert result.is_too_far is True
        assert result.distance_points == pytest.approx(250.0)

    def test_boundary_at_min_distance_valid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4505.0, sl_price=4500.0, direction="BUY",
        )
        assert result.distance_points == pytest.approx(5.0)
        assert result.is_valid is True

    def test_boundary_at_max_distance_valid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4300.0, direction="BUY",
        )
        assert result.distance_points == pytest.approx(200.0)
        assert result.is_valid is True

    def test_unknown_direction_invalid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4490.0, direction="X",
        )
        assert result.is_valid is False
        assert "unknown direction" in (result.error or "")

    def test_custom_min_max_via_config(
        self, mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        mgr = SLManager(
            mt5=mock_mt5, supabase=mock_supabase,
            config=SLManagerConfig(
                min_sl_distance_points=10.0,
                max_sl_distance_points=50.0,
            ),
        )
        r = mgr.validate_sl_distance(4500.0, 4492.0, "BUY")
        assert r.is_too_close is True
        r = mgr.validate_sl_distance(4500.0, 4440.0, "BUY")
        assert r.is_too_far is True


# --------------------------------------------------------------------------- #
# apply_sl_to_setup
# --------------------------------------------------------------------------- #


class TestApplySLToSetup:
    def test_three_open_positions_all_succeed_returns_true(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=1, mt5_ticket=11111),
            make_trade(setup_id=setup.id, layer_number=2, mt5_ticket=22222),
            make_trade(setup_id=setup.id, layer_number=3, mt5_ticket=33333),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is True
        assert mock_mt5.modify_order.call_count == 3
        for call in mock_mt5.modify_order.call_args_list:
            assert call.kwargs["sl"] == 4485.0
        assert mock_supabase.update_trade.call_count == 3

    def test_one_of_three_fails_returns_false_others_still_attempted(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=1, mt5_ticket=11111),
            make_trade(setup_id=setup.id, layer_number=2, mt5_ticket=22222),
            make_trade(setup_id=setup.id, layer_number=3, mt5_ticket=33333),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.modify_order.side_effect = [None, RuntimeError("busy"), None]

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is False
        assert mock_mt5.modify_order.call_count == 3
        # CRITICAL log written for the failure.
        log_levels = [c.kwargs.get("level") for c in mock_supabase.log_event.call_args_list]
        assert "ERROR" in log_levels

    def test_no_open_positions_returns_true(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup()
        mock_supabase.get_trades_for_setup.return_value = []
        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)
        assert ok is True
        mock_mt5.modify_order.assert_not_called()

    def test_terminal_trades_skipped(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=1, status="CLOSED",
                       mt5_ticket=11111),
            make_trade(setup_id=setup.id, layer_number=2, status="FILLED",
                       mt5_ticket=22222),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        manager.apply_sl_to_setup(setup, sl_price=4485.0)
        # Only the FILLED trade gets a modify.
        assert mock_mt5.modify_order.call_count == 1


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


class TestDataclassDefaults:
    def test_sl_manager_config_defaults(self) -> None:
        c = SLManagerConfig()
        assert c.symbol == "XAUUSD"
        assert c.sl_buffer_points == 17.5
        assert c.min_sl_distance_points == 5.0
        assert c.max_sl_distance_points == 200.0

    def test_sl_validation_default_error_is_none(self) -> None:
        v = SLValidation(
            is_valid=True, distance_points=20.0,
            is_too_close=False, is_too_far=False,
        )
        assert v.error is None

    def test_sl_calculation_fields(self) -> None:
        c = SLCalculation(
            sl_price=4482.5, reference_swing_price=4500.0,
            buffer_used=17.5, lookback_used=0, direction="BUY",
        )
        assert c.fallback_used is False
