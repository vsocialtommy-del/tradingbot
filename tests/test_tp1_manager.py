"""Tests for ``bot.exits.tp1_manager``.

Same pattern as test_entry_trigger / test_position_tracker:
``MagicMock(spec=...)`` for the dependencies, helper builders for the
typed Setup/Trade fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pytest_mock import MockerFixture

from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import PositionTracker
from bot.exits.tp1_manager import (
    TP1Manager,
    TP1ManagerConfig,
    TP1Result,
    _compute_be_lot_weighted,
    _round_down_to_step,
    _trigger_met,
)
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
    planned_tp1_price: float = 1904.0,
    planned_sl_price: float = 1880.0,
    activated_at: datetime | None = NOW,
) -> Setup:
    return Setup(
        id=id or uuid4(),
        zone_id=uuid4(),
        direction=direction,  # type: ignore[arg-type]
        entry_mode="STRONG_POINT_FIRST_TOUCH",
        planned_layer1_price=Decimal("1900"),
        planned_layer2_price=Decimal("1897.5"),
        planned_layer3_price=Decimal("1895"),
        planned_sl_price=Decimal(str(planned_sl_price)),
        planned_tp1_price=Decimal(str(planned_tp1_price)),
        status=status,  # type: ignore[arg-type]
        skip_reason=None,
        activated_at=activated_at,
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
    direction: str = "BUY",
    mt5_ticket: int | None = 11111,
    entry_price: float | None = 1900.0,
    lot_size: float = 0.01,
    sl_price: float = 1880.0,
) -> Trade:
    return Trade(
        id=id or uuid4(),
        setup_id=setup_id or uuid4(),
        layer_number=layer_number,
        direction=direction,  # type: ignore[arg-type]
        order_type="MARKET",
        mt5_ticket=mt5_ticket,
        entry_price=Decimal(str(entry_price)) if entry_price is not None else None,
        exit_price=None,
        lot_size=Decimal(str(lot_size)),
        sl_price=Decimal(str(sl_price)),
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
def mock_tracker(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=PositionTracker)


@pytest.fixture
def manager(
    mock_mt5: MagicMock,
    mock_supabase: MagicMock,
    mock_tracker: MagicMock,
) -> TP1Manager:
    return TP1Manager(
        mt5=mock_mt5,
        supabase=mock_supabase,
        position_tracker=mock_tracker,
    )


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


class TestTriggerMet:
    def test_buy_fires_when_bid_at_or_above_tp1(self) -> None:
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        # Boundary inclusive.
        assert _trigger_met(s, 1904.0, bid=1904.0, ask=1904.1) is True
        # Above trigger.
        assert _trigger_met(s, 1904.0, bid=1904.5, ask=1904.6) is True
        # Below trigger.
        assert _trigger_met(s, 1904.0, bid=1903.9, ask=1904.0) is False

    def test_sell_fires_when_ask_at_or_below_tp1(self) -> None:
        s = make_setup(direction="SELL", planned_tp1_price=1896.0)
        # Boundary inclusive.
        assert _trigger_met(s, 1896.0, bid=1895.9, ask=1896.0) is True
        # Below trigger.
        assert _trigger_met(s, 1896.0, bid=1895.4, ask=1895.5) is True
        # Above trigger.
        assert _trigger_met(s, 1896.0, bid=1896.0, ask=1896.1) is False


class TestRoundDownToStep:
    def test_value_below_step_returns_zero(self) -> None:
        # The headline v1 case: 50% of 0.01 = 0.005, step 0.01 → 0.
        assert _round_down_to_step(0.005, 0.01) == 0.0

    def test_value_equals_step_returns_step(self) -> None:
        assert _round_down_to_step(0.01, 0.01) == 0.01

    def test_value_above_step_floors(self) -> None:
        # 50% of 0.10 = 0.05, step 0.01 → 0.05.
        assert _round_down_to_step(0.05, 0.01) == 0.05
        # 0.07 floored to 0.01 step → 0.07.
        assert _round_down_to_step(0.07, 0.01) == 0.07
        # Inexact float — Decimal-backed floor avoids drift.
        assert _round_down_to_step(0.025, 0.01) == 0.02

    def test_zero_step_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _round_down_to_step(0.05, 0.0)


class TestComputeBELotWeighted:
    def test_three_equal_layers_returns_simple_average(self) -> None:
        trades = [
            make_trade(layer_number=1, entry_price=1900.0, lot_size=0.01),
            make_trade(layer_number=2, entry_price=1897.5, lot_size=0.01),
            make_trade(layer_number=3, entry_price=1895.0, lot_size=0.01),
        ]
        be = _compute_be_lot_weighted(trades)
        # Lot-weighted == arithmetic mean when lots are equal.
        assert be == pytest.approx((1900.0 + 1897.5 + 1895.0) / 3.0)

    def test_lot_weighted_diverges_from_simple_when_lots_unequal(self) -> None:
        # v1.1 forward-compat: layers with different sizes.
        trades = [
            make_trade(layer_number=1, entry_price=1900.0, lot_size=0.01),
            make_trade(layer_number=2, entry_price=1898.0, lot_size=0.03),
        ]
        # Weighted: (1900*0.01 + 1898*0.03) / (0.01+0.03) = 75.94 / 0.04 = 1898.5
        be = _compute_be_lot_weighted(trades)
        assert be == pytest.approx(1898.5)
        # Simple mean would be (1900+1898)/2 = 1899 — different.
        assert be != pytest.approx(1899.0)

    def test_skips_trades_with_no_entry_price(self) -> None:
        # Defensive: a FILLED trade should always have entry_price, but
        # if one doesn't (data corruption), we just skip it.
        trades = [
            make_trade(layer_number=1, entry_price=1900.0, lot_size=0.01),
            make_trade(layer_number=2, entry_price=None, lot_size=0.01),
        ]
        be = _compute_be_lot_weighted(trades)
        assert be == pytest.approx(1900.0)

    def test_empty_or_all_unpriced_raises(self) -> None:
        with pytest.raises(ValueError, match="no filled trades"):
            _compute_be_lot_weighted([])
        with pytest.raises(ValueError, match="no filled trades"):
            _compute_be_lot_weighted([
                make_trade(entry_price=None),
            ])


# --------------------------------------------------------------------------- #
# check() — short-circuit paths
# --------------------------------------------------------------------------- #


class TestCheckShortCircuits:
    def test_setup_already_tp1_hit_returns_not_triggered(
        self, manager: TP1Manager,
        mock_mt5: MagicMock, mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(status="TP1_HIT", planned_tp1_price=1904.0)
        result = manager.check(s, bid=1910.0, ask=1910.1)

        assert result.triggered is False
        assert result.tp1_price == 1904.0
        # Nothing should have been called.
        mock_mt5.close_position.assert_not_called()
        mock_mt5.modify_order.assert_not_called()
        mock_tracker.update_setup_status.assert_not_called()

    def test_non_active_setup_returns_not_triggered(
        self, manager: TP1Manager, mock_mt5: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        # PENDING / SKIPPED / etc. shouldn't drive TP1.
        s = make_setup(status="PENDING")
        result = manager.check(s, bid=1910.0, ask=1910.1)
        assert result.triggered is False
        mock_mt5.close_position.assert_not_called()
        mock_tracker.update_setup_status.assert_not_called()

    def test_buy_price_below_tp1_no_trigger(
        self, manager: TP1Manager, mock_mt5: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        result = manager.check(s, bid=1903.9, ask=1904.0)
        assert result.triggered is False
        assert result.tp1_price == 1904.0
        mock_mt5.close_position.assert_not_called()
        mock_tracker.update_setup_status.assert_not_called()

    def test_sell_price_above_tp1_no_trigger(
        self, manager: TP1Manager, mock_mt5: MagicMock,
    ) -> None:
        s = make_setup(direction="SELL", planned_tp1_price=1896.0)
        result = manager.check(s, bid=1896.0, ask=1896.1)
        assert result.triggered is False
        mock_mt5.close_position.assert_not_called()

    def test_no_filled_trades_returns_error(
        self, manager: TP1Manager,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        # Trigger met but Supabase has no FILLED rows (shouldn't happen
        # for ACTIVE setup, but be defensive).
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(layer_number=1, status="WAITING", mt5_ticket=None),
        ]
        result = manager.check(s, bid=1904.0, ask=1904.1)
        assert result.triggered is False
        assert result.error == "no_filled_trades"
        mock_tracker.update_setup_status.assert_not_called()


# --------------------------------------------------------------------------- #
# check() — happy paths (Option B vs partial-close)
# --------------------------------------------------------------------------- #


class TestCheckOptionB:
    """v1 — fixed 0.01 lots — every TP1 takes the lot-rounding skip."""

    def test_three_filled_layers_at_min_lot_skip_partial_move_be(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        trades = [
            make_trade(
                setup_id=s.id, layer_number=1,
                entry_price=1900.0, lot_size=0.01, mt5_ticket=11111,
            ),
            make_trade(
                setup_id=s.id, layer_number=2,
                entry_price=1897.5, lot_size=0.01, mt5_ticket=22222,
            ),
            make_trade(
                setup_id=s.id, layer_number=3,
                entry_price=1895.0, lot_size=0.01, mt5_ticket=33333,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        result = manager.check(s, bid=1904.0, ask=1904.1)

        assert result.triggered is True
        assert result.tp1_price == 1904.0
        # Option B: no partial closes ran.
        assert result.closed_lots == 0.0
        mock_mt5.close_position.assert_not_called()
        # BE is the lot-weighted avg (== arithmetic mean for equal lots).
        expected_be = (1900.0 + 1897.5 + 1895.0) / 3.0
        assert result.new_sl_price == pytest.approx(expected_be)
        # SL moved on every layer.
        assert mock_mt5.modify_order.call_count == 3
        for call in mock_mt5.modify_order.call_args_list:
            assert call.kwargs["sl"] == pytest.approx(expected_be)
        # Setup → TP1_HIT.
        mock_tracker.update_setup_status.assert_called_once_with(s.id, "TP1_HIT")
        assert result.error is None
        assert result.sl_modify_pending is False

    def test_only_layer_1_filled_skip_partial_move_be(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        # Layers 2/3 still WAITING; only Layer 1 filled.
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        trades = [
            make_trade(
                setup_id=s.id, layer_number=1, status="FILLED",
                entry_price=1900.0, lot_size=0.01, mt5_ticket=11111,
            ),
            make_trade(
                setup_id=s.id, layer_number=2, status="WAITING",
                entry_price=None, lot_size=0.01, mt5_ticket=None,
            ),
            make_trade(
                setup_id=s.id, layer_number=3, status="WAITING",
                entry_price=None, lot_size=0.01, mt5_ticket=None,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        result = manager.check(s, bid=1904.0, ask=1904.1)

        assert result.triggered is True
        assert result.closed_lots == 0.0
        assert result.new_sl_price == pytest.approx(1900.0)  # only Layer 1
        # Only one filled layer → only one SL modify.
        assert mock_mt5.modify_order.call_count == 1
        assert mock_mt5.close_position.call_count == 0
        mock_tracker.update_setup_status.assert_called_once_with(s.id, "TP1_HIT")


class TestCheckPartialCloseExecutes:
    """v1.1-style — larger lots make 50 % round above the step."""

    def test_three_filled_layers_at_010_each_partial_closes_50pct(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        trades = [
            make_trade(
                setup_id=s.id, layer_number=1,
                entry_price=1900.0, lot_size=0.10, mt5_ticket=11111,
            ),
            make_trade(
                setup_id=s.id, layer_number=2,
                entry_price=1897.5, lot_size=0.10, mt5_ticket=22222,
            ),
            make_trade(
                setup_id=s.id, layer_number=3,
                entry_price=1895.0, lot_size=0.10, mt5_ticket=33333,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        result = manager.check(s, bid=1904.0, ask=1904.1)

        assert result.triggered is True
        # 50% of 0.10 each = 0.05 each, total 0.15.
        assert result.closed_lots == pytest.approx(0.15)
        assert mock_mt5.close_position.call_count == 3
        for call in mock_mt5.close_position.call_args_list:
            assert call.kwargs["partial_lots"] == pytest.approx(0.05)
        # SL → BE on all three remaining.
        assert mock_mt5.modify_order.call_count == 3
        # Three trades transitioned FILLED → PARTIALLY_CLOSED with reason TP1.
        partial_close_calls = [
            c for c in mock_tracker.update_trade_status.call_args_list
            if c.kwargs.get("close_reason") == "TP1"
        ]
        assert len(partial_close_calls) == 3
        for c in partial_close_calls:
            assert c.args[1] == "PARTIALLY_CLOSED"


# --------------------------------------------------------------------------- #
# SELL direction mirror
# --------------------------------------------------------------------------- #


class TestSellDirection:
    def test_sell_setup_uses_ask_for_trigger_and_be_with_filled_entries(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(direction="SELL", planned_tp1_price=1896.0)
        trades = [
            make_trade(
                setup_id=s.id, layer_number=1, direction="SELL",
                entry_price=1900.0, lot_size=0.10, mt5_ticket=11111,
            ),
            make_trade(
                setup_id=s.id, layer_number=2, direction="SELL",
                entry_price=1902.5, lot_size=0.10, mt5_ticket=22222,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        # Ask at 1896.0 → exact boundary triggers.
        result = manager.check(s, bid=1895.9, ask=1896.0)

        assert result.triggered is True
        assert result.new_sl_price == pytest.approx((1900.0 + 1902.5) / 2.0)
        # Two partial closes.
        assert mock_mt5.close_position.call_count == 2
        # Setup → TP1_HIT.
        mock_tracker.update_setup_status.assert_called_once_with(s.id, "TP1_HIT")


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestPartialCloseFailures:
    def test_close_position_fails_on_one_layer_others_proceed(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        trades = [
            make_trade(
                setup_id=s.id, layer_number=1,
                entry_price=1900.0, lot_size=0.10, mt5_ticket=11111,
            ),
            make_trade(
                setup_id=s.id, layer_number=2,
                entry_price=1897.5, lot_size=0.10, mt5_ticket=22222,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        # First close succeeds, second fails.
        mock_mt5.close_position.side_effect = [None, RuntimeError("requote")]

        result = manager.check(s, bid=1904.0, ask=1904.1)

        assert result.triggered is True
        # Only Layer 1's 0.05 closed.
        assert result.closed_lots == pytest.approx(0.05)
        assert "close_position failed" in (result.error or "")
        # Both SL moves should still attempt — runner protection is critical.
        assert mock_mt5.modify_order.call_count == 2
        # Setup → TP1_HIT regardless.
        mock_tracker.update_setup_status.assert_called_once_with(s.id, "TP1_HIT")


class TestBEMoveFailures:
    def test_modify_order_fails_setup_still_tp1_hit_pending_flag_set(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        # Decision: BE-move failure → critical alert, retry on next loop,
        # do NOT close runner. Setup still transitions to TP1_HIT.
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        trades = [
            make_trade(
                setup_id=s.id, layer_number=1,
                entry_price=1900.0, lot_size=0.01, mt5_ticket=11111,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.modify_order.side_effect = RuntimeError("broker busy")

        result = manager.check(s, bid=1904.0, ask=1904.1)

        assert result.triggered is True
        assert result.sl_modify_pending is True
        assert "modify_order" in (result.error or "")
        # Critically: setup STILL transitioned (partial-close stage finished).
        mock_tracker.update_setup_status.assert_called_once_with(s.id, "TP1_HIT")
        # And we did NOT close the runner.
        mock_mt5.close_position.assert_not_called()
        # Critical alert logged.
        assert any(
            c.args[0] == "ERROR"
            for c in mock_supabase.log_event.call_args_list
        )

    def test_partial_modify_failure_only_failed_layer_flagged(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        # Two filled layers. SL move on layer 1 succeeds, layer 2 fails.
        # sl_modify_pending should be True so main loop retries.
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        trades = [
            make_trade(
                setup_id=s.id, layer_number=1,
                entry_price=1900.0, lot_size=0.01, mt5_ticket=11111,
            ),
            make_trade(
                setup_id=s.id, layer_number=2,
                entry_price=1898.0, lot_size=0.01, mt5_ticket=22222,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.modify_order.side_effect = [None, RuntimeError("broker busy")]

        result = manager.check(s, bid=1904.0, ask=1904.1)

        assert result.triggered is True
        assert result.sl_modify_pending is True
        mock_tracker.update_setup_status.assert_called_once_with(s.id, "TP1_HIT")


# --------------------------------------------------------------------------- #
# Cascade — TP1Manager delegates to position_tracker.update_setup_status
# --------------------------------------------------------------------------- #


class TestCascadeWiring:
    """The actual cascade lives in PositionTracker. This module's job
    is to call ``update_setup_status(... 'TP1_HIT')`` so the cascade
    fires. Verifying the call is enough here; the cascade behaviour
    itself is covered in test_position_tracker."""

    def test_check_triggers_position_tracker_update(
        self, manager: TP1Manager, mock_mt5: MagicMock,
        mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(direction="BUY", planned_tp1_price=1904.0)
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, layer_number=1,
                entry_price=1900.0, lot_size=0.01, mt5_ticket=11111,
            ),
        ]

        manager.check(s, bid=1904.0, ask=1904.1)

        # Single call with TP1_HIT — the tracker handles the cascade.
        mock_tracker.update_setup_status.assert_called_once_with(s.id, "TP1_HIT")


# --------------------------------------------------------------------------- #
# Result dataclass shape
# --------------------------------------------------------------------------- #


class TestResultShape:
    def test_default_result_is_not_triggered(self) -> None:
        r = TP1Result(triggered=False)
        assert r.tp1_price == 0.0
        assert r.closed_lots == 0.0
        assert r.new_sl_price == 0.0
        assert r.error is None
        assert r.sl_modify_pending is False

    def test_config_defaults_match_xauusd_vantage(self) -> None:
        c = TP1ManagerConfig()
        assert c.symbol == "XAUUSD"
        assert c.lot_step == 0.01
        assert c.comment_prefix == "bot:tp1"
