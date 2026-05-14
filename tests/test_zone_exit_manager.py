"""Tests for ``bot.exits.zone_exit_manager`` — body-close BE trigger.

PR #47: when the just-closed M5 bar's body closes past the trade's L1
entry in the profit direction, fire a zone-exit action:

* If ≥2 layers FILLED: close shallowest + BE rest + cancel waiting.
* If exactly 1 layer FILLED: BE only (don't close).
* Idempotent: re-firing on a setup whose SL is already at BE is a no-op.

Covers:
* Trigger condition (BUY close > L1; SELL close < L1; inclusive boundary).
* Wick alone (high above L1 but close inside zone) does NOT fire.
* Two-or-more-filled branch: closes L1 + BE on L2/L3 + cancels waiting.
* One-filled branch: BE only on that layer, no close, no cancel.
* No filled layers (all waiting): cancels waiting layers, no close.
* Idempotency: shallowest layer's SL already at entry → no-op.
* Pending setup (status != ACTIVE) → no-op.
* Broker failures don't break the rest of the pass.
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
from bot.exits.zone_exit_manager import ZoneExitManager
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    setup_id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
    # BUY: L1 = zone.top (4694), L3 = zone.bottom (4689) typically
    # SELL: L1 = zone.bottom, L3 = zone.top
    planned_layer1_price: Decimal = Decimal("4694.00"),
    planned_layer2_price: Decimal = Decimal("4691.50"),
    planned_layer3_price: Decimal = Decimal("4689.00"),
    planned_sl_price: Decimal = Decimal("4671.50"),
    planned_tp1_price: Decimal = Decimal("4708.00"),
    planned_tp2_price: Decimal | None = Decimal("4720.00"),
    planned_tp3_price: Decimal | None = Decimal("4735.00"),
) -> Setup:
    return Setup(
        id=setup_id or uuid4(),
        zone_id=uuid4(),
        direction=direction,  # type: ignore[arg-type]
        entry_mode="STRONG_POINT_FIRST_TOUCH",
        planned_layer1_price=planned_layer1_price,
        planned_layer2_price=planned_layer2_price,
        planned_layer3_price=planned_layer3_price,
        planned_sl_price=planned_sl_price,
        planned_tp1_price=planned_tp1_price,
        planned_tp2_price=planned_tp2_price,
        planned_tp3_price=planned_tp3_price,
        status=status,  # type: ignore[arg-type]
        skip_reason=None,
        activated_at=NOW,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_trade(
    *,
    setup_id: UUID,
    layer_number: int,
    status: str = "FILLED",
    mt5_ticket: int | None = 11111,
    entry_price: Decimal | None = Decimal("4694.00"),
    sl_price: Decimal = Decimal("4671.50"),
    direction: str = "BUY",
) -> Trade:
    # WAITING layers don't have a ticket yet; default to None for that.
    if status == "WAITING" and mt5_ticket == 11111:
        mt5_ticket = None
    return Trade(
        id=uuid4(),
        setup_id=setup_id,
        layer_number=layer_number,
        direction=direction,  # type: ignore[arg-type]
        order_type="MARKET" if layer_number == 1 else "LIMIT",
        mt5_ticket=mt5_ticket,
        entry_price=entry_price if status == "FILLED" else None,
        exit_price=None,
        lot_size=Decimal("0.01"),
        sl_price=sl_price,
        tp_price=None,
        status=status,  # type: ignore[arg-type]
        pnl=None,
        commission=Decimal("0"),
        swap=Decimal("0"),
        close_reason=None,
        filled_at=NOW if status == "FILLED" else None,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=MT5Connector)


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=SupabaseLogger)


@pytest.fixture
def mock_tracker(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=PositionTracker)


@pytest.fixture
def manager(
    mock_mt5: MagicMock,
    mock_supabase: MagicMock,
    mock_tracker: MagicMock,
) -> ZoneExitManager:
    return ZoneExitManager(mock_mt5, mock_supabase, mock_tracker)


# --------------------------------------------------------------------------- #
# Trigger condition
# --------------------------------------------------------------------------- #


class TestTriggerCondition:
    def test_buy_fires_when_close_above_l1(
        self, manager: ZoneExitManager, mock_supabase: MagicMock,
    ) -> None:
        # BUY zone with L1=4694 (zone.top). Close at 4695 > 4694 → fire.
        s = make_setup(direction="BUY", planned_layer1_price=Decimal("4694.00"))
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, layer_number=1,
                entry_price=Decimal("4694.00"),
                sl_price=Decimal("4671.50"),
            ),
            make_trade(
                setup_id=s.id, layer_number=2,
                entry_price=Decimal("4691.50"),
                sl_price=Decimal("4671.50"),
            ),
        ]
        result = manager.check(s, last_close=4695.0, bid=4695.0, ask=4695.1)
        assert result is not None
        assert result.closed_layer == 1
        assert result.be_layer_count == 1  # layer 2

    def test_buy_does_not_fire_at_exact_l1(
        self, manager: ZoneExitManager, mock_supabase: MagicMock,
    ) -> None:
        # BUY close == L1 is NOT "out" — strict inequality.
        s = make_setup(direction="BUY", planned_layer1_price=Decimal("4694.00"))
        result = manager.check(s, last_close=4694.0, bid=4694.0, ask=4694.1)
        assert result is None
        mock_supabase.get_trades_for_setup.assert_not_called()

    def test_buy_does_not_fire_when_close_inside_zone(
        self, manager: ZoneExitManager, mock_supabase: MagicMock,
    ) -> None:
        # BUY close below L1 (still inside zone or below it) → no fire.
        s = make_setup(direction="BUY", planned_layer1_price=Decimal("4694.00"))
        result = manager.check(s, last_close=4691.0, bid=4691.0, ask=4691.1)
        assert result is None
        mock_supabase.get_trades_for_setup.assert_not_called()

    def test_sell_fires_when_close_below_l1(
        self, manager: ZoneExitManager, mock_supabase: MagicMock,
    ) -> None:
        # SELL zone with L1=4689 (zone.bottom). Close at 4688 < 4689 → fire.
        s = make_setup(
            direction="SELL",
            planned_layer1_price=Decimal("4689.00"),
            planned_layer3_price=Decimal("4694.00"),
            planned_sl_price=Decimal("4711.50"),
            planned_tp1_price=Decimal("4670.00"),
            planned_tp2_price=Decimal("4660.00"),
            planned_tp3_price=Decimal("4650.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, layer_number=1, direction="SELL",
                entry_price=Decimal("4689.00"),
                sl_price=Decimal("4711.50"),
            ),
            make_trade(
                setup_id=s.id, layer_number=2, direction="SELL",
                entry_price=Decimal("4691.50"),
                sl_price=Decimal("4711.50"),
            ),
        ]
        result = manager.check(s, last_close=4688.0, bid=4687.9, ask=4688.0)
        assert result is not None
        assert result.closed_layer == 1

    def test_sell_does_not_fire_at_exact_l1(
        self, manager: ZoneExitManager, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup(
            direction="SELL",
            planned_layer1_price=Decimal("4689.00"),
            planned_layer3_price=Decimal("4694.00"),
            planned_sl_price=Decimal("4711.50"),
            planned_tp1_price=Decimal("4670.00"),
            planned_tp2_price=Decimal("4660.00"),
            planned_tp3_price=Decimal("4650.00"),
        )
        result = manager.check(s, last_close=4689.0, bid=4689.0, ask=4689.1)
        assert result is None

    def test_pending_setup_short_circuits(
        self, manager: ZoneExitManager, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup(status="PENDING")
        result = manager.check(s, last_close=4700.0, bid=4700.0, ask=4700.1)
        assert result is None
        mock_supabase.get_trades_for_setup.assert_not_called()


# --------------------------------------------------------------------------- #
# Two-or-more-filled branch: close shallowest + BE rest + cancel waiting
# --------------------------------------------------------------------------- #


class TestTwoOrMoreFilled:
    def test_buy_closes_layer_1_be_layer_2_and_3(
        self, manager: ZoneExitManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(direction="BUY")
        l1 = make_trade(
            setup_id=s.id, layer_number=1, mt5_ticket=11111,
            entry_price=Decimal("4694.00"),
            sl_price=Decimal("4671.50"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2, mt5_ticket=22222,
            entry_price=Decimal("4691.50"),
            sl_price=Decimal("4671.50"),
        )
        l3 = make_trade(
            setup_id=s.id, layer_number=3, mt5_ticket=33333,
            entry_price=Decimal("4689.00"),
            sl_price=Decimal("4671.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2, l3]

        result = manager.check(s, last_close=4696.0, bid=4696.0, ask=4696.1)

        assert result is not None
        assert result.closed_trade_id == l1.id
        assert result.closed_layer == 1
        assert result.be_layer_count == 2
        assert result.cancelled_waiting_count == 0
        assert result.error is None

        # L1 closed at bid (BUY exit).
        mock_mt5.close_position.assert_called_once_with(11111)
        mock_tracker.update_trade_status.assert_any_call(
            l1.id, "CLOSED", close_reason="ZONE_EXIT", exit_price=4696.0,
        )
        # L2 and L3 SL modified to their own entry prices.
        mt5_modify_calls = mock_mt5.modify_order.call_args_list
        modified = {c.args[0]: c.kwargs["sl"] for c in mt5_modify_calls}
        assert modified[22222] == 4691.5
        assert modified[33333] == 4689.0
        # Trade rows updated to reflect new sl_price.
        update_trade_calls = mock_supabase.update_trade.call_args_list
        sl_updates = {
            c.args[0]: float(c.kwargs["sl_price"])
            for c in update_trade_calls
        }
        assert sl_updates[l2.id] == 4691.5
        assert sl_updates[l3.id] == 4689.0

    def test_sell_closes_layer_1_be_layer_2(
        self, manager: ZoneExitManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        s = make_setup(
            direction="SELL",
            planned_layer1_price=Decimal("4689.00"),
            planned_layer2_price=Decimal("4691.50"),
            planned_layer3_price=Decimal("4694.00"),
            planned_sl_price=Decimal("4711.50"),
            planned_tp1_price=Decimal("4670.00"),
            planned_tp2_price=Decimal("4660.00"),
            planned_tp3_price=Decimal("4650.00"),
        )
        l1 = make_trade(
            setup_id=s.id, layer_number=1, direction="SELL",
            mt5_ticket=11111, entry_price=Decimal("4689.00"),
            sl_price=Decimal("4711.50"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2, direction="SELL",
            mt5_ticket=22222, entry_price=Decimal("4691.50"),
            sl_price=Decimal("4711.50"),
        )
        # L3 is WAITING (price never reached zone.top from below).
        l3 = make_trade(
            setup_id=s.id, layer_number=3, direction="SELL",
            status="WAITING",
            sl_price=Decimal("4711.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2, l3]

        result = manager.check(s, last_close=4687.0, bid=4686.9, ask=4687.0)

        assert result is not None
        assert result.closed_layer == 1
        assert result.be_layer_count == 1  # only L2 filled remaining
        assert result.cancelled_waiting_count == 1

        # SELL exit closes at ask (the price to BUY back the short).
        mock_tracker.update_trade_status.assert_any_call(
            l1.id, "CLOSED", close_reason="ZONE_EXIT", exit_price=4687.0,
        )
        # L2 SL → its own entry (4691.50).
        modify_calls = mock_mt5.modify_order.call_args_list
        assert any(
            c.args[0] == 22222 and c.kwargs["sl"] == 4691.5
            for c in modify_calls
        )
        # L3 (WAITING) cancelled with ZONE_EXIT_CANCELLED.
        mock_tracker.update_trade_status.assert_any_call(
            l3.id, "CANCELLED", close_reason="ZONE_EXIT_CANCELLED",
        )


# --------------------------------------------------------------------------- #
# One-filled branch: BE only, no close
# --------------------------------------------------------------------------- #


class TestOnlyOneFilled:
    def test_one_filled_two_waiting_be_only(
        self, manager: ZoneExitManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        # Price only touched zone.top, never went deeper. L1 filled,
        # L2/L3 still WAITING. Zone-exit fires → BE L1 (don't close it).
        s = make_setup(direction="BUY")
        l1 = make_trade(
            setup_id=s.id, layer_number=1, mt5_ticket=11111,
            entry_price=Decimal("4694.00"),
            sl_price=Decimal("4671.50"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2, status="WAITING",
            sl_price=Decimal("4671.50"),
        )
        l3 = make_trade(
            setup_id=s.id, layer_number=3, status="WAITING",
            sl_price=Decimal("4671.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2, l3]

        result = manager.check(s, last_close=4696.0, bid=4696.0, ask=4696.1)

        assert result is not None
        assert result.closed_trade_id is None
        assert result.closed_layer is None
        assert result.be_layer_count == 1  # L1 BE'd, not closed
        assert result.cancelled_waiting_count == 2

        # close_position must NOT be called (L1 stays open at BE).
        mock_mt5.close_position.assert_not_called()
        # L1 SL → its own entry.
        mock_mt5.modify_order.assert_called_once_with(11111, sl=4694.0)
        # Both WAITING layers cancelled.
        cancel_calls = [
            c for c in mock_tracker.update_trade_status.call_args_list
            if c.kwargs.get("close_reason") == "ZONE_EXIT_CANCELLED"
        ]
        assert len(cancel_calls) == 2


# --------------------------------------------------------------------------- #
# No filled (all waiting) — cancel waiting layers
# --------------------------------------------------------------------------- #


class TestNoFilled:
    def test_all_waiting_cancels_them_no_close_no_be(
        self, manager: ZoneExitManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        # Setup is ACTIVE but no layer ever filled (price moved past
        # the zone without entering). Zone-exit triggers; cancel
        # waiting layers, no close, no BE move.
        s = make_setup(direction="BUY")
        l1 = make_trade(
            setup_id=s.id, layer_number=1, status="WAITING",
            sl_price=Decimal("4671.50"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2, status="WAITING",
            sl_price=Decimal("4671.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2]

        result = manager.check(s, last_close=4696.0, bid=4696.0, ask=4696.1)

        assert result is not None
        assert result.closed_trade_id is None
        assert result.be_layer_count == 0
        assert result.cancelled_waiting_count == 2

        mock_mt5.close_position.assert_not_called()
        mock_mt5.modify_order.assert_not_called()


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


class TestIdempotency:
    def test_shallowest_sl_already_at_entry_no_op(
        self, manager: ZoneExitManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        # L1's sl_price already at its entry — BE already done on a
        # previous M5 close. Re-evaluating should be a no-op.
        s = make_setup(direction="BUY")
        l1 = make_trade(
            setup_id=s.id, layer_number=1, mt5_ticket=11111,
            entry_price=Decimal("4694.00"),
            sl_price=Decimal("4694.00"),  # already at entry
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2, mt5_ticket=22222,
            entry_price=Decimal("4691.50"),
            sl_price=Decimal("4691.50"),  # already at entry
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2]

        result = manager.check(s, last_close=4700.0, bid=4700.0, ask=4700.1)

        assert result is None
        mock_mt5.close_position.assert_not_called()
        mock_mt5.modify_order.assert_not_called()
        mock_tracker.update_trade_status.assert_not_called()

    def test_be_tolerance_treats_within_0_01_as_equal(
        self, manager: ZoneExitManager,
        mock_supabase: MagicMock,
    ) -> None:
        # Broker fill at entry 4694.005, SL set at 4694.00 — should
        # still be detected as "already done".
        s = make_setup(direction="BUY")
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            entry_price=Decimal("4694.005"),
            sl_price=Decimal("4694.00"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2,
            entry_price=Decimal("4691.50"),
            sl_price=Decimal("4691.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2]
        result = manager.check(s, last_close=4700.0, bid=4700.0, ask=4700.1)
        assert result is None


# --------------------------------------------------------------------------- #
# Broker-error resilience
# --------------------------------------------------------------------------- #


class TestBrokerFailures:
    def test_close_failure_logged_be_still_proceeds(
        self, manager: ZoneExitManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        # close_position raises → result.error populated, but the BE
        # cascade on the remaining layer still proceeds.
        s = make_setup(direction="BUY")
        l1 = make_trade(
            setup_id=s.id, layer_number=1, mt5_ticket=11111,
            entry_price=Decimal("4694.00"),
            sl_price=Decimal("4671.50"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2, mt5_ticket=22222,
            entry_price=Decimal("4691.50"),
            sl_price=Decimal("4671.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2]
        mock_mt5.close_position.side_effect = RuntimeError("broker down")

        result = manager.check(s, last_close=4696.0, bid=4696.0, ask=4696.1)

        assert result is not None
        assert result.error is not None
        assert "close_position failed" in result.error
        # BE on L2 still attempted.
        mock_mt5.modify_order.assert_called_once_with(22222, sl=4691.5)
