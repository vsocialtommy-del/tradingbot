"""Tests for ``bot.execution.position_tracker``.

Same testing pattern as ``test_order_manager``: ``mocker.MagicMock(spec=…)``
for the dependencies, sequential ``return_value`` / ``side_effect`` for
multi-call methods. ``Setup`` and ``Trade`` pydantic models are
constructed with the ``make_setup`` / ``make_trade`` helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pytest_mock import MockerFixture

from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import (
    ACTIVE_SETUP_STATUSES,
    OPEN_TRADE_STATUSES,
    VALID_SETUP_TRANSITIONS,
    VALID_TRADE_TRANSITIONS,
    ClosedPosition,
    PositionTracker,
    ReconcileResult,
    StateTransitionError,
    _validate_setup_transition,
    _validate_trade_transition,
)
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Helpers — Setup/Trade fixtures
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    id: UUID | None = None,
    status: str = "ACTIVE",
    direction: str = "BUY",
    activated_at: datetime | None = None,
    closed_at: datetime | None = None,
    skip_reason: str | None = None,
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
        planned_tp1_price=Decimal("1904"),
        status=status,  # type: ignore[arg-type]
        skip_reason=skip_reason,
        activated_at=activated_at,
        closed_at=closed_at,
        created_at=NOW,
        updated_at=NOW,
    )


def make_trade(
    *,
    id: UUID | None = None,
    setup_id: UUID | None = None,
    layer_number: int = 1,
    status: str = "PENDING",
    mt5_ticket: int | None = 11111,
    order_type: str = "LIMIT",
    entry_price: Decimal | None = None,
) -> Trade:
    return Trade(
        id=id or uuid4(),
        setup_id=setup_id or uuid4(),
        layer_number=layer_number,
        direction="BUY",
        order_type=order_type,  # type: ignore[arg-type]
        mt5_ticket=mt5_ticket,
        entry_price=entry_price,
        exit_price=None,
        lot_size=Decimal("0.01"),
        sl_price=Decimal("1880"),
        tp_price=None,
        status=status,  # type: ignore[arg-type]
        pnl=None,
        commission=Decimal("0"),
        swap=Decimal("0"),
        close_reason=None,
        filled_at=None,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=MT5Connector)
    m.get_open_positions.return_value = []
    return m


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.get_setups_by_status.return_value = []
    m.get_setup_by_id.return_value = None
    m.get_trade_by_id.return_value = None
    m.get_trades_for_setup.return_value = []
    return m


@pytest.fixture
def tracker(mock_mt5: MagicMock, mock_supabase: MagicMock) -> PositionTracker:
    return PositionTracker(mt5=mock_mt5, supabase=mock_supabase)


# --------------------------------------------------------------------------- #
# State-machine validation (module-level helpers)
# --------------------------------------------------------------------------- #


class TestStateMachineValidation:
    @pytest.mark.parametrize(
        "current,new",
        [
            ("PENDING", "ACTIVE"),
            ("PENDING", "SKIPPED"),
            ("ACTIVE", "TP1_HIT"),
            ("ACTIVE", "STOPPED_OUT"),
            ("ACTIVE", "CLOSED"),
            ("TP1_HIT", "CLOSED"),
        ],
    )
    def test_valid_setup_transitions(self, current: str, new: str) -> None:
        _validate_setup_transition(current, new)  # no raise

    @pytest.mark.parametrize(
        "current,new",
        [
            ("PENDING", "TP1_HIT"),  # skipped ACTIVE
            ("TP1_HIT", "ACTIVE"),    # backwards
            ("CLOSED", "ACTIVE"),     # from terminal
            ("SKIPPED", "ACTIVE"),    # from terminal
            ("STOPPED_OUT", "CLOSED"),  # from terminal
            ("ACTIVE", "PENDING"),    # backwards
            ("ACTIVE", "ACTIVE"),     # self-loop
        ],
    )
    def test_invalid_setup_transitions_raise(
        self, current: str, new: str
    ) -> None:
        with pytest.raises(StateTransitionError):
            _validate_setup_transition(current, new)

    @pytest.mark.parametrize(
        "current,new",
        [
            ("PENDING", "FILLED"),
            ("PENDING", "CANCELLED"),
            ("FILLED", "PARTIALLY_CLOSED"),
            ("FILLED", "CLOSED"),
            ("PARTIALLY_CLOSED", "CLOSED"),
        ],
    )
    def test_valid_trade_transitions(self, current: str, new: str) -> None:
        _validate_trade_transition(current, new)

    @pytest.mark.parametrize(
        "current,new",
        [
            ("FILLED", "PENDING"),       # backwards
            ("CLOSED", "FILLED"),         # from terminal
            ("CANCELLED", "PENDING"),     # from terminal
            ("PARTIALLY_CLOSED", "FILLED"),  # backwards
        ],
    )
    def test_invalid_trade_transitions_raise(
        self, current: str, new: str
    ) -> None:
        with pytest.raises(StateTransitionError):
            _validate_trade_transition(current, new)


class TestStateMachineCompleteness:
    """Sanity: the transition tables cover every defined status."""

    def test_setup_transitions_cover_all_statuses(self) -> None:
        from bot.logging.supabase_logger import SetupStatus
        from typing import get_args
        for status in get_args(SetupStatus):
            assert status in VALID_SETUP_TRANSITIONS, status

    def test_trade_transitions_cover_all_statuses(self) -> None:
        from bot.logging.supabase_logger import TradeStatus
        from typing import get_args
        for status in get_args(TradeStatus):
            assert status in VALID_TRADE_TRANSITIONS, status


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


class TestQueries:
    def test_get_active_setups_filters_by_active_statuses(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        s_pending = make_setup(status="PENDING")
        s_active = make_setup(status="ACTIVE")
        s_tp1 = make_setup(status="TP1_HIT")
        mock_supabase.get_setups_by_status.return_value = [s_pending, s_active, s_tp1]

        result = tracker.get_active_setups()

        assert result == [s_pending, s_active, s_tp1]
        # Verify the right statuses were requested.
        call_arg = mock_supabase.get_setups_by_status.call_args.args[0]
        assert set(call_arg) == ACTIVE_SETUP_STATUSES

    def test_get_setup_by_id_returns_setup(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        s = make_setup()
        mock_supabase.get_setup_by_id.return_value = s
        assert tracker.get_setup_by_id(s.id) == s

    def test_get_setup_by_id_returns_none_when_missing(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        mock_supabase.get_setup_by_id.return_value = None
        assert tracker.get_setup_by_id(uuid4()) is None

    def test_get_trades_for_setup_passes_through(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        setup_id = uuid4()
        trades = [make_trade(layer_number=i) for i in (1, 2, 3)]
        mock_supabase.get_trades_for_setup.return_value = trades
        assert tracker.get_trades_for_setup(setup_id) == trades
        mock_supabase.get_trades_for_setup.assert_called_once_with(setup_id)

    def test_empty_database_returns_empty_lists(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        mock_supabase.get_setups_by_status.return_value = []
        mock_supabase.get_trades_for_setup.return_value = []
        assert tracker.get_active_setups() == []
        assert tracker.get_trades_for_setup(uuid4()) == []


# --------------------------------------------------------------------------- #
# update_setup_status
# --------------------------------------------------------------------------- #


class TestUpdateSetupStatus:
    def test_valid_transition_updates_db(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_setup(status="PENDING")
        updated = make_setup(id=current.id, status="ACTIVE", activated_at=NOW)
        mock_supabase.get_setup_by_id.return_value = current
        mock_supabase.update_setup.return_value = updated

        result = tracker.update_setup_status(current.id, "ACTIVE")

        assert result == updated
        mock_supabase.update_setup.assert_called_once()
        # activated_at should have been set on PENDING → ACTIVE.
        kwargs = mock_supabase.update_setup.call_args.kwargs
        assert kwargs["status"] == "ACTIVE"
        assert "activated_at" in kwargs

    def test_invalid_transition_raises_no_db_write(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_setup(status="CLOSED")  # terminal
        mock_supabase.get_setup_by_id.return_value = current

        with pytest.raises(StateTransitionError):
            tracker.update_setup_status(current.id, "ACTIVE")

        mock_supabase.update_setup.assert_not_called()

    def test_setup_not_found_raises(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        mock_supabase.get_setup_by_id.return_value = None
        with pytest.raises(ValueError, match="not found"):
            tracker.update_setup_status(uuid4(), "ACTIVE")

    def test_pending_to_active_sets_activated_at(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_setup(status="PENDING", activated_at=None)
        mock_supabase.get_setup_by_id.return_value = current
        mock_supabase.update_setup.return_value = current

        tracker.update_setup_status(current.id, "ACTIVE")

        kwargs = mock_supabase.update_setup.call_args.kwargs
        assert "activated_at" in kwargs

    def test_to_closed_sets_closed_at(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_setup(status="ACTIVE")
        mock_supabase.get_setup_by_id.return_value = current
        mock_supabase.update_setup.return_value = current

        tracker.update_setup_status(current.id, "CLOSED")

        kwargs = mock_supabase.update_setup.call_args.kwargs
        assert kwargs["status"] == "CLOSED"
        assert "closed_at" in kwargs

    def test_to_skipped_sets_closed_at_and_skip_reason(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_setup(status="PENDING")
        mock_supabase.get_setup_by_id.return_value = current
        mock_supabase.update_setup.return_value = current

        tracker.update_setup_status(
            current.id, "SKIPPED", skip_reason="GAP_THROUGH_ZONE"
        )

        kwargs = mock_supabase.update_setup.call_args.kwargs
        assert kwargs["skip_reason"] == "GAP_THROUGH_ZONE"
        assert "closed_at" in kwargs

    def test_already_active_no_redundant_activated_at(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        # ACTIVE → CLOSED: don't set activated_at again.
        current = make_setup(status="ACTIVE", activated_at=NOW)
        mock_supabase.get_setup_by_id.return_value = current
        mock_supabase.update_setup.return_value = current

        tracker.update_setup_status(current.id, "CLOSED")

        kwargs = mock_supabase.update_setup.call_args.kwargs
        assert "activated_at" not in kwargs


# --------------------------------------------------------------------------- #
# update_trade_status
# --------------------------------------------------------------------------- #


class TestUpdateTradeStatus:
    def test_pending_to_filled_sets_filled_at(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_trade(status="PENDING")
        mock_supabase.get_trade_by_id.return_value = current
        mock_supabase.update_trade.return_value = current

        tracker.update_trade_status(current.id, "FILLED")

        kwargs = mock_supabase.update_trade.call_args.kwargs
        assert kwargs["status"] == "FILLED"
        assert "filled_at" in kwargs

    def test_filled_to_closed_with_close_reason(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_trade(status="FILLED")
        mock_supabase.get_trade_by_id.return_value = current
        mock_supabase.update_trade.return_value = current

        tracker.update_trade_status(
            current.id, "CLOSED", close_reason="SL_HIT", pnl=-15.5
        )

        kwargs = mock_supabase.update_trade.call_args.kwargs
        assert kwargs["status"] == "CLOSED"
        assert kwargs["close_reason"] == "SL_HIT"
        assert kwargs["pnl"] == -15.5

    def test_invalid_trade_transition_raises(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_trade(status="CLOSED")  # terminal
        mock_supabase.get_trade_by_id.return_value = current

        with pytest.raises(StateTransitionError):
            tracker.update_trade_status(current.id, "FILLED")

        mock_supabase.update_trade.assert_not_called()


# --------------------------------------------------------------------------- #
# detect_filled_layers
# --------------------------------------------------------------------------- #


class TestDetectFilledLayers:
    def test_pending_setup_with_no_fills_returns_empty(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        setup = make_setup(status="PENDING")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="PENDING"),
            make_trade(layer_number=2, mt5_ticket=22222, status="PENDING"),
            make_trade(layer_number=3, mt5_ticket=33333, status="PENDING"),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = []  # no fills yet

        result = tracker.detect_filled_layers(setup)
        assert result == []
        mock_supabase.update_trade.assert_not_called()
        mock_supabase.update_setup.assert_not_called()

    def test_one_layer_filled_transitions_setup_to_active(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        setup = make_setup(status="PENDING")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="PENDING"),
            make_trade(layer_number=2, mt5_ticket=22222, status="PENDING"),
            make_trade(layer_number=3, mt5_ticket=33333, status="PENDING"),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 22222, "price_open": 1897.50},
        ]
        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup

        result = tracker.detect_filled_layers(setup)

        assert result == [2]
        # update_trade called once for layer 2.
        assert mock_supabase.update_trade.call_count == 1
        update_kwargs = mock_supabase.update_trade.call_args.kwargs
        assert update_kwargs["status"] == "FILLED"
        assert update_kwargs["entry_price"] == 1897.50
        # Setup transitioned PENDING → ACTIVE.
        mock_supabase.update_setup.assert_called_once()

    def test_all_three_filled(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        setup = make_setup(status="PENDING")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="PENDING"),
            make_trade(layer_number=2, mt5_ticket=22222, status="PENDING"),
            make_trade(layer_number=3, mt5_ticket=33333, status="PENDING"),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1900.00},
            {"ticket": 22222, "price_open": 1897.50},
            {"ticket": 33333, "price_open": 1895.00},
        ]
        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup

        result = tracker.detect_filled_layers(setup)
        assert result == [1, 2, 3]
        assert mock_supabase.update_trade.call_count == 3

    def test_only_for_pending_setups(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        # ACTIVE setup — should NOT detect fills (already active).
        setup = make_setup(status="ACTIVE")
        result = tracker.detect_filled_layers(setup)
        assert result == []
        mock_supabase.get_trades_for_setup.assert_not_called()

    def test_mt5_failure_returns_empty(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        setup = make_setup(status="PENDING")
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(layer_number=1, mt5_ticket=11111, status="PENDING"),
        ]
        mock_mt5.get_open_positions.side_effect = RuntimeError("network blip")

        result = tracker.detect_filled_layers(setup)
        assert result == []
        mock_supabase.update_trade.assert_not_called()


# --------------------------------------------------------------------------- #
# detect_closed_positions
# --------------------------------------------------------------------------- #


class TestDetectClosedPositions:
    def test_active_setup_no_closures(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        setup = make_setup(status="ACTIVE")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="FILLED"),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1900.00},  # still open
        ]
        result = tracker.detect_closed_positions(setup)
        assert result == []

    def test_one_closed_externally_marked_closed(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        setup = make_setup(status="ACTIVE")
        trade = make_trade(layer_number=1, mt5_ticket=11111, status="FILLED")
        mock_supabase.get_trades_for_setup.return_value = [trade]
        mock_mt5.get_open_positions.return_value = []  # ticket gone
        mock_supabase.get_trade_by_id.return_value = trade
        mock_supabase.update_trade.return_value = trade

        result = tracker.detect_closed_positions(setup)
        assert len(result) == 1
        assert result[0].mt5_ticket == 11111
        assert result[0].close_reason == "MANUAL_CLOSE"
        # Trade was updated.
        kwargs = mock_supabase.update_trade.call_args.kwargs
        assert kwargs["status"] == "CLOSED"
        assert kwargs["close_reason"] == "MANUAL_CLOSE"

    def test_pending_setup_returns_no_closures(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # PENDING setups have no filled positions yet.
        setup = make_setup(status="PENDING")
        result = tracker.detect_closed_positions(setup)
        assert result == []
        mock_supabase.get_trades_for_setup.assert_not_called()

    def test_terminal_setup_returns_no_closures(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup(status="CLOSED")
        result = tracker.detect_closed_positions(setup)
        assert result == []

    def test_pending_trades_in_active_setup_ignored(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Layer 1 FILLED, Layers 2/3 PENDING. Only FILLED should be checked.
        setup = make_setup(status="ACTIVE")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="FILLED"),
            make_trade(layer_number=2, mt5_ticket=22222, status="PENDING"),
            make_trade(layer_number=3, mt5_ticket=33333, status="PENDING"),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = []  # all gone
        mock_supabase.get_trade_by_id.return_value = trades[0]
        mock_supabase.update_trade.return_value = trades[0]

        result = tracker.detect_closed_positions(setup)
        # Only Layer 1 (FILLED) gets closed; Layers 2/3 are pending limits
        # that just haven't been filled yet — not "closed externally".
        assert len(result) == 1
        assert result[0].trade_id == trades[0].id


# --------------------------------------------------------------------------- #
# reconcile_with_mt5
# --------------------------------------------------------------------------- #


class TestReconcile:
    def test_ghost_position_logged_not_managed(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # MT5 has a position the bot doesn't know about.
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 99999, "price_open": 1850.0},
        ]
        mock_supabase.get_setups_by_status.return_value = []

        result = tracker.reconcile_with_mt5()

        assert result.ghost_tickets == [99999]
        assert result.lost_trade_ids == []
        # Crucially — no DB writes for ghosts.
        mock_supabase.update_trade.assert_not_called()

    def test_lost_position_marked_closed(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Supabase says ticket 11111 is FILLED, MT5 doesn't have it.
        setup = make_setup(status="ACTIVE")
        trade = make_trade(layer_number=1, mt5_ticket=11111, status="FILLED")
        mock_supabase.get_setups_by_status.return_value = [setup]
        mock_supabase.get_trades_for_setup.return_value = [trade]
        mock_mt5.get_open_positions.return_value = []
        mock_supabase.get_trade_by_id.return_value = trade
        mock_supabase.update_trade.return_value = trade

        result = tracker.reconcile_with_mt5()

        assert trade.id in result.lost_trade_ids
        assert result.closed_externally_count == 1
        # Trade was updated to CLOSED.
        kwargs = mock_supabase.update_trade.call_args.kwargs
        assert kwargs["status"] == "CLOSED"
        assert kwargs["close_reason"] == "MANUAL_CLOSE"

    def test_matched_count_correct(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # 3 tickets in MT5 == 3 in Supabase, all matched.
        setup = make_setup(status="ACTIVE")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="FILLED"),
            make_trade(layer_number=2, mt5_ticket=22222, status="FILLED"),
            make_trade(layer_number=3, mt5_ticket=33333, status="FILLED"),
        ]
        mock_supabase.get_setups_by_status.return_value = [setup]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111}, {"ticket": 22222}, {"ticket": 33333},
        ]

        result = tracker.reconcile_with_mt5()
        assert result.matched_count == 3
        assert result.ghost_tickets == []
        assert result.lost_trade_ids == []

    def test_mt5_failure_returns_empty_no_writes(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        mock_mt5.get_open_positions.side_effect = RuntimeError("net down")

        result = tracker.reconcile_with_mt5()
        assert result == ReconcileResult([], [], 0, 0)
        mock_supabase.update_trade.assert_not_called()

    def test_mixed_ghost_lost_matched(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # 11111: matched (both have it)
        # 22222: lost (Supabase only)
        # 99999: ghost (MT5 only)
        setup = make_setup(status="ACTIVE")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="FILLED"),
            make_trade(layer_number=2, mt5_ticket=22222, status="FILLED"),
        ]
        mock_supabase.get_setups_by_status.return_value = [setup]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111}, {"ticket": 99999},
        ]
        mock_supabase.get_trade_by_id.return_value = trades[1]
        mock_supabase.update_trade.return_value = trades[1]

        result = tracker.reconcile_with_mt5()
        assert result.ghost_tickets == [99999]
        assert trades[1].id in result.lost_trade_ids
        assert result.matched_count == 1
