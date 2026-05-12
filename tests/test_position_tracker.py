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
    CASCADE_CANCEL_SETUP_STATUSES,
    OPEN_TRADE_STATUSES,
    TERMINAL_SETUP_STATUSES,
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
    status: str = "WAITING",
    mt5_ticket: int | None = 11111,
    order_type: str = "MARKET",
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
            ("WAITING", "FILLED"),
            ("WAITING", "CANCELLED"),
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
            ("FILLED", "WAITING"),         # backwards
            ("CLOSED", "FILLED"),          # from terminal
            ("CANCELLED", "WAITING"),      # from terminal
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
# Cascade-cancel WAITING trades on terminal setup transitions
# --------------------------------------------------------------------------- #


class TestCascadeCancel:
    def _setup_with_waiting_layers(
        self, mock_supabase: MagicMock,
        *,
        setup_status: str = "ACTIVE",
        layer_2_status: str = "WAITING",
        layer_3_status: str = "WAITING",
    ) -> tuple[Setup, list[Trade]]:
        setup = make_setup(status=setup_status)
        trades = [
            make_trade(
                setup_id=setup.id, layer_number=1,
                status="FILLED", mt5_ticket=11111,
            ),
            make_trade(
                setup_id=setup.id, layer_number=2,
                status=layer_2_status, mt5_ticket=None,
            ),
            make_trade(
                setup_id=setup.id, layer_number=3,
                status=layer_3_status, mt5_ticket=None,
            ),
        ]
        # The cascade fetches trades on every cancel call, so make
        # get_trades_for_setup return them. update_trade_status fetches
        # each trade by id; configure that too.
        mock_supabase.get_trades_for_setup.return_value = trades
        # get_trade_by_id is called inside update_trade_status. Return
        # the matching trade based on the requested id.
        def by_id(tid):
            for t in trades:
                if str(t.id) == str(tid):
                    return t
            return None
        mock_supabase.get_trade_by_id.side_effect = by_id
        # update_trade returns the (now-CANCELLED) trade.
        mock_supabase.update_trade.side_effect = lambda *args, **kwargs: trades[0]
        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup
        return setup, trades

    def test_setup_to_stopped_out_cancels_waiting_trades(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        setup, trades = self._setup_with_waiting_layers(mock_supabase)

        tracker.update_setup_status(setup.id, "STOPPED_OUT")

        # 2 WAITING trades should be cancelled (layers 2 and 3).
        # update_trade is called 1× for setup status + 2× for cascade.
        cascade_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.kwargs.get("status") == "CANCELLED"
        ]
        assert len(cascade_calls) == 2

    def test_setup_to_closed_cancels_waiting_trades(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        setup, _ = self._setup_with_waiting_layers(mock_supabase)
        tracker.update_setup_status(setup.id, "CLOSED")
        cascade_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.kwargs.get("status") == "CANCELLED"
        ]
        assert len(cascade_calls) == 2

    def test_setup_pending_to_skipped_no_waiting_trades(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # PENDING setups have no trade rows yet (order_manager hasn't
        # written them). Cascade is a no-op.
        setup = make_setup(status="PENDING")
        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup
        mock_supabase.get_trades_for_setup.return_value = []

        tracker.update_setup_status(setup.id, "SKIPPED")

        # No update_trade calls for cascade.
        cascade_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.kwargs.get("status") == "CANCELLED"
        ]
        assert len(cascade_calls) == 0

    def test_filled_trades_not_cascaded(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # All three layers FILLED (Layer 1 from order_manager + Layers
        # 2/3 fired by entry_trigger). No WAITING → no cascade.
        setup = make_setup(status="ACTIVE")
        trades = [
            make_trade(setup_id=setup.id, layer_number=i, status="FILLED")
            for i in (1, 2, 3)
        ]
        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup
        mock_supabase.get_trades_for_setup.return_value = trades

        tracker.update_setup_status(setup.id, "STOPPED_OUT")

        cascade_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.kwargs.get("status") == "CANCELLED"
        ]
        assert len(cascade_calls) == 0

    def test_only_waiting_trades_cancelled_not_filled(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # Layer 1 FILLED, Layer 2 WAITING, Layer 3 already CANCELLED
        # (some prior reason). Only Layer 2 should get cancelled by
        # this cascade.
        setup, trades = self._setup_with_waiting_layers(
            mock_supabase, layer_3_status="CANCELLED"
        )

        tracker.update_setup_status(setup.id, "STOPPED_OUT")

        cascade_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.kwargs.get("status") == "CANCELLED"
        ]
        # Only Layer 2 cancelled this time.
        assert len(cascade_calls) == 1

    def test_tp1_hit_cancels_waiting_trades(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # ACTIVE → TP1_HIT also triggers cascade. Once profit is locked
        # at TP1, scaling deeper just adds risk in already-booked
        # territory (spec 6.1).
        setup, trades = self._setup_with_waiting_layers(mock_supabase)

        tracker.update_setup_status(setup.id, "TP1_HIT")

        cascade_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.kwargs.get("status") == "CANCELLED"
        ]
        # Layers 2 and 3 cancelled (Layer 1 stays FILLED for the runner).
        assert len(cascade_calls) == 2

    def test_cascade_set_includes_tp1_hit_and_terminal_statuses(self) -> None:
        # Sanity: the canonical set used by update_setup_status is the
        # union of terminal statuses and TP1_HIT.
        assert "TP1_HIT" in CASCADE_CANCEL_SETUP_STATUSES
        assert TERMINAL_SETUP_STATUSES.issubset(CASCADE_CANCEL_SETUP_STATUSES)


# --------------------------------------------------------------------------- #
# update_trade_status
# --------------------------------------------------------------------------- #


class TestUpdateTradeStatus:
    def test_waiting_to_filled_sets_filled_at(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        current = make_trade(status="WAITING")
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

    def test_waiting_to_filled_with_entry_price_and_ticket(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        # Used by entry_trigger when it fires Layer 2/3.
        current = make_trade(status="WAITING", mt5_ticket=None)
        mock_supabase.get_trade_by_id.return_value = current
        mock_supabase.update_trade.return_value = current

        tracker.update_trade_status(
            current.id, "FILLED", entry_price=1897.5, mt5_ticket=44444,
        )
        kwargs = mock_supabase.update_trade.call_args.kwargs
        assert kwargs["status"] == "FILLED"
        assert kwargs["entry_price"] == 1897.5
        assert kwargs["mt5_ticket"] == 44444
        assert "filled_at" in kwargs

    def test_waiting_to_cancelled_sets_closed_at(
        self, tracker: PositionTracker, mock_supabase: MagicMock
    ) -> None:
        # CANCELLED is also a terminal state — should stamp closed_at.
        current = make_trade(status="WAITING")
        mock_supabase.get_trade_by_id.return_value = current
        mock_supabase.update_trade.return_value = current

        tracker.update_trade_status(current.id, "CANCELLED")
        kwargs = mock_supabase.update_trade.call_args.kwargs
        assert kwargs["status"] == "CANCELLED"
        assert "closed_at" in kwargs


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

    def test_waiting_trades_in_active_setup_ignored(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Layer 1 FILLED, Layers 2/3 WAITING. Only FILLED should be checked.
        # WAITING trades have no broker position to be "closed externally" —
        # they're bot-tracked, awaiting trigger.
        setup = make_setup(status="ACTIVE")
        trades = [
            make_trade(layer_number=1, mt5_ticket=11111, status="FILLED"),
            make_trade(layer_number=2, mt5_ticket=None, status="WAITING"),
            make_trade(layer_number=3, mt5_ticket=None, status="WAITING"),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.get_open_positions.return_value = []  # all gone
        mock_supabase.get_trade_by_id.return_value = trades[0]
        mock_supabase.update_trade.return_value = trades[0]

        result = tracker.detect_closed_positions(setup)
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


# --------------------------------------------------------------------------- #
# TTL cache on get_active_setups()
#
# Why this exists: bot loop polls Supabase via get_active_setups() ~3× per
# tick. At the old 10 Hz pace that was ~18 queries/sec, tripping Supabase's
# HTTP/2 max-requests-per-connection limit (~10K) every ~14 minutes and
# cycling the connection with a noisy traceback. A short TTL cache cuts
# steady-state query rate by ~30× without losing correctness — the
# cache is invalidated explicitly on every mutation that could change
# active-setups membership.
# --------------------------------------------------------------------------- #


class TestActiveSetupsCache:
    def test_two_calls_within_ttl_hit_supabase_once(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        mock_supabase.get_setups_by_status.return_value = [make_setup()]
        a = tracker.get_active_setups()
        b = tracker.get_active_setups()
        assert mock_supabase.get_setups_by_status.call_count == 1
        # Both calls return equivalent data.
        assert len(a) == len(b) == 1

    def test_force_refresh_bypasses_cache(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        mock_supabase.get_setups_by_status.return_value = [make_setup()]
        tracker.get_active_setups()
        tracker.get_active_setups(force_refresh=True)
        assert mock_supabase.get_setups_by_status.call_count == 2

    def test_explicit_invalidate_forces_re_query(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        mock_supabase.get_setups_by_status.return_value = [make_setup()]
        tracker.get_active_setups()
        tracker.invalidate_active_setups_cache()
        tracker.get_active_setups()
        assert mock_supabase.get_setups_by_status.call_count == 2

    def test_cache_expires_after_ttl(
        self, mock_mt5: MagicMock, mock_supabase: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        # Tiny TTL so we don't have to sleep.
        t = PositionTracker(
            mt5=mock_mt5, supabase=mock_supabase,
            active_setups_cache_ttl_seconds=0.05,
        )
        mock_supabase.get_setups_by_status.return_value = [make_setup()]
        t.get_active_setups()
        import time as _time
        _time.sleep(0.1)
        t.get_active_setups()
        assert mock_supabase.get_setups_by_status.call_count == 2

    def test_update_setup_status_invalidates_cache(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup(status="PENDING")
        mock_supabase.get_setups_by_status.return_value = [setup]
        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = make_setup(status="ACTIVE")
        mock_supabase.get_trades_for_setup.return_value = []

        # Prime cache.
        tracker.get_active_setups()
        # Mutation should invalidate.
        tracker.update_setup_status(setup.id, "ACTIVE")
        tracker.get_active_setups()
        assert mock_supabase.get_setups_by_status.call_count == 2

    def test_cache_returns_defensive_copy(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # Mutating the returned list must not poison subsequent reads.
        mock_supabase.get_setups_by_status.return_value = [make_setup()]
        first = tracker.get_active_setups()
        first.clear()
        # Within TTL — still a cache hit, but result should be fresh
        # (length 1, not 0).
        second = tracker.get_active_setups()
        assert len(second) == 1


# --------------------------------------------------------------------------- #
# Zone promotion CONFIRMED → ACTIVE on first PENDING → ACTIVE setup transition
# --------------------------------------------------------------------------- #


def make_zone_row(
    *, id: UUID | None = None, status: str = "CONFIRMED",
) -> Any:
    """Build a Zone read model with the lifecycle fields populated."""
    from bot.logging.supabase_logger import Zone
    return Zone(
        id=id or uuid4(),
        symbol="XAUUSD",
        direction="BUY",
        zone_type="STRONG_POINT",
        pattern_type="RBR",
        top=Decimal("105"),
        bottom=Decimal("100"),
        approach_count=0,
        formed_at=NOW,
        status=status,  # type: ignore[arg-type]
        created_at=NOW,
        updated_at=NOW,
    )


class TestZonePromotionOnSetupActivation:
    """PENDING → ACTIVE on a setup also promotes its zone CONFIRMED → ACTIVE."""

    def test_promotes_confirmed_zone_to_active(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        zone_id = uuid4()
        setup = make_setup(status="PENDING")
        setup_active = make_setup(id=setup.id, status="ACTIVE", activated_at=NOW)
        # update_setup returns the new ACTIVE row; we need its zone_id
        # for the promotion path. The Setup model carries zone_id, so
        # patch it on the returned object.
        setup_active = setup_active.model_copy(update={"zone_id": zone_id})

        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup_active
        mock_supabase.get_zone_by_id.return_value = make_zone_row(
            id=zone_id, status="CONFIRMED",
        )

        tracker.update_setup_status(setup.id, "ACTIVE")

        mock_supabase.update_zone_status.assert_called_once_with(
            zone_id, "ACTIVE",
        )

    def test_skips_promotion_when_zone_already_active(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        zone_id = uuid4()
        setup = make_setup(status="PENDING")
        setup_active = make_setup(id=setup.id, status="ACTIVE")
        setup_active = setup_active.model_copy(update={"zone_id": zone_id})

        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup_active
        # Zone already ACTIVE (concurrent setup activation, or replay)
        mock_supabase.get_zone_by_id.return_value = make_zone_row(
            id=zone_id, status="ACTIVE",
        )

        tracker.update_setup_status(setup.id, "ACTIVE")

        mock_supabase.update_zone_status.assert_not_called()

    def test_skips_promotion_when_zone_consumed(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # CONSUMED zones shouldn't get yanked back.
        zone_id = uuid4()
        setup = make_setup(status="PENDING")
        setup_active = make_setup(id=setup.id, status="ACTIVE")
        setup_active = setup_active.model_copy(update={"zone_id": zone_id})

        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup_active
        mock_supabase.get_zone_by_id.return_value = make_zone_row(
            id=zone_id, status="CONSUMED",
        ).model_copy(update={"consumed_at": NOW})

        tracker.update_setup_status(setup.id, "ACTIVE")

        mock_supabase.update_zone_status.assert_not_called()

    def test_promotes_flipped_zone_to_active(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # PR #38: a setup placed on a FLIPPED zone (SnD Flip trade)
        # transitions the zone FLIPPED → ACTIVE — same hook as
        # CONFIRMED → ACTIVE, just a different source status.
        zone_id = uuid4()
        setup = make_setup(status="PENDING")
        setup_active = make_setup(
            id=setup.id, status="ACTIVE", activated_at=NOW,
        ).model_copy(update={"zone_id": zone_id})

        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup_active
        mock_supabase.get_zone_by_id.return_value = make_zone_row(
            id=zone_id, status="FLIPPED",
        ).model_copy(update={
            "violated_at": NOW,
            "flipped_at": NOW,
            "flipped_direction": "SELL",
        })

        tracker.update_setup_status(setup.id, "ACTIVE")

        # Promotion fires for FLIPPED zones now (PR #38).
        mock_supabase.update_zone_status.assert_called_once_with(
            zone_id, "ACTIVE",
        )

    def test_non_pending_to_active_transition_does_not_promote(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # ACTIVE → TP1_HIT shouldn't drive zone promotion.
        setup = make_setup(status="ACTIVE")
        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = make_setup(
            id=setup.id, status="TP1_HIT",
        )

        tracker.update_setup_status(setup.id, "TP1_HIT")

        mock_supabase.update_zone_status.assert_not_called()

    def test_promotion_failure_does_not_break_setup_update(
        self, tracker: PositionTracker, mock_supabase: MagicMock,
    ) -> None:
        # Best-effort: a Supabase error on the zone update is logged
        # + swallowed; the setup transition has already succeeded and
        # must remain successful.
        zone_id = uuid4()
        setup = make_setup(status="PENDING")
        setup_active = make_setup(
            id=setup.id, status="ACTIVE", activated_at=NOW,
        ).model_copy(update={"zone_id": zone_id})

        mock_supabase.get_setup_by_id.return_value = setup
        mock_supabase.update_setup.return_value = setup_active
        mock_supabase.get_zone_by_id.side_effect = RuntimeError("blip")

        # No exception should escape.
        result = tracker.update_setup_status(setup.id, "ACTIVE")
        assert result.status == "ACTIVE"
