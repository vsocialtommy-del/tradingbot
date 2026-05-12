"""Tests for ``bot.exits.tp_manager`` — per-layer TPs with cascading SL.

Covers (PR #41):
* Each layer's TP fires independently → close that layer's ticket.
* Cascading SL: remaining FILLED layers get ``modify_order``; remaining
  WAITING layers get ``trades.sl_price`` patched.
* WAITING trades stay alive after TP1/TP2 (Q-A decision).
* ``needs_next_tp_recompute`` flag set iff next layer's TP slot is NULL.
* Trigger semantics (BUY: bid >= TP; SELL: ask <= TP) inclusive boundary.
* Layer with NULL TP rides on cascaded SL — no close fires (Q-B).
* The position_tracker setup-completion hook (PR #41) is exercised
  separately in test_position_tracker.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pytest_mock import MockerFixture

from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import PositionTracker
from bot.exits.tp_manager import (
    TPManager,
    _layer_tp,
    _resolve_cascade_sl,
    _trigger_met,
)
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

from datetime import datetime, timezone

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    setup_id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
    planned_tp1_price: Decimal = Decimal("1910.00"),
    planned_tp2_price: Decimal | None = Decimal("1920.00"),
    planned_tp3_price: Decimal | None = Decimal("1930.00"),
    planned_layer1_price: Decimal = Decimal("1900.00"),
    planned_layer2_price: Decimal = Decimal("1897.50"),
    planned_layer3_price: Decimal = Decimal("1895.00"),
) -> Setup:
    return Setup(
        id=setup_id or uuid4(),
        zone_id=uuid4(),
        direction=direction,  # type: ignore[arg-type]
        entry_mode="STRONG_POINT_FIRST_TOUCH",
        planned_layer1_price=planned_layer1_price,
        planned_layer2_price=planned_layer2_price,
        planned_layer3_price=planned_layer3_price,
        planned_sl_price=Decimal("1880.00"),
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
    entry_price: Decimal | None = Decimal("1900.00"),
    direction: str = "BUY",
    sl_price: Decimal = Decimal("1882.50"),
) -> Trade:
    return Trade(
        id=uuid4(),
        setup_id=setup_id,
        layer_number=layer_number,
        direction=direction,  # type: ignore[arg-type]
        order_type="MARKET",
        mt5_ticket=mt5_ticket,
        entry_price=entry_price,
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
) -> TPManager:
    return TPManager(mock_mt5, mock_supabase, mock_tracker)


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


class TestTriggerMet:
    def test_buy_fires_when_bid_at_or_above_tp(self) -> None:
        assert _trigger_met("BUY", 1910.0, bid=1910.0, ask=1910.1) is True
        assert _trigger_met("BUY", 1910.0, bid=1911.0, ask=1911.1) is True

    def test_buy_not_fires_when_bid_below(self) -> None:
        assert _trigger_met("BUY", 1910.0, bid=1909.99, ask=1910.09) is False

    def test_sell_fires_when_ask_at_or_below_tp(self) -> None:
        assert _trigger_met("SELL", 1890.0, bid=1889.9, ask=1890.0) is True
        assert _trigger_met("SELL", 1890.0, bid=1888.9, ask=1889.0) is True

    def test_sell_not_fires_when_ask_above(self) -> None:
        assert _trigger_met("SELL", 1890.0, bid=1890.0, ask=1890.01) is False


class TestLayerTp:
    def test_layer_1_always_set(self) -> None:
        s = make_setup()
        assert _layer_tp(s, 1) == 1910.0

    def test_layer_2_set_returns_value(self) -> None:
        s = make_setup(planned_tp2_price=Decimal("1920"))
        assert _layer_tp(s, 2) == 1920.0

    def test_layer_2_null_returns_none(self) -> None:
        s = make_setup(planned_tp2_price=None)
        assert _layer_tp(s, 2) is None

    def test_layer_3_null_returns_none(self) -> None:
        s = make_setup(planned_tp3_price=None)
        assert _layer_tp(s, 3) is None

    def test_invalid_layer_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown layer_number"):
            _layer_tp(make_setup(), 4)


class TestResolveCascadeSl:
    def test_prefers_entry_price(self) -> None:
        s = make_setup()
        t = make_trade(
            setup_id=s.id, layer_number=1,
            entry_price=Decimal("1900.50"),
        )
        assert _resolve_cascade_sl(t, s) == 1900.5

    def test_falls_back_to_planned(self) -> None:
        # Defensive: a FILLED trade with no entry_price (broker query
        # failure at fill time). Use the planned layer price instead.
        s = make_setup(planned_layer1_price=Decimal("1899.75"))
        t = make_trade(
            setup_id=s.id, layer_number=1, entry_price=None,
        )
        assert _resolve_cascade_sl(t, s) == 1899.75


# --------------------------------------------------------------------------- #
# TPManager.check — per-layer close logic
# --------------------------------------------------------------------------- #


class TestPerLayerClose:
    def test_no_trigger_returns_empty(
        self, manager: TPManager, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup()
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(setup_id=s.id, layer_number=1),
        ]
        # bid below TP1 (1910) → no fire.
        results = manager.check(s, bid=1905.0, ask=1905.1)
        assert results == []

    def test_pending_setup_short_circuits(
        self, manager: TPManager, mock_supabase: MagicMock,
    ) -> None:
        # tp_manager only acts on ACTIVE setups; PENDING ones haven't
        # had Layer 1 filled yet (PR #40 activation gate).
        s = make_setup(status="PENDING")
        results = manager.check(s, bid=1950.0, ask=1950.1)
        assert results == []
        mock_supabase.get_trades_for_setup.assert_not_called()

    def test_layer_1_tp_fires_closes_ticket(
        self, manager: TPManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=Decimal("1900.00"),
        )
        l2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING",
                        mt5_ticket=None, entry_price=None)
        l3 = make_trade(setup_id=s.id, layer_number=3, status="WAITING",
                        mt5_ticket=None, entry_price=None)
        mock_supabase.get_trades_for_setup.return_value = [l1, l2, l3]

        results = manager.check(s, bid=1910.0, ask=1910.1)

        assert len(results) == 1
        r = results[0]
        assert r.layer_number == 1
        assert r.tp_price == 1910.0
        assert r.close_price == 1910.0  # bid for BUY close
        assert r.cascaded_sl == 1900.0
        assert r.error is None
        # Broker close called on L1's specific ticket only.
        mock_mt5.close_position.assert_called_once_with(11111)
        # Trade row marked CLOSED via the tracker (which fires the
        # setup-completion hook).
        mock_tracker.update_trade_status.assert_called_once()
        call = mock_tracker.update_trade_status.call_args
        assert call.args[0] == l1.id
        assert call.args[1] == "CLOSED"
        assert call.kwargs["close_reason"] == "TP1"
        assert call.kwargs["exit_price"] == 1910.0

    def test_sell_layer_1_uses_ask_as_close_price(
        self, manager: TPManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        s = make_setup(
            direction="SELL",
            planned_tp1_price=Decimal("1890"),
            planned_layer1_price=Decimal("1900.00"),
        )
        l1 = make_trade(
            setup_id=s.id, layer_number=1, direction="SELL",
            mt5_ticket=22222, entry_price=Decimal("1900.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        results = manager.check(s, bid=1889.9, ask=1890.0)

        assert len(results) == 1
        r = results[0]
        # SELL closes at ask, not bid.
        assert r.close_price == 1890.0
        mock_mt5.close_position.assert_called_once_with(22222)


class TestCascadingSL:
    """When a layer closes at its TP, remaining FILLED layers get
    ``modify_order`` to the closed layer's entry price; remaining
    WAITING layers get their ``trades.sl_price`` patched."""

    def test_filled_remaining_layer_gets_modify_order(
        self, manager: TPManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=Decimal("1900.00"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2,
            mt5_ticket=22222, entry_price=Decimal("1897.50"),
            sl_price=Decimal("1882.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2]

        manager.check(s, bid=1910.0, ask=1910.1)

        # Two MT5 calls: close L1, modify L2's SL.
        mock_mt5.close_position.assert_called_once_with(11111)
        mock_mt5.modify_order.assert_called_once_with(22222, sl=1900.0)
        # L2's sl_price row also updated.
        update_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.args[0] == l2.id
        ]
        assert len(update_calls) == 1
        assert update_calls[0].kwargs["sl_price"] == Decimal("1900.0")

    def test_waiting_remaining_layer_gets_row_update_only(
        self, manager: TPManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=Decimal("1900.00"),
        )
        l3 = make_trade(
            setup_id=s.id, layer_number=3, status="WAITING",
            mt5_ticket=None, entry_price=None,
            sl_price=Decimal("1882.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l3]

        manager.check(s, bid=1910.0, ask=1910.1)

        # No modify_order for WAITING — there's no broker ticket yet.
        mock_mt5.modify_order.assert_not_called()
        # But L3's row gets the new sl_price so entry_trigger uses it
        # when it eventually fires.
        update_calls = [
            c for c in mock_supabase.update_trade.call_args_list
            if c.args[0] == l3.id
        ]
        assert len(update_calls) == 1
        assert update_calls[0].kwargs["sl_price"] == Decimal("1900.0")

    def test_cascade_uses_fallback_when_entry_price_null(
        self, manager: TPManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        # Defensive: L1 has entry_price=None (broker fill query failed).
        # Cascade SL falls back to planned_layer1_price (1899.50).
        s = make_setup(planned_layer1_price=Decimal("1899.50"))
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=None,
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2,
            mt5_ticket=22222, entry_price=Decimal("1897.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2]

        results = manager.check(s, bid=1910.0, ask=1910.1)

        assert results[0].cascaded_sl == 1899.5
        mock_mt5.modify_order.assert_called_once_with(22222, sl=1899.5)


class TestNeedsNextTpRecompute:
    def test_flag_set_when_next_tp_null(
        self, manager: TPManager, mock_supabase: MagicMock,
    ) -> None:
        # TP1 fires; TP2 is NULL → caller should recompute.
        s = make_setup(planned_tp2_price=None, planned_tp3_price=None)
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=Decimal("1900.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]
        results = manager.check(s, bid=1910.0, ask=1910.1)
        assert results[0].needs_next_tp_recompute is True

    def test_flag_not_set_when_next_tp_present(
        self, manager: TPManager, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup()  # TP2 = 1920 set
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=Decimal("1900.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]
        results = manager.check(s, bid=1910.0, ask=1910.1)
        assert results[0].needs_next_tp_recompute is False

    def test_flag_never_set_for_layer_3_close(
        self, manager: TPManager, mock_supabase: MagicMock,
    ) -> None:
        # Layer 3 is the last — no TP4 exists by construction.
        s = make_setup()
        # L1 + L2 already CLOSED, L3 FILLED.
        l3 = make_trade(
            setup_id=s.id, layer_number=3,
            mt5_ticket=33333, entry_price=Decimal("1895.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l3]
        # bid >= TP3=1930 → fire.
        results = manager.check(s, bid=1930.0, ask=1930.1)
        assert len(results) == 1
        assert results[0].layer_number == 3
        assert results[0].needs_next_tp_recompute is False


class TestLayerWithNullTpRidesCascadedSl:
    """Q-B decision: a layer whose TP is NULL doesn't auto-close.
    It rides on the cascaded SL until external close (SL hit,
    manual, news)."""

    def test_layer_with_null_tp_not_fired_even_at_high_price(
        self, manager: TPManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Hypothetical L2 with TP2 still NULL after a TP1 fire but
        # before recompute happened (or no peak available).
        s = make_setup(planned_tp1_price=Decimal("1910"), planned_tp2_price=None)
        # L1 already closed; L2 still FILLED.
        l2 = make_trade(
            setup_id=s.id, layer_number=2,
            mt5_ticket=22222, entry_price=Decimal("1897.50"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l2]
        # Big move — bid way past TP1. L1 isn't in the list; L2 has
        # no TP → no fire.
        results = manager.check(s, bid=1950.0, ask=1950.1)
        assert results == []
        mock_mt5.close_position.assert_not_called()


class TestMultipleLayersFireSameTick:
    """A sufficiently large price move can cross multiple TPs in a
    single tick. The check loop should process them in layer order
    (1 → 2 → 3) so cascading SL flows monotonically."""

    def test_l1_and_l2_both_fire_when_price_spikes(
        self, manager: TPManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup()  # TP1=1910, TP2=1920, TP3=1930
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=Decimal("1900.00"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2,
            mt5_ticket=22222, entry_price=Decimal("1897.50"),
        )
        l3 = make_trade(
            setup_id=s.id, layer_number=3, status="WAITING",
            mt5_ticket=None, entry_price=None,
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2, l3]

        # bid jumps to 1925 — past TP1 (1910) AND TP2 (1920), short of TP3 (1930).
        results = manager.check(s, bid=1925.0, ask=1925.1)

        assert len(results) == 2
        assert results[0].layer_number == 1
        assert results[1].layer_number == 2
        # Both broker positions closed.
        assert mock_mt5.close_position.call_count == 2
        assert mock_mt5.close_position.call_args_list[0].args == (11111,)
        assert mock_mt5.close_position.call_args_list[1].args == (22222,)


class TestCloseFailureSurfacesError:
    def test_close_position_raises_returns_error_result(
        self, manager: TPManager,
        mock_mt5: MagicMock, mock_supabase: MagicMock, mock_tracker: MagicMock,
    ) -> None:
        # Broker error during the close. tp_manager records the
        # error in the result; no cascade attempted; trade NOT
        # marked CLOSED (broker still has the position).
        mock_mt5.close_position.side_effect = RuntimeError("broker down")
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, layer_number=1,
            mt5_ticket=11111, entry_price=Decimal("1900"),
        )
        l2 = make_trade(
            setup_id=s.id, layer_number=2,
            mt5_ticket=22222, entry_price=Decimal("1897.5"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1, l2]

        results = manager.check(s, bid=1910.0, ask=1910.1)

        assert len(results) == 1
        assert results[0].error is not None
        assert "close_position failed" in results[0].error
        mock_tracker.update_trade_status.assert_not_called()
        mock_mt5.modify_order.assert_not_called()
