"""Tests for ``bot.exits.trailing_stop_manager`` — L1 trailing stop.

The trailing-stop manager locks in progressive profit on a setup
whose Layer 1 is the only filled layer. It runs on every M5 close
(by default) and:

* Skips when more than one layer is FILLED (zone-exit / TP cascade
  own multi-layer SL moves).
* Skips when current profit is below the activation threshold
  (30 % of the distance from entry to TP1, by default).
* Computes a new SL = current_price ± (50 % × current profit), so
  half of the running profit is locked in and the other half is
  given back on a reversal.
* Only modifies the broker SL if the new value is BETTER (closer to
  current price) than the existing one — no untrailing.
* Returns a result describing what was done; on broker failure the
  result carries the error string and the next M5 close re-evaluates.
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
from bot.exits.trailing_stop_manager import (
    TrailingStopConfig,
    TrailingStopManager,
    _compute_trailed_sl,
)
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    setup_id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
    # BUY default: L1 at zone.top, TP1 above. distance_to_tp1 = 10.
    planned_layer1_price: Decimal = Decimal("4700.00"),
    planned_layer2_price: Decimal = Decimal("4697.50"),
    planned_layer3_price: Decimal = Decimal("4695.00"),
    planned_sl_price: Decimal = Decimal("4685.00"),
    planned_tp1_price: Decimal = Decimal("4710.00"),
    planned_tp2_price: Decimal | None = Decimal("4720.00"),
    planned_tp3_price: Decimal | None = Decimal("4730.00"),
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


def make_sell_setup(**overrides: object) -> Setup:
    """SELL setup mirroring the BUY defaults. distance_to_tp1 = 10."""
    defaults: dict[str, object] = dict(
        direction="SELL",
        planned_layer1_price=Decimal("4700.00"),
        planned_layer2_price=Decimal("4702.50"),
        planned_layer3_price=Decimal("4705.00"),
        planned_sl_price=Decimal("4715.00"),
        planned_tp1_price=Decimal("4690.00"),
        planned_tp2_price=Decimal("4680.00"),
        planned_tp3_price=Decimal("4670.00"),
    )
    defaults.update(overrides)
    return make_setup(**defaults)  # type: ignore[arg-type]


def make_trade(
    *,
    setup_id: UUID,
    layer_number: int = 1,
    status: str = "FILLED",
    mt5_ticket: int | None = 11111,
    entry_price: Decimal | None = Decimal("4700.00"),
    sl_price: Decimal = Decimal("4685.00"),
    direction: str = "BUY",
) -> Trade:
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
) -> TrailingStopManager:
    return TrailingStopManager(mock_mt5, mock_supabase, mock_tracker)


# --------------------------------------------------------------------------- #
# Pure helper: _compute_trailed_sl
# --------------------------------------------------------------------------- #


class TestComputeTrailedSL:
    """Direct unit tests on the SL-arithmetic helper."""

    def test_buy_below_threshold_returns_none(self) -> None:
        # entry=4700, tp1=4710, distance=10, threshold at 30% = 3.
        # bid=4702 → profit=2 < 3 → no trail.
        assert (
            _compute_trailed_sl(
                direction="BUY", entry=4700.0, tp1=4710.0,
                bid=4702.0, ask=4702.1, old_sl=4685.0,
                activation_pct=0.30, trail_pct=0.50,
            )
            is None
        )

    def test_buy_at_threshold_activates(self) -> None:
        # bid=4703 → profit=3 == threshold → activate.
        # trail_distance = 0.5 * 3 = 1.5 → new_sl = 4703 - 1.5 = 4701.5.
        result = _compute_trailed_sl(
            direction="BUY", entry=4700.0, tp1=4710.0,
            bid=4703.0, ask=4703.1, old_sl=4685.0,
            activation_pct=0.30, trail_pct=0.50,
        )
        assert result == pytest.approx(4701.5)

    def test_buy_well_in_profit(self) -> None:
        # bid=4708 → profit=8 → trail_dist=4 → new_sl=4704.
        result = _compute_trailed_sl(
            direction="BUY", entry=4700.0, tp1=4710.0,
            bid=4708.0, ask=4708.1, old_sl=4685.0,
            activation_pct=0.30, trail_pct=0.50,
        )
        assert result == pytest.approx(4704.0)

    def test_buy_no_tighten_returns_none(self) -> None:
        # Existing SL already at 4707 (above the trail target of 4704).
        # No improvement → None.
        assert (
            _compute_trailed_sl(
                direction="BUY", entry=4700.0, tp1=4710.0,
                bid=4708.0, ask=4708.1, old_sl=4707.0,
                activation_pct=0.30, trail_pct=0.50,
            )
            is None
        )

    def test_sell_below_threshold_returns_none(self) -> None:
        # entry=4700, tp1=4690, distance=10. ask=4698 → profit=2 < 3.
        assert (
            _compute_trailed_sl(
                direction="SELL", entry=4700.0, tp1=4690.0,
                bid=4697.9, ask=4698.0, old_sl=4715.0,
                activation_pct=0.30, trail_pct=0.50,
            )
            is None
        )

    def test_sell_at_threshold_activates(self) -> None:
        # ask=4697 → profit=3 → trail_dist=1.5 → new_sl=4697+1.5=4698.5.
        result = _compute_trailed_sl(
            direction="SELL", entry=4700.0, tp1=4690.0,
            bid=4696.9, ask=4697.0, old_sl=4715.0,
            activation_pct=0.30, trail_pct=0.50,
        )
        assert result == pytest.approx(4698.5)

    def test_sell_well_in_profit(self) -> None:
        # ask=4692 → profit=8 → trail_dist=4 → new_sl=4696.
        result = _compute_trailed_sl(
            direction="SELL", entry=4700.0, tp1=4690.0,
            bid=4691.9, ask=4692.0, old_sl=4715.0,
            activation_pct=0.30, trail_pct=0.50,
        )
        assert result == pytest.approx(4696.0)

    def test_buy_new_sl_on_wrong_side_of_bid_blocked(self) -> None:
        # Pathological: trail_pct=0 → new_sl == bid. Broker would
        # reject (SL must sit below bid for a BUY).
        assert (
            _compute_trailed_sl(
                direction="BUY", entry=4700.0, tp1=4710.0,
                bid=4708.0, ask=4708.1, old_sl=4685.0,
                activation_pct=0.30, trail_pct=0.0,
            )
            is None
        )

    def test_degenerate_geometry_returns_none(self) -> None:
        # TP1 below entry on a BUY (shouldn't happen for a real setup
        # but defensive guard).
        assert (
            _compute_trailed_sl(
                direction="BUY", entry=4700.0, tp1=4690.0,
                bid=4705.0, ask=4705.1, old_sl=4685.0,
                activation_pct=0.30, trail_pct=0.50,
            )
            is None
        )


# --------------------------------------------------------------------------- #
# Setup-level: scope / activation gates
# --------------------------------------------------------------------------- #


class TestScopeGates:
    def test_pending_setup_short_circuits(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup(status="PENDING")
        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is None
        mock_supabase.get_trades_for_setup.assert_not_called()

    def test_no_filled_layers_returns_none(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup()
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, layer_number=1, status="WAITING",
            ),
        ]
        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is None

    def test_multi_layer_filled_skips(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # L1 + L2 both FILLED → zone-exit/tp-cascade territory.
        s = make_setup()
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, layer_number=1, mt5_ticket=11111,
                entry_price=Decimal("4700.00"),
                sl_price=Decimal("4685.00"),
            ),
            make_trade(
                setup_id=s.id, layer_number=2, mt5_ticket=22222,
                entry_price=Decimal("4697.50"),
                sl_price=Decimal("4685.00"),
            ),
        ]
        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is None
        mock_mt5.modify_order.assert_not_called()

    def test_only_l2_filled_skips(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # L1 already closed (TP1) and only L2 FILLED — TP cascade
        # owns this SL; trailing-stop must not interfere.
        s = make_setup()
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, layer_number=1, status="CLOSED",
            ),
            make_trade(
                setup_id=s.id, layer_number=2, mt5_ticket=22222,
                entry_price=Decimal("4697.50"),
                sl_price=Decimal("4700.00"),
            ),
        ]
        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is None
        mock_mt5.modify_order.assert_not_called()

    def test_l1_filled_l2_l3_waiting_proceeds(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # WAITING layers don't count against "L1 only filled".
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, layer_number=1, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4685.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [
            l1,
            make_trade(setup_id=s.id, layer_number=2, status="WAITING"),
            make_trade(setup_id=s.id, layer_number=3, status="WAITING"),
        ]
        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is not None
        mock_mt5.modify_order.assert_called_once()


# --------------------------------------------------------------------------- #
# Activation + tightening behaviour
# --------------------------------------------------------------------------- #


class TestActivation:
    def test_below_threshold_no_action(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # entry=4700, tp1=4710, threshold=3. bid=4702 → profit=2 < 3.
        s = make_setup()
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, mt5_ticket=11111,
                entry_price=Decimal("4700.00"),
                sl_price=Decimal("4685.00"),
            ),
        ]
        result = manager.check(s, bid=4702.0, ask=4702.1)
        assert result is None
        mock_mt5.modify_order.assert_not_called()

    def test_at_threshold_activates_buy(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # bid=4703 → profit=3 == threshold → fire.
        # new_sl = 4703 - 0.5*3 = 4701.5.
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4685.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        result = manager.check(s, bid=4703.0, ask=4703.1)

        assert result is not None
        assert result.trade_id == l1.id
        assert result.new_sl == pytest.approx(4701.5)
        assert result.old_sl == 4685.0
        assert result.current_profit == pytest.approx(3.0)
        mock_mt5.modify_order.assert_called_once_with(11111, sl=4701.5)

    def test_at_threshold_activates_sell(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        s = make_sell_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111, direction="SELL",
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4715.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        # ask=4697 → profit=3 → trail_dist=1.5 → new_sl=4698.5.
        result = manager.check(s, bid=4696.9, ask=4697.0)

        assert result is not None
        assert result.new_sl == pytest.approx(4698.5)
        mock_mt5.modify_order.assert_called_once_with(11111, sl=4698.5)

    def test_keeps_trailing_as_profit_grows(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Two M5 closes: first activates at 4704, second tightens at
        # the new profit level. We model the second by feeding a
        # higher bid AND the supabase-side new sl_price (the manager
        # patched it after the first call).
        s = make_setup()
        l1_after_first = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4702.00"),  # ← already trailed once
        )
        mock_supabase.get_trades_for_setup.return_value = [l1_after_first]

        # bid=4706 → profit=6 → trail_dist=3 → new_sl=4703.
        # 4703 > 4702 → tighten.
        result = manager.check(s, bid=4706.0, ask=4706.1)

        assert result is not None
        assert result.new_sl == pytest.approx(4703.0)
        mock_mt5.modify_order.assert_called_once_with(11111, sl=4703.0)

    def test_does_not_untrail_on_retrace(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Previous trail moved SL to 4704. Profit retraces to a level
        # where naive recalc gives a LOWER SL. We must not move it.
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4704.00"),  # ← peak trail
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        # bid=4705 → profit=5 → trail_dist=2.5 → new_sl=4702.5 < 4704.
        # No tighten.
        result = manager.check(s, bid=4705.0, ask=4705.1)
        assert result is None
        mock_mt5.modify_order.assert_not_called()

    def test_existing_sl_already_better_no_update(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # SL already at the same value the trail calc would produce.
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4701.5"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        # bid=4703 → trail target = 4701.5 == current → no tighten.
        result = manager.check(s, bid=4703.0, ask=4703.1)
        assert result is None
        mock_mt5.modify_order.assert_not_called()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_updates_trade_row_with_new_sl(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
    ) -> None:
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4685.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        manager.check(s, bid=4708.0, ask=4708.1)

        # new_sl = 4708 - 0.5*8 = 4704.
        mock_supabase.update_trade.assert_called_once()
        call = mock_supabase.update_trade.call_args
        assert call.args[0] == l1.id
        assert float(call.kwargs["sl_price"]) == pytest.approx(4704.0)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestFailureModes:
    def test_modify_order_failure_caught(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4685.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]
        mock_mt5.modify_order.side_effect = RuntimeError(
            "broker requote during modify"
        )

        result = manager.check(s, bid=4708.0, ask=4708.1)

        # No crash; the error is reported and an event logged.
        assert result is not None
        assert result.error is not None
        assert "broker requote during modify" in result.error
        # update_trade NOT called when broker modify failed.
        mock_supabase.update_trade.assert_not_called()
        mock_supabase.log_event.assert_called_once()

    def test_get_trades_failure_returns_none(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        s = make_setup()
        mock_supabase.get_trades_for_setup.side_effect = RuntimeError(
            "supabase 500"
        )
        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is None
        mock_mt5.modify_order.assert_not_called()

    def test_update_trade_failure_caught(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Broker succeeded, supabase patch failed. The result carries
        # the error but the broker SL is moved (the safer side of the
        # desync).
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4685.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]
        mock_supabase.update_trade.side_effect = RuntimeError(
            "supabase update timeout"
        )

        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is not None
        assert result.error is not None
        assert "supabase update timeout" in result.error
        mock_mt5.modify_order.assert_called_once_with(11111, sl=4704.0)

    def test_missing_entry_price_skipped(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        # Defensive: a FILLED row with NULL entry_price (shouldn't
        # happen at the broker) is just skipped, no crash.
        s = make_setup()
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=None,
            sl_price=Decimal("4685.00"),
        )
        # bypass pydantic's FILLED-implies-entry_price assumption by
        # directly mutating the model_dump (the validator allows None).
        mock_supabase.get_trades_for_setup.return_value = [l1]
        result = manager.check(s, bid=4708.0, ask=4708.1)
        assert result is None
        mock_mt5.modify_order.assert_not_called()


# --------------------------------------------------------------------------- #
# Integration scenarios from the design doc
# --------------------------------------------------------------------------- #


class TestProductionScenarios:
    def test_real_sell_4700_to_4690_scenario(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        """Real production: SELL entry $4700, TP1 $4685, distance=15.

        Activation = 0.30 * 15 = 4.5. Price drops to $4694.50 → profit
        = 5.5 > 4.5 → trailing fires. new_sl = 4694.5 + 0.5*5.5 = 4697.25.
        """
        s = make_sell_setup(
            planned_layer1_price=Decimal("4700.00"),
            planned_tp1_price=Decimal("4685.00"),
            planned_sl_price=Decimal("4715.00"),
        )
        l1 = make_trade(
            setup_id=s.id, direction="SELL", mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4715.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        # ask=4694.5 (price moved $5.50 in our favour).
        result = manager.check(s, bid=4694.4, ask=4694.5)

        assert result is not None
        assert result.new_sl == pytest.approx(4697.25)
        mock_mt5.modify_order.assert_called_once_with(11111, sl=4697.25)

    def test_buy_at_be_then_trailing_takes_over(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        """Zone-exit already moved SL to entry (BE). Trailing kicks in
        once profit crosses 30 % of distance to TP1 and tightens
        further than BE."""
        s = make_setup()
        # SL already at entry (post-zone-exit BE).
        l1 = make_trade(
            setup_id=s.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4700.00"),
        )
        mock_supabase.get_trades_for_setup.return_value = [l1]

        # bid=4706 → profit=6 → trail=3 → new_sl=4703. 4703 > 4700 BE.
        result = manager.check(s, bid=4706.0, ask=4706.1)
        assert result is not None
        assert result.new_sl == pytest.approx(4703.0)


# --------------------------------------------------------------------------- #
# Multiple setups (independence)
# --------------------------------------------------------------------------- #


class TestMultipleSetups:
    def test_two_setups_trailed_independently(
        self, manager: TrailingStopManager, mock_supabase: MagicMock,
        mock_mt5: MagicMock,
    ) -> None:
        s_a = make_setup()
        s_b = make_setup()

        l1_a = make_trade(
            setup_id=s_a.id, mt5_ticket=11111,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4685.00"),
        )
        l1_b = make_trade(
            setup_id=s_b.id, mt5_ticket=22222,
            entry_price=Decimal("4700.00"),
            sl_price=Decimal("4685.00"),
        )

        def get_trades(setup_id: UUID) -> list[Trade]:
            return [l1_a] if setup_id == s_a.id else [l1_b]

        mock_supabase.get_trades_for_setup.side_effect = get_trades

        r_a = manager.check(s_a, bid=4708.0, ask=4708.1)
        r_b = manager.check(s_b, bid=4706.0, ask=4706.1)

        assert r_a is not None and r_b is not None
        assert r_a.new_sl == pytest.approx(4704.0)
        assert r_b.new_sl == pytest.approx(4703.0)

        modify_calls = mock_mt5.modify_order.call_args_list
        modified = {c.args[0]: c.kwargs["sl"] for c in modify_calls}
        assert modified[11111] == pytest.approx(4704.0)
        assert modified[22222] == pytest.approx(4703.0)


# --------------------------------------------------------------------------- #
# Custom config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_custom_activation_pct(
        self, mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        # 50 % activation: profit=4 < 5 threshold → no fire.
        mgr = TrailingStopManager(
            mock_mt5, mock_supabase, mock_tracker,
            config=TrailingStopConfig(activation_pct_of_tp1=0.50),
        )
        s = make_setup()
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, mt5_ticket=11111,
                entry_price=Decimal("4700.00"),
                sl_price=Decimal("4685.00"),
            ),
        ]
        assert mgr.check(s, bid=4704.0, ask=4704.1) is None

    def test_custom_trail_pct(
        self, mock_mt5: MagicMock, mock_supabase: MagicMock,
        mock_tracker: MagicMock,
    ) -> None:
        # 25 % trail: bid=4708, profit=8, trail_dist=2 → new_sl=4706.
        mgr = TrailingStopManager(
            mock_mt5, mock_supabase, mock_tracker,
            config=TrailingStopConfig(trail_pct_of_profit=0.25),
        )
        s = make_setup()
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(
                setup_id=s.id, mt5_ticket=11111,
                entry_price=Decimal("4700.00"),
                sl_price=Decimal("4685.00"),
            ),
        ]
        result = mgr.check(s, bid=4708.0, ask=4708.1)
        assert result is not None
        assert result.new_sl == pytest.approx(4706.0)
