"""XAUTISM entries (step 6): 0.01 first touch, layer 2 broker stop at the
zone edge, broker SL/TP on both, M5 close exit."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.execution.entry_trigger import EntryTrigger
from bot.execution.layer2_stop import Layer2StopManager, XautismEntryConfig
from bot.execution.mt5_connector import MT5Connector
from bot.execution.order_manager import place_xautism_orders, _floor_lots
from bot.execution.position_tracker import PositionTracker
from bot.exits.m5_close_exit import closed_beyond_far_edge, run_m5_close_exit
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade

from tests.test_order_manager import make_zone_for_test

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def make_setup(*, direction: str = "BUY", edge: float = 1900.0,
               far: float = 1895.0, status: str = "ACTIVE",
               activated_at: datetime | None = NOW) -> Setup:
    return Setup(
        id=uuid4(), zone_id=uuid4(), direction=direction,  # type: ignore[arg-type]
        entry_mode="STRONG_POINT_FIRST_TOUCH",
        planned_layer1_price=Decimal(str(edge)),
        planned_layer2_price=Decimal(str(edge)),
        planned_layer3_price=Decimal(str(far)),
        planned_sl_price=Decimal("1893.5"), planned_tp1_price=Decimal("1910"),
        status=status,  # type: ignore[arg-type]
        activated_at=activated_at, created_at=NOW, updated_at=NOW,
    )


def make_trade(setup: Setup, *, layer: int, status: str, ticket: int | None,
               order_type: str = "MARKET", lots: str = "0.01") -> Trade:
    return Trade(
        id=uuid4(), setup_id=setup.id, layer_number=layer,
        direction=setup.direction, order_type=order_type,  # type: ignore[arg-type]
        mt5_ticket=ticket, lot_size=Decimal(lots),
        sl_price=Decimal("1893.5"), tp_price=Decimal("1910"),
        status=status,  # type: ignore[arg-type]
        created_at=NOW, updated_at=NOW,
    )


# --------------------------------------------------------------------------- #
# Order placement
# --------------------------------------------------------------------------- #


@pytest.fixture
def mt5(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=MT5Connector)
    m.place_market_order.return_value = 111
    m.get_open_positions.return_value = [{"ticket": 111, "price_open": 1900.0}]
    m.get_pending_orders.return_value = []
    m.get_min_stop_distance.return_value = 0.0
    return m


@pytest.fixture
def supa(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.log_setup.return_value = {"id": str(uuid4())}
    m.log_trade.side_effect = [{"id": str(uuid4())} for _ in range(5)]
    return m


@pytest.fixture
def tracker(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=PositionTracker)


class TestPlaceXautismOrders:
    def test_first_touch_and_layer2_stop_row(self, mt5, supa, tracker) -> None:
        zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
        r = place_xautism_orders(zone, uuid4(), total_lots=0.05,
                                 sl_price=1893.5, tp1_price=1910.0,
                                 mt5=mt5, supabase=supa, tracker=tracker)
        assert r.status == "PLACED"
        kw = mt5.place_market_order.call_args.kwargs
        assert kw["lot_size"] == 0.01
        assert kw["sl"] == 1893.5
        assert kw["tp"] == 1910.0          # TP held at the broker
        rows = [c.args[0] for c in supa.log_trade.call_args_list]
        assert len(rows) == 2               # no layer 3
        l1, l2 = rows
        assert l1.layer_number == 1 and l1.status == "FILLED"
        assert l1.tp_price == Decimal("1910.0")
        assert l2.layer_number == 2 and l2.order_type == "STOP"
        assert l2.status == "WAITING" and l2.mt5_ticket is None
        assert l2.lot_size == Decimal("0.04")
        assert l2.sl_price == Decimal("1893.5") and l2.tp_price == Decimal("1910.0")
        setup_in = supa.log_setup.call_args.args[0]
        assert setup_in.planned_layer1_price == Decimal("1900.0")
        assert setup_in.planned_layer2_price == Decimal("1900.0")
        assert setup_in.planned_layer3_price == Decimal("1895.0")
        tracker.update_setup_status.assert_called_once()

    def test_sell_edge_is_bottom(self, mt5, supa, tracker) -> None:
        mt5.get_open_positions.return_value = [{"ticket": 111, "price_open": 1900.0}]
        zone = make_zone_for_test(direction="SELL", top=1906.0, bottom=1900.0)
        place_xautism_orders(zone, uuid4(), total_lots=0.03,
                             sl_price=1907.8, tp1_price=1890.0,
                             mt5=mt5, supabase=supa, tracker=tracker)
        setup_in = supa.log_setup.call_args.args[0]
        assert setup_in.planned_layer2_price == Decimal("1900.0")
        assert setup_in.planned_layer3_price == Decimal("1906.0")
        assert supa.log_trade.call_args_list[1].args[0].lot_size == Decimal("0.02")

    def test_total_of_one_micro_lot_means_no_layer2(self, mt5, supa, tracker) -> None:
        zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
        r = place_xautism_orders(zone, uuid4(), total_lots=0.01,
                                 sl_price=1893.5, tp1_price=1910.0,
                                 mt5=mt5, supabase=supa, tracker=tracker)
        assert r.status == "PLACED" and r.layer_2_trade_id is None
        assert supa.log_trade.call_count == 1

    def test_rejected_layer1_skips_setup(self, mt5, supa, tracker) -> None:
        # 5 Nov 2026: MT5 rejected 8 orders ("Invalid stops"); each setup
        # stayed PENDING and blocked the one-trade limit all night.
        mt5.place_market_order.side_effect = RuntimeError(
            "order_send rejected: retcode=10016 comment='Invalid stops'")
        zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
        r = place_xautism_orders(zone, uuid4(), total_lots=0.05,
                                 sl_price=1893.5, tp1_price=1910.0,
                                 mt5=mt5, supabase=supa, tracker=tracker)
        assert r.status == "FAILED"
        args, kwargs = tracker.update_setup_status.call_args
        assert args == (r.setup_id, "SKIPPED")
        assert kwargs["skip_reason"].startswith("L1_REJECTED")
        supa.log_trade.assert_not_called()

    def test_gap_through_skips_setup(self, mt5, supa, tracker) -> None:
        # Filled well below a buy zone's bottom -> closed straight away.
        mt5.get_open_positions.return_value = [{"ticket": 111, "price_open": 1890.0}]
        zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
        r = place_xautism_orders(zone, uuid4(), total_lots=0.05,
                                 sl_price=1893.5, tp1_price=1910.0,
                                 mt5=mt5, supabase=supa, tracker=tracker)
        assert r.status == "SKIPPED"
        mt5.close_position.assert_called_once_with(111)
        args, kwargs = tracker.update_setup_status.call_args
        assert args == (r.setup_id, "SKIPPED")
        assert kwargs["skip_reason"] == "GAP_THROUGH"

    def test_skip_failure_never_raises(self, mt5, supa, tracker) -> None:
        mt5.place_market_order.side_effect = RuntimeError("rejected")
        tracker.update_setup_status.side_effect = RuntimeError("supabase down")
        zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
        r = place_xautism_orders(zone, uuid4(), total_lots=0.05,
                                 sl_price=1893.5, tp1_price=1910.0,
                                 mt5=mt5, supabase=supa, tracker=tracker)
        assert r.status == "FAILED"

    def test_floor_lots_float_noise(self) -> None:
        assert _floor_lots(0.03 - 0.01, 0.01) == 0.02
        assert _floor_lots(0.0, 0.01) == 0.0


# --------------------------------------------------------------------------- #
# Layer 2 stop manager
# --------------------------------------------------------------------------- #


def _mgr(mt5, supa, tracker, setup, trades) -> Layer2StopManager:
    tracker.get_active_setups.return_value = [setup]
    supa.get_trades_for_setup.return_value = trades
    return Layer2StopManager(mt5, supa, tracker, XautismEntryConfig())


class TestLayer2Placement:
    def test_waits_until_price_is_inside_the_zone(self, mt5, supa, tracker) -> None:
        s = make_setup()
        trades = [make_trade(s, layer=1, status="FILLED", ticket=111),
                  make_trade(s, layer=2, status="WAITING", ticket=None,
                             order_type="STOP", lots="0.04")]
        m = _mgr(mt5, supa, tracker, s, trades)
        # ask 1899.95: only 0.05 inside, min distance is 0.10.
        assert m.check_live(1899.75, 1899.95) == []
        mt5.place_stop_order.assert_not_called()

    def test_places_buy_stop_at_zone_top(self, mt5, supa, tracker) -> None:
        mt5.place_stop_order.return_value = 222
        s = make_setup()
        l2 = make_trade(s, layer=2, status="WAITING", ticket=None,
                        order_type="STOP", lots="0.04")
        m = _mgr(mt5, supa, tracker, s,
                 [make_trade(s, layer=1, status="FILLED", ticket=111), l2])
        ev = m.check_live(1898.0, 1898.2)
        assert ev[0].action == "PLACED"
        kw = mt5.place_stop_order.call_args.kwargs
        assert kw["direction"] == "BUY" and kw["price"] == 1900.0
        assert kw["lot_size"] == 0.04 and kw["sl"] == 1893.5 and kw["tp"] == 1910.0
        supa.update_trade.assert_called_once_with(l2.id, mt5_ticket=222)

    def test_sell_stop_mirror(self, mt5, supa, tracker) -> None:
        mt5.place_stop_order.return_value = 222
        s = make_setup(direction="SELL", edge=1900.0, far=1906.0)
        m = _mgr(mt5, supa, tracker, s, [
            make_trade(s, layer=1, status="FILLED", ticket=111),
            make_trade(s, layer=2, status="WAITING", ticket=None, order_type="STOP"),
        ])
        assert m.check_live(1900.05, 1900.3) == []          # too close
        assert m.check_live(1901.0, 1901.3)[0].action == "PLACED"

    def test_broker_stop_level_widens_distance(self, mt5, supa, tracker) -> None:
        mt5.get_min_stop_distance.return_value = 0.5
        s = make_setup()
        m = _mgr(mt5, supa, tracker, s, [
            make_trade(s, layer=1, status="FILLED", ticket=111),
            make_trade(s, layer=2, status="WAITING", ticket=None, order_type="STOP"),
        ])
        assert m.check_live(1899.5, 1899.7) == []
        mt5.place_stop_order.assert_not_called()


class TestLayer2Cancel:
    def test_cancel_when_tp_hit_before_layer2_placed(self, mt5, supa, tracker) -> None:
        mt5.get_open_positions.return_value = []   # layer 1 gone (TP hit)
        s = make_setup()
        l2 = make_trade(s, layer=2, status="WAITING", ticket=None, order_type="STOP")
        m = _mgr(mt5, supa, tracker, s,
                 [make_trade(s, layer=1, status="FILLED", ticket=111), l2])
        ev = m.check_live(1911.0, 1911.2)
        assert ev[0].action == "CANCELLED"
        tracker.update_trade_status.assert_called_once_with(
            l2.id, "CANCELLED", close_reason="CASCADE_CANCELLED",
        )
        mt5.place_stop_order.assert_not_called()

    def test_cancel_pending_stop_at_broker_when_layer1_closed(self, mt5, supa, tracker) -> None:
        mt5.get_open_positions.return_value = []          # layer 1 stopped out
        mt5.get_pending_orders.return_value = [{"ticket": 222}]
        s = make_setup()
        l2 = make_trade(s, layer=2, status="WAITING", ticket=222, order_type="STOP")
        m = _mgr(mt5, supa, tracker, s,
                 [make_trade(s, layer=1, status="FILLED", ticket=111), l2])
        m.check_live(1893.0, 1893.2)
        mt5.cancel_pending_order.assert_called_once_with(222)
        tracker.update_trade_status.assert_called_once_with(
            l2.id, "CANCELLED", close_reason="CASCADE_CANCELLED",
        )

    def test_pending_stop_left_alone_while_layer1_open(self, mt5, supa, tracker) -> None:
        mt5.get_pending_orders.return_value = [{"ticket": 222}]
        s = make_setup()
        m = _mgr(mt5, supa, tracker, s, [
            make_trade(s, layer=1, status="FILLED", ticket=111),
            make_trade(s, layer=2, status="WAITING", ticket=222, order_type="STOP"),
        ])
        assert m.check_live(1898.0, 1898.2) == []
        mt5.cancel_pending_order.assert_not_called()


class TestLayer2Fill:
    def test_marked_filled_when_stop_triggers(self, mt5, supa, tracker) -> None:
        mt5.get_open_positions.return_value = [
            {"ticket": 111, "price_open": 1900.0},
            {"ticket": 222, "price_open": 1900.1},
        ]
        s = make_setup()
        l2 = make_trade(s, layer=2, status="WAITING", ticket=222, order_type="STOP")
        m = _mgr(mt5, supa, tracker, s,
                 [make_trade(s, layer=1, status="FILLED", ticket=111), l2])
        ev = m.check_live(1900.2, 1900.4)
        assert ev[0].action == "FILLED"
        tracker.update_trade_status.assert_called_once_with(
            l2.id, "FILLED", entry_price=1900.1,
        )

    def test_filled_and_closed_between_ticks(self, mt5, supa, tracker) -> None:
        mt5.get_open_positions.return_value = []
        mt5.get_order_history_state.return_value = "FILLED"
        s = make_setup()
        l2 = make_trade(s, layer=2, status="WAITING", ticket=222, order_type="STOP")
        m = _mgr(mt5, supa, tracker, s,
                 [make_trade(s, layer=1, status="FILLED", ticket=111), l2])
        ev = m.check_live(1911.0, 1911.2)
        assert ev[0].action == "FILLED_AND_CLOSED"

    def test_ignores_setups_without_stop_layer(self, mt5, supa, tracker) -> None:
        s = make_setup()
        m = _mgr(mt5, supa, tracker, s,
                 [make_trade(s, layer=1, status="FILLED", ticket=111)])
        assert m.check_live(1898.0, 1898.2) == []
        mt5.get_open_positions.assert_not_called()


# --------------------------------------------------------------------------- #
# Entry trigger must never market-fire a STOP layer
# --------------------------------------------------------------------------- #


def test_entry_trigger_skips_stop_layers(mt5, supa, tracker) -> None:
    s = make_setup()
    tracker.get_active_setups.return_value = [s]
    supa.get_trades_for_setup.return_value = [
        make_trade(s, layer=2, status="WAITING", ticket=None, order_type="STOP"),
    ]
    trig = EntryTrigger(mt5, supa, tracker)
    assert trig.check_live(1899.0, 1899.2) == []
    mt5.place_market_order.assert_not_called()


# --------------------------------------------------------------------------- #
# M5 close exit
# --------------------------------------------------------------------------- #


def _bar(close: float) -> pd.Series:
    return pd.Series({"open": 1896.0, "high": 1897.0, "low": 1893.9, "close": close})


class TestM5CloseExit:
    def test_rule(self) -> None:
        assert closed_beyond_far_edge("BUY", 1894.9, 1900.0, 1895.0)
        assert not closed_beyond_far_edge("BUY", 1895.0, 1900.0, 1895.0)
        assert closed_beyond_far_edge("SELL", 1906.1, 1906.0, 1900.0)
        assert not closed_beyond_far_edge("SELL", 1905.0, 1906.0, 1900.0)

    def test_closes_layers_and_cancels_pending_stop(self, mt5, tracker) -> None:
        s = make_setup()
        l1 = make_trade(s, layer=1, status="FILLED", ticket=111)
        l2 = make_trade(s, layer=2, status="WAITING", ticket=222, order_type="STOP")
        tracker.get_trades_for_setup.return_value = [l1, l2]
        r = run_m5_close_exit(
            s, _bar(1894.5), pd.Timestamp(NOW),
            zone_top=1900.0, zone_bottom=1895.0, mt5=mt5, tracker=tracker,
        )
        assert r is not None and r.closed_layers == 1 and r.cancelled_layers == 1
        mt5.close_position.assert_called_once_with(111)
        mt5.cancel_pending_order.assert_called_once_with(222)
        tracker.update_trade_status.assert_any_call(
            l1.id, "CLOSED", close_reason="M5_CLOSE_EXIT")
        tracker.update_trade_status.assert_any_call(
            l2.id, "CANCELLED", close_reason="M5_CLOSE_EXIT")

    def test_close_inside_zone_does_nothing(self, mt5, tracker) -> None:
        s = make_setup()
        assert run_m5_close_exit(
            s, _bar(1895.5), pd.Timestamp(NOW),
            zone_top=1900.0, zone_bottom=1895.0, mt5=mt5, tracker=tracker,
        ) is None
        mt5.close_position.assert_not_called()

    def test_candle_that_closed_before_entry_is_ignored(self, mt5, tracker) -> None:
        s = make_setup(activated_at=NOW)
        bar_open = pd.Timestamp(NOW - timedelta(minutes=10))  # closed 5 min before
        assert run_m5_close_exit(
            s, _bar(1890.0), bar_open,
            zone_top=1900.0, zone_bottom=1895.0, mt5=mt5, tracker=tracker,
        ) is None
