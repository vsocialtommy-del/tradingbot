"""75/25 runner with a trailing stop (Nov 2026): placement split, TP1
handling, the 0.01-only runner, the trail itself and restart recovery."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture

from bot.execution.layer2_stop import Layer2StopManager, XautismEntryConfig
from bot.execution.mt5_connector import MT5Connector
from bot.execution.order_manager import place_xautism_orders
from bot.execution.position_tracker import PositionTracker
from bot.exits.runner_trail import RunnerTrailManager
from bot.logging.supabase_logger import SupabaseLogger

from tests.test_order_manager import make_zone_for_test
from tests.test_xautism_entries import make_setup, make_trade

TP = 1910.0  # make_setup's planned TP1 (BUY, entry 1900)


# --------------------------------------------------------------------------- #
# Placement split
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


class TestRunnerPlacement:
    def _place(self, mt5, supa, tracker, l2: float):
        zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
        return place_xautism_orders(
            zone, uuid4(), total_lots=round(0.01 + l2, 2), layer2_lots=l2,
            runner_keep_frac=0.25, sl_price=1893.5, tp1_price=TP,
            mt5=mt5, supabase=supa, tracker=tracker,
        )

    def test_splits_layer2_75_25_and_layer1_has_no_broker_tp(self, mt5, supa, tracker) -> None:
        r = self._place(mt5, supa, tracker, l2=0.17)   # 0.18 total
        assert r.status == "PLACED" and r.layer_3_trade_id is not None
        assert mt5.place_market_order.call_args.kwargs["tp"] is None
        rows = [c.args[0] for c in supa.log_trade.call_args_list]
        l1, l2, run = rows
        assert l2.layer_number == 2 and l2.lot_size == Decimal("0.13")
        assert l2.tp_price == Decimal(str(TP))         # main part: broker TP
        assert run.layer_number == 3 and run.lot_size == Decimal("0.04")
        assert run.order_type == "STOP" and run.status == "WAITING"
        assert run.tp_price is None                     # runner: no TP
        # 75% at TP1 = L1 0.01 + L2 0.13 = 0.14 of 0.18
        assert float(l1.lot_size + l2.lot_size) == pytest.approx(0.14)

    def test_tiny_position_keeps_layer2_whole(self, mt5, supa, tracker) -> None:
        r = self._place(mt5, supa, tracker, l2=0.02)   # 0.03 → 25% rounds to 0
        assert r.layer_3_trade_id is None
        rows = [c.args[0] for c in supa.log_trade.call_args_list]
        assert len(rows) == 2 and rows[1].lot_size == Decimal("0.02")
        assert mt5.place_market_order.call_args.kwargs["tp"] is None

    def test_runner_off_is_unchanged(self, mt5, supa, tracker) -> None:
        zone = make_zone_for_test(direction="BUY", top=1900.0, bottom=1895.0)
        r = place_xautism_orders(zone, uuid4(), total_lots=0.18, layer2_lots=0.17,
                                 sl_price=1893.5, tp1_price=TP,
                                 mt5=mt5, supabase=supa, tracker=tracker)
        assert r.layer_3_trade_id is None
        assert mt5.place_market_order.call_args.kwargs["tp"] == TP


class TestLayer2ManagerPlacesRunnerStop:
    def test_both_waiting_stops_are_placed(self, mocker: MockerFixture) -> None:
        setup = make_setup()
        l1 = make_trade(setup, layer=1, status="FILLED", ticket=111)
        l2 = make_trade(setup, layer=2, status="WAITING", ticket=None,
                        order_type="STOP", lots="0.13")
        run = make_trade(setup, layer=3, status="WAITING", ticket=None,
                         order_type="STOP", lots="0.04")
        run = run.model_copy(update={"tp_price": None})
        mt5 = mocker.MagicMock(spec=MT5Connector)
        mt5.get_open_positions.return_value = [{"ticket": 111, "price_open": 1900.0}]
        mt5.get_pending_orders.return_value = []
        mt5.get_min_stop_distance.return_value = 0.0
        mt5.place_stop_order.side_effect = [201, 202]
        supa = mocker.MagicMock(spec=SupabaseLogger)
        supa.get_trades_for_setup.return_value = [l1, l2, run]
        trk = mocker.MagicMock(spec=PositionTracker)
        trk.get_active_setups.return_value = [setup]
        evs = Layer2StopManager(mt5, supa, trk).check_live(bid=1899.7, ask=1899.8)
        assert [e.action for e in evs] == ["PLACED", "PLACED"]
        calls = mt5.place_stop_order.call_args_list
        assert calls[0].kwargs["lot_size"] == 0.13 and calls[0].kwargs["tp"] == TP
        assert calls[1].kwargs["lot_size"] == 0.04 and calls[1].kwargs["tp"] is None


# --------------------------------------------------------------------------- #
# Runner management
# --------------------------------------------------------------------------- #


class Broker:
    """Tiny stand-in for the MT5 positions + SL modifications."""

    def __init__(self, positions: list[dict]) -> None:
        self.pos = {p["ticket"]: dict(p) for p in positions}

    def get_open_positions(self, symbol=None):
        return [dict(p) for p in self.pos.values()]

    def modify_order(self, ticket, sl=None, tp=None):
        if sl is not None:
            self.pos[ticket]["sl"] = sl

    def close_position(self, ticket, partial_lots=None):
        self.pos.pop(ticket)


def make_manager(mocker: MockerFixture, setup, trades, broker: Broker):
    mt5 = mocker.MagicMock(spec=MT5Connector)
    mt5.get_open_positions.side_effect = broker.get_open_positions
    mt5.modify_order.side_effect = broker.modify_order
    mt5.close_position.side_effect = broker.close_position
    trk = mocker.MagicMock(spec=PositionTracker)
    trk.get_active_setups.return_value = [setup]
    trk.get_trades_for_setup.return_value = trades
    return RunnerTrailManager(mt5, trk, XautismEntryConfig()), mt5, trk


class TestRunner:
    def _full_setup(self):
        setup = make_setup()
        l1 = make_trade(setup, layer=1, status="FILLED", ticket=111)
        l2 = make_trade(setup, layer=2, status="FILLED", ticket=201, order_type="STOP", lots="0.13")
        run = make_trade(setup, layer=3, status="FILLED", ticket=202, order_type="STOP", lots="0.04")
        broker = Broker([
            {"ticket": 111, "price_open": 1900.0, "sl": 1893.5, "tp": 0.0},
            {"ticket": 201, "price_open": 1900.0, "sl": 1893.5, "tp": TP},
            {"ticket": 202, "price_open": 1900.0, "sl": 1893.5, "tp": 0.0},
        ])
        return setup, [l1, l2, run], broker

    def test_nothing_before_tp1(self, mocker) -> None:
        setup, trades, broker = self._full_setup()
        mgr, mt5, trk = make_manager(mocker, setup, trades, broker)
        assert mgr.check_live(bid=1909.0, ask=1909.1) == []
        mt5.close_position.assert_not_called()
        mt5.modify_order.assert_not_called()

    def test_tp1_closes_layer1_and_runner_goes_to_breakeven(self, mocker) -> None:
        setup, trades, broker = self._full_setup()
        mgr, mt5, trk = make_manager(mocker, setup, trades, broker)
        evs = mgr.check_live(bid=TP, ask=TP + 0.1)
        assert evs[0].action == "STARTED"
        mt5.close_position.assert_called_once_with(111)
        trk.update_trade_status.assert_called_once_with(
            trades[0].id, "CLOSED", close_reason="TP1", exit_price=TP)
        assert broker.pos[202]["sl"] == 1900.0          # runner at breakeven
        assert broker.pos[201]["sl"] == 1893.5          # main part left to its broker TP

    def test_trail_follows_4_points_behind_and_never_back(self, mocker) -> None:
        setup, trades, broker = self._full_setup()
        mgr, mt5, trk = make_manager(mocker, setup, trades, broker)
        mgr.check_live(bid=TP, ask=TP + 0.1)
        broker.pos.pop(201)                              # main part hit its broker TP
        trk.get_trades_for_setup.return_value = [
            trades[0].model_copy(update={"status": "CLOSED"}),
            trades[1].model_copy(update={"status": "CLOSED"}),
            trades[2],
        ]
        mgr.check_live(bid=1913.0, ask=1913.1)
        assert broker.pos[202]["sl"] == pytest.approx(1909.0)  # 1913 - 4
        mgr.check_live(bid=1916.0, ask=1916.1)
        assert broker.pos[202]["sl"] == pytest.approx(1912.0)
        mgr.check_live(bid=1914.0, ask=1914.1)          # pullback: SL doesn't move back
        assert broker.pos[202]["sl"] == pytest.approx(1912.0)
        mgr.check_live(bid=1920.05, ask=1920.15)
        assert broker.pos[202]["sl"] == pytest.approx(1916.05)

    def test_only_layer1_in_becomes_the_runner(self, mocker) -> None:
        setup = make_setup()
        l1 = make_trade(setup, layer=1, status="FILLED", ticket=111)
        l2 = make_trade(setup, layer=2, status="WAITING", ticket=None, order_type="STOP", lots="0.13")
        run = make_trade(setup, layer=3, status="WAITING", ticket=None, order_type="STOP", lots="0.04")
        broker = Broker([{"ticket": 111, "price_open": 1900.0, "sl": 1893.5, "tp": 0.0}])
        mgr, mt5, trk = make_manager(mocker, setup, [l1, l2, run], broker)
        evs = mgr.check_live(bid=TP, ask=TP + 0.1)
        assert evs[0].action == "STARTED"
        mt5.close_position.assert_not_called()          # the 0.01 keeps running
        assert broker.pos[111]["sl"] == 1900.0          # at breakeven
        cancelled = {c.args[0] for c in trk.update_trade_status.call_args_list
                     if c.args[1] == "CANCELLED"}
        assert cancelled == {l2.id, run.id}
        mgr.check_live(bid=1915.0, ask=1915.1)
        assert broker.pos[111]["sl"] == pytest.approx(1911.0)

    def test_sell_runner(self, mocker) -> None:
        setup = make_setup(direction="SELL", edge=1900.0, far=1905.0)
        setup = setup.model_copy(update={"planned_tp1_price": Decimal("1890")})
        l1 = make_trade(setup, layer=1, status="FILLED", ticket=111)
        broker = Broker([{"ticket": 111, "price_open": 1900.0, "sl": 1906.5, "tp": 0.0}])
        mgr, mt5, trk = make_manager(mocker, setup, [l1], broker)
        assert mgr.check_live(bid=1890.0, ask=1890.2) == []   # ask not at TP yet
        mgr.check_live(bid=1889.8, ask=1889.95)
        assert broker.pos[111]["sl"] == 1900.0
        mgr.check_live(bid=1884.9, ask=1885.0)
        assert broker.pos[111]["sl"] == pytest.approx(1889.0)

    def test_restart_recognises_a_started_runner(self, mocker) -> None:
        setup, trades, broker = self._full_setup()
        broker.pos.pop(111); broker.pos.pop(201)
        broker.pos[202]["sl"] = 1905.0                   # trail already above BE
        trades = [trades[0].model_copy(update={"status": "CLOSED"}),
                  trades[1].model_copy(update={"status": "CLOSED"}), trades[2]]
        mgr, mt5, trk = make_manager(mocker, setup, trades, broker)
        mgr.check_live(bid=1912.0, ask=1912.1)           # below TP: no "start" again
        mt5.close_position.assert_not_called()
        assert broker.pos[202]["sl"] == pytest.approx(1908.0)  # 1912 - 4, trail carries on

    def test_legacy_setup_with_broker_tp_is_ignored(self, mocker) -> None:
        setup = make_setup()
        l1 = make_trade(setup, layer=1, status="FILLED", ticket=111)
        broker = Broker([{"ticket": 111, "price_open": 1900.0, "sl": 1893.5, "tp": TP}])
        mgr, mt5, trk = make_manager(mocker, setup, [l1], broker)
        assert mgr.check_live(bid=TP, ask=TP + 0.1) == []
        mt5.modify_order.assert_not_called()

    def test_switched_off_does_nothing(self, mocker) -> None:
        setup, trades, broker = self._full_setup()
        mgr, mt5, trk = make_manager(mocker, setup, trades, broker)
        off = XautismEntryConfig(runner_enabled=False)
        assert mgr.check_live(bid=TP, ask=TP + 0.1, config=off) == []
        mt5.close_position.assert_not_called()
