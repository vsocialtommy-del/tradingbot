"""Stacked (two-zone) layering: first zone + a fresh zone just past it."""

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
from bot.logging.supabase_logger import SupabaseLogger
from bot.strategy.stacked import StackedConfig, find_outer_zone, stacked_plan

from tests.test_order_manager import make_zone_for_test
from tests.test_xautism_entries import make_setup, make_trade


def z(direction: str, bottom: float, top: float, **kw):
    return make_zone_for_test(direction=direction, top=top, bottom=bottom, **kw)


class TestFindOuterZone:
    def test_29_oct_sell_example(self) -> None:
        first = z("SELL", 4018.0, 4025.5)
        m1 = z("SELL", 4030.4, 4033.7)
        plan = stacked_plan(first, [("M1", m1)])
        assert plan is not None
        assert plan.outer_timeframe == "M1"
        assert plan.gap_points == pytest.approx(4.9)
        assert plan.outer_entry == 4030.4           # sell limit at its bottom
        assert plan.outer_far_edge == 4033.7
        assert plan.sl == pytest.approx(4034.69)    # 30% of 3.3 above 4033.7

    def test_buy_mirror(self) -> None:
        first = z("BUY", 3995.0, 4000.0)
        outer = z("BUY", 3985.0, 3990.0)
        plan = stacked_plan(first, [("M5", outer)])
        assert plan is not None
        assert plan.outer_entry == 3990.0
        assert plan.sl == pytest.approx(3983.5)     # 30% of 5 below 3985

    def test_gap_over_10_points_is_not_stacked(self) -> None:
        first = z("SELL", 4018.0, 4025.5)
        far = z("SELL", 4035.6, 4038.0)             # gap 10.1
        assert find_outer_zone(first, [("M5", far)]) is None

    def test_gap_of_exactly_10_counts(self) -> None:
        first = z("SELL", 4018.0, 4025.5)
        assert find_outer_zone(first, [("M5", z("SELL", 4035.5, 4038.0))]) is not None

    def test_ignores_wrong_side_overlap_direction_and_stale(self) -> None:
        first = z("SELL", 4018.0, 4025.5)
        cands = [
            ("M5", z("SELL", 4010.0, 4015.0)),       # below a sell: wrong side
            ("M1", z("SELL", 4024.0, 4028.0)),       # overlaps the first zone
            ("M1", z("BUY", 4027.0, 4030.0)),        # other direction
            ("M5", z("SELL", 4027.0, 4030.0, is_strong_point=False)),  # not fresh
            ("M5", first),                           # itself
        ]
        assert find_outer_zone(first, cands) is None

    def test_nearest_wins_and_ties_go_to_m5(self) -> None:
        first = z("BUY", 3995.0, 4000.0)
        near_m1 = z("BUY", 3990.0, 3993.0)          # gap 2
        tie_m5 = z("BUY", 3989.0, 3993.0)           # gap 2
        far_m5 = z("BUY", 3985.0, 3988.0)           # gap 7
        tf, outer, gap = find_outer_zone(first, [("M1", near_m1), ("M5", far_m5), ("M5", tie_m5)])
        assert gap == 2.0 and tf == "M5" and outer is tie_m5

    def test_custom_gap(self) -> None:
        first = z("BUY", 3995.0, 4000.0)
        outer = z("BUY", 3985.0, 3990.0)            # gap 5
        assert find_outer_zone(first, [("M5", outer)], StackedConfig(max_gap_points=3)) is None


# --------------------------------------------------------------------------- #


@pytest.fixture
def mt5(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=MT5Connector)
    m.place_market_order.return_value = 111
    m.get_open_positions.return_value = [{"ticket": 111, "price_open": 4018.0}]
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


def test_stacked_orders(mt5, supa, tracker) -> None:
    first = z("SELL", 4018.0, 4025.5)
    plan = stacked_plan(first, [("M1", z("SELL", 4030.4, 4033.7))])
    r = place_xautism_orders(first, uuid4(), total_lots=0.05,
                             sl_price=plan.sl, tp1_price=4012.87,
                             stacked=plan, mt5=mt5, supabase=supa, tracker=tracker)
    assert r.status == "PLACED"
    kw = mt5.place_market_order.call_args.kwargs
    assert kw["lot_size"] == 0.01 and kw["sl"] == pytest.approx(4034.69)
    l2 = supa.log_trade.call_args_list[1].args[0]
    assert l2.order_type == "LIMIT" and l2.lot_size == Decimal("0.04")
    assert l2.sl_price == Decimal("4034.69")
    s = supa.log_setup.call_args.args[0]
    assert s.planned_layer1_price == Decimal("4018.0")
    assert s.planned_layer2_price == Decimal("4030.4")
    assert s.planned_layer3_price == Decimal("4033.7")   # M5 exit uses the outer zone


def test_layer2_limit_placed_at_outer_zone(mt5, supa, tracker) -> None:
    mt5.place_limit_order.return_value = 333
    s = make_setup(direction="SELL", edge=4018.0, far=4033.7)
    s = s.model_copy(update={"planned_layer2_price": Decimal("4030.4")})
    tracker.get_active_setups.return_value = [s]
    l2 = make_trade(s, layer=2, status="WAITING", ticket=None,
                    order_type="LIMIT", lots="0.04")
    supa.get_trades_for_setup.return_value = [
        make_trade(s, layer=1, status="FILLED", ticket=111), l2,
    ]
    m = Layer2StopManager(mt5, supa, tracker, XautismEntryConfig())
    ev = m.check_live(4018.2, 4018.5)    # price at the first zone, below the limit
    assert ev[0].action == "PLACED"
    kw = mt5.place_limit_order.call_args.kwargs
    assert kw["direction"] == "SELL" and kw["price"] == 4030.4
    mt5.place_stop_order.assert_not_called()
