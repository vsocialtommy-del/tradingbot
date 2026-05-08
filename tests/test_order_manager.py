"""Tests for ``bot.execution.order_manager``.

Strategy change (PR #15): Layers 2 and 3 are no longer placed at the
broker. ``order_manager`` writes them as Supabase rows with status
``WAITING``, and ``entry_trigger`` fires them when triggers are met.

Tests verify:
- Layer 1 still goes to MT5.
- ``place_limit_order`` is **never called**.
- Trade rows for Layers 2/3 have ``status="WAITING"``,
  ``mt5_ticket=None``, ``entry_price=None``.
- ``OrderPlacementResult`` returns ``layer_2_trade_id`` /
  ``layer_3_trade_id`` (Supabase UUIDs) instead of broker tickets.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.execution.mt5_connector import MT5Connector
from bot.execution.order_manager import (
    OrderManagerConfig,
    OrderPlacementResult,
    place_layered_orders,
)
from bot.logging.supabase_logger import SupabaseLogger
from bot.strategy.imbalance import ImbalanceZone
from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.structure import Swing
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Helpers — synthetic ImbalanceZone construction
# --------------------------------------------------------------------------- #


def make_imbalance_zone(
    *,
    direction: str = "BUY",
    top: float = 1900.0,
    bottom: float = 1895.0,
    is_strong_point: bool = True,
    is_imbalance: bool = True,
    is_tradeable: bool = True,
    rejection_reason: str | None = None,
) -> ImbalanceZone:
    ts = pd.Timestamp("2026-05-08T12:00:00Z")
    if direction == "BUY":
        pattern: Any = WPattern(
            low1=Swing(index=2, time=ts, price=top, kind="LOW"),
            low2=Swing(index=9, time=ts, price=top, kind="LOW"),
            peak_index=6, peak_time=ts, peak_price=top + 10,
            formed_at=ts, completed=True,
        )
    else:
        pattern = MPattern(
            high1=Swing(index=2, time=ts, price=bottom, kind="HIGH"),
            high2=Swing(index=9, time=ts, price=bottom, kind="HIGH"),
            trough_index=6, trough_time=ts, trough_price=bottom - 10,
            formed_at=ts, completed=True,
        )

    initial = Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
    )
    refined = RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=initial,
    )
    validated = ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=initial, refined_zone=refined,
        is_strong_point=is_strong_point, validation_failures=[], bos_event=None,
    )
    return ImbalanceZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=initial, refined_zone=refined,
        is_strong_point=is_strong_point, validation_failures=[], bos_event=None,
        validated_zone=validated,
        approach_count=2 if is_imbalance else 0, is_imbalance=is_imbalance,
        approach_events=[], qualified_at=ts if is_imbalance else None,
        is_tapped=False, tapped_at=None,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def setup_id() -> UUID:
    return uuid4()


@pytest.fixture
def zone_id() -> UUID:
    return uuid4()


@pytest.fixture
def layer_2_trade_id() -> UUID:
    return uuid4()


@pytest.fixture
def layer_3_trade_id() -> UUID:
    return uuid4()


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    """MT5Connector mock — no defaults on place_limit_order since it
    should never be called in the new model."""
    m = mocker.MagicMock(spec=MT5Connector)
    m.place_market_order.return_value = 11111
    m.get_open_positions.return_value = [
        {"ticket": 11111, "price_open": 1900.00},
    ]
    m.close_position.return_value = None
    return m


@pytest.fixture
def mock_supabase(
    mocker: MockerFixture,
    setup_id: UUID,
    layer_2_trade_id: UUID,
    layer_3_trade_id: UUID,
) -> MagicMock:
    """log_trade returns distinct IDs for Layers 1, 2, 3 in order.

    Layer 1 goes first (FILLED), then Layer 2 (WAITING), then Layer 3
    (WAITING). The result of the order_manager call uses the L2/L3 IDs
    in OrderPlacementResult.
    """
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.log_setup.return_value = {"id": str(setup_id)}
    layer_1_id = uuid4()
    m.log_trade.side_effect = [
        {"id": str(layer_1_id)},
        {"id": str(layer_2_trade_id)},
        {"id": str(layer_3_trade_id)},
    ]
    m.log_event.return_value = {"id": str(uuid4())}
    return m


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestHappyPath:
    def test_buy_layer_1_places_and_l2_l3_are_waiting_rows(
        self,
        mock_mt5: MagicMock,
        mock_supabase: MagicMock,
        zone_id: UUID,
        setup_id: UUID,
        layer_2_trade_id: UUID,
        layer_3_trade_id: UUID,
    ) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, lot_size=0.01, sl_price=1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.setup_id == setup_id
        assert result.layer_1_ticket == 11111
        # NEW MODEL: Layer 2/3 are Supabase trade IDs, not broker tickets.
        assert result.layer_2_trade_id == layer_2_trade_id
        assert result.layer_3_trade_id == layer_3_trade_id
        assert result.layer_1_filled_price == 1900.00
        assert result.error_messages == []
        assert result.tp1_price == 1904.0  # top + 4

    def test_sell_layer_1_places_and_l2_l3_are_waiting_rows(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
        layer_2_trade_id, layer_3_trade_id,
    ) -> None:
        zone = make_imbalance_zone(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, lot_size=0.01, sl_price=1925.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.layer_1_ticket == 11111
        assert result.layer_2_trade_id == layer_2_trade_id
        assert result.layer_3_trade_id == layer_3_trade_id
        assert result.tp1_price == 1896.0  # bottom - 4


class TestLayerOrderToBroker:
    """The headline behaviour change: only Layer 1 hits MT5."""

    def test_layer_2_and_3_NOT_sent_to_mt5(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        # The whole point of the refactor: place_limit_order is dead.
        mock_mt5.place_limit_order.assert_not_called()
        # Only ONE market call (Layer 1).
        assert mock_mt5.place_market_order.call_count == 1

    def test_only_layer_1_gets_a_broker_comment(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
    ) -> None:
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        comment = mock_mt5.place_market_order.call_args.kwargs["comment"]
        assert comment.startswith("bot:L1:s=")
        assert str(setup_id)[:8] in comment
        assert len(comment) <= 31


class TestCallOrder:
    def test_setup_then_market_then_trades(
        self, mock_mt5, mock_supabase, zone_id, mocker: MockerFixture,
    ) -> None:
        zone = make_imbalance_zone()
        parent = mocker.MagicMock()
        parent.attach_mock(mock_supabase.log_setup, "log_setup")
        parent.attach_mock(mock_mt5.place_market_order, "place_market")
        parent.attach_mock(mock_mt5.place_limit_order, "place_limit")
        parent.attach_mock(mock_supabase.log_trade, "log_trade")

        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )

        names = [c[0] for c in parent.mock_calls]
        # log_setup first.
        assert names[0] == "log_setup"
        # place_market next, before any log_trade.
        assert "place_market" in names
        assert "log_trade" in names
        market_idx = names.index("place_market")
        first_log_trade_idx = names.index("log_trade")
        assert market_idx < first_log_trade_idx
        # place_limit never appears.
        assert "place_limit" not in names


# --------------------------------------------------------------------------- #
# Pre-checks — fail before any side effects
# --------------------------------------------------------------------------- #


class TestPreChecks:
    def test_zone_not_tradeable_returns_failed_no_mt5_calls(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(
            is_tradeable=False, rejection_reason="ZONE_TOO_NARROW"
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert result.setup_id is None
        mock_mt5.place_market_order.assert_not_called()
        mock_mt5.place_limit_order.assert_not_called()
        mock_supabase.log_setup.assert_not_called()
        assert any("not tradeable" in m for m in result.error_messages)

    def test_neither_strong_point_nor_imbalance_returns_failed(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(
            is_strong_point=False, is_imbalance=False
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        mock_mt5.place_market_order.assert_not_called()

    def test_lot_size_zero_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.0, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        mock_mt5.place_market_order.assert_not_called()

    def test_sl_above_zone_top_for_buy_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1910.0,  # SL on wrong side
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        mock_mt5.place_market_order.assert_not_called()

    def test_sl_below_zone_bottom_for_sell_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1895.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestFailureModes:
    def test_layer_1_market_failure_returns_failed(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
    ) -> None:
        mock_mt5.place_market_order.side_effect = RuntimeError("broker error")
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        # Setup is still created (Supabase first), so id is set.
        assert result.setup_id == setup_id
        # No trade rows because Layer 1 failed; we never write L2/L3 either.
        mock_supabase.log_trade.assert_not_called()

    def test_supabase_log_setup_failure_no_mt5_calls(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        mock_supabase.log_setup.side_effect = RuntimeError("DB down")
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert result.setup_id is None
        mock_mt5.place_market_order.assert_not_called()


# --------------------------------------------------------------------------- #
# Gap-through detection
# --------------------------------------------------------------------------- #


class TestGapThrough:
    def test_buy_filled_below_zone_bottom_triggers_skip(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1893.0},
        ]
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "SKIPPED"
        assert result.layer_1_ticket == 11111
        assert result.layer_2_trade_id is None
        assert result.layer_3_trade_id is None
        mock_mt5.close_position.assert_called_once_with(11111)
        # No WAITING rows written — this setup is dead.
        mock_supabase.log_trade.assert_not_called()
        # Event logged for the dashboard.
        mock_supabase.log_event.assert_called()

    def test_sell_filled_above_zone_top_triggers_skip(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1908.0},
        ]
        zone = make_imbalance_zone(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1925.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "SKIPPED"
        mock_mt5.close_position.assert_called_once_with(11111)


# --------------------------------------------------------------------------- #
# Trade-row contents — the new WAITING semantics
# --------------------------------------------------------------------------- #


class TestTradeRowContents:
    def test_layer_1_row_is_filled_with_ticket_and_entry_price(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        # First log_trade call = Layer 1.
        layer_1_input = mock_supabase.log_trade.call_args_list[0].args[0]
        assert layer_1_input.layer_number == 1
        assert layer_1_input.status == "FILLED"
        assert layer_1_input.order_type == "MARKET"
        assert layer_1_input.mt5_ticket == 11111
        assert layer_1_input.entry_price is not None  # Decimal of fill price
        assert float(layer_1_input.entry_price) == 1900.00

    def test_layer_2_and_3_rows_are_waiting_with_no_ticket(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        # 3 log_trade calls total: L1 (FILLED), L2 (WAITING), L3 (WAITING).
        assert mock_supabase.log_trade.call_count == 3
        layer_2_input = mock_supabase.log_trade.call_args_list[1].args[0]
        layer_3_input = mock_supabase.log_trade.call_args_list[2].args[0]

        for ti in (layer_2_input, layer_3_input):
            assert ti.status == "WAITING"
            assert ti.order_type == "MARKET"
            assert ti.mt5_ticket is None
            assert ti.entry_price is None

        assert layer_2_input.layer_number == 2
        assert layer_3_input.layer_number == 3


# --------------------------------------------------------------------------- #
# TP1 / SL not on broker orders
# --------------------------------------------------------------------------- #


class TestNoTpOnBroker:
    def test_layer_1_market_order_has_no_tp(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        kwargs = mock_mt5.place_market_order.call_args.kwargs
        assert kwargs["tp"] is None
        # SL is set as the broker-side backstop.
        assert kwargs["sl"] == 1880.0


# --------------------------------------------------------------------------- #
# Setup record
# --------------------------------------------------------------------------- #


class TestSetupRecord:
    def test_imbalance_uses_imbalance_entry_mode(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(is_imbalance=True, is_strong_point=True)
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        setup_input = mock_supabase.log_setup.call_args.args[0]
        assert setup_input.entry_mode == "IMBALANCE_FIRST_TOUCH"

    def test_strong_point_only_uses_strong_point_entry_mode(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(is_imbalance=False, is_strong_point=True)
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        setup_input = mock_supabase.log_setup.call_args.args[0]
        assert setup_input.entry_mode == "STRONG_POINT_FIRST_TOUCH"

    def test_setup_record_planned_prices_match_zone_geometry(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        s = mock_supabase.log_setup.call_args.args[0]
        assert float(s.planned_layer1_price) == 1900.0  # zone top
        assert float(s.planned_layer2_price) == 1897.5  # midpoint
        assert float(s.planned_layer3_price) == 1895.0  # zone bottom
        assert float(s.planned_sl_price) == 1880.0
        assert float(s.planned_tp1_price) == 1904.0  # top + 4
        assert s.status == "PENDING"


# --------------------------------------------------------------------------- #
# Custom config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_custom_tp1_distance(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
            config=OrderManagerConfig(tp1_distance_dollars=10.0),
        )
        assert result.tp1_price == 1910.0  # top + 10
