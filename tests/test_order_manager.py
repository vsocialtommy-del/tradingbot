"""Tests for ``bot.execution.order_manager``.

This is the first execution-layer module with side effects, so the
testing pattern matters — later modules (position_tracker, tp1_manager,
sl_manager) should follow the same shape.

Pattern
-------
* ``mocker`` fixture (from pytest-mock) creates ``MagicMock(spec=…)``
  instances of the dependencies. ``spec=`` enforces that we only call
  methods that actually exist on the real class — catches API drift.
* Each test sets default ``return_value`` / ``side_effect`` on the
  mocks and asserts both the return value AND the call sequence.
* Synthetic :class:`ImbalanceZone` is built with the
  ``make_imbalance_zone`` helper — minimal valid object, no real
  Phase-B pipeline needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
from bot.strategy.pattern_detection import WPattern, MPattern
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone
from bot.strategy.strong_point import ValidatedZone


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
    """Build a synthetic ImbalanceZone with the minimum fields populated."""
    ts = pd.Timestamp("2026-05-08T12:00:00Z")

    # Build a minimal pattern based on direction.
    if direction == "BUY":
        pattern: Any = WPattern(
            low1=Swing(index=2, time=ts, price=top, kind="LOW"),
            low2=Swing(index=9, time=ts, price=top, kind="LOW"),
            peak_index=6,
            peak_time=ts,
            peak_price=top + 10,
            formed_at=ts,
            completed=True,
        )
    else:  # SELL
        pattern = MPattern(
            high1=Swing(index=2, time=ts, price=bottom, kind="HIGH"),
            high2=Swing(index=9, time=ts, price=bottom, kind="HIGH"),
            trough_index=6,
            trough_time=ts,
            trough_price=bottom - 10,
            formed_at=ts,
            completed=True,
        )

    initial = Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top,
        bottom=bottom,
        formed_at=ts,
        source_pattern=pattern,
    )
    refined = RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top,
        bottom=bottom,
        formed_at=ts,
        source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=initial,
    )
    validated = ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top,
        bottom=bottom,
        formed_at=ts,
        source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=initial,
        refined_zone=refined,
        is_strong_point=is_strong_point,
        validation_failures=[],
        bos_event=None,
    )
    return ImbalanceZone(
        direction=direction,  # type: ignore[arg-type]
        top=top,
        bottom=bottom,
        formed_at=ts,
        source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=initial,
        refined_zone=refined,
        is_strong_point=is_strong_point,
        validation_failures=[],
        bos_event=None,
        validated_zone=validated,
        approach_count=2 if is_imbalance else 0,
        is_imbalance=is_imbalance,
        approach_events=[],
        qualified_at=ts if is_imbalance else None,
        is_tapped=False,
        tapped_at=None,
    )


# --------------------------------------------------------------------------- #
# Mock fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def setup_id() -> UUID:
    return uuid4()


@pytest.fixture
def zone_id() -> UUID:
    return uuid4()


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    """MT5Connector mock with sane defaults for a happy-path BUY at 1900."""
    m = mocker.MagicMock(spec=MT5Connector)
    m.place_market_order.return_value = 11111
    m.place_limit_order.side_effect = [22222, 33333]
    m.get_open_positions.return_value = [
        {"ticket": 11111, "price_open": 1900.00},
    ]
    m.close_position.return_value = None
    return m


@pytest.fixture
def mock_supabase(mocker: MockerFixture, setup_id: UUID) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.log_setup.return_value = {"id": str(setup_id)}
    m.log_trade.return_value = {"id": str(uuid4())}
    m.log_event.return_value = {"id": str(uuid4())}
    return m


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #


class TestHappyPath:
    def test_buy_all_three_layers_placed(
        self,
        mock_mt5: MagicMock,
        mock_supabase: MagicMock,
        zone_id: UUID,
        setup_id: UUID,
    ) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id,
            lot_size=0.01,
            sl_price=1880.0,
            mt5=mock_mt5,
            supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.setup_id == setup_id
        assert result.layer_1_ticket == 11111
        assert result.layer_2_ticket == 22222
        assert result.layer_3_ticket == 33333
        assert result.layer_1_filled_price == 1900.00
        assert result.error_messages == []
        # tp1 = top + 4 = 1904
        assert result.tp1_price == 1904.0

    def test_sell_all_three_layers_placed(
        self,
        mock_mt5: MagicMock,
        mock_supabase: MagicMock,
        zone_id: UUID,
    ) -> None:
        # Override mt5 so the filled price is in-zone for SELL.
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1900.00},
        ]
        zone = make_imbalance_zone(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id,
            lot_size=0.01,
            sl_price=1925.0,
            mt5=mock_mt5,
            supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        # tp1 = bottom - 4 = 1896
        assert result.tp1_price == 1896.0

    def test_call_order_setup_then_market_then_limits_then_trades(
        self, mock_mt5, mock_supabase, zone_id, mocker: MockerFixture,
    ) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        # Use a parent mock to record the call order across mocks.
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
        # log_setup must come first.
        assert names[0] == "log_setup"
        # First MT5 call after setup is the market order.
        assert "place_market" in names
        market_idx = names.index("place_market")
        setup_idx = names.index("log_setup")
        assert setup_idx < market_idx
        # Limit orders come after market.
        for limit_idx in [i for i, n in enumerate(names) if n == "place_limit"]:
            assert market_idx < limit_idx
        # log_trade comes last.
        last_log_trade = max(i for i, n in enumerate(names) if n == "log_trade")
        for limit_idx in [i for i, n in enumerate(names) if n == "place_limit"]:
            assert limit_idx < last_log_trade


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
        # No MT5 or Supabase calls.
        mock_mt5.place_market_order.assert_not_called()
        mock_mt5.place_limit_order.assert_not_called()
        mock_supabase.log_setup.assert_not_called()
        mock_supabase.log_trade.assert_not_called()
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
        mock_supabase.log_setup.assert_not_called()

    def test_lot_size_zero_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.0, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert any("lot_size" in m for m in result.error_messages)
        mock_mt5.place_market_order.assert_not_called()

    def test_lot_size_negative_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, -0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        mock_mt5.place_market_order.assert_not_called()

    def test_sl_above_zone_top_for_buy_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # BUY: SL must be < zone.top (= 1900 below). Test SL above.
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1910.0,  # SL above zone — wrong side
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert any("below zone.top" in m for m in result.error_messages)
        mock_mt5.place_market_order.assert_not_called()

    def test_sl_below_zone_bottom_for_sell_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1895.0,  # SL below zone — wrong side
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert any("above zone.bottom" in m for m in result.error_messages)
        mock_mt5.place_market_order.assert_not_called()


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
        # Setup record IS created (Supabase first), so setup_id is set.
        assert result.setup_id == setup_id
        # No limit orders attempted, no trade records.
        mock_mt5.place_limit_order.assert_not_called()
        mock_supabase.log_trade.assert_not_called()
        assert any("Layer 1" in m for m in result.error_messages)

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
        # NEVER reached MT5.
        mock_mt5.place_market_order.assert_not_called()
        mock_mt5.place_limit_order.assert_not_called()

    def test_layer_2_failure_returns_partial(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Layer 2 fails, Layer 3 succeeds.
        mock_mt5.place_limit_order.side_effect = [
            RuntimeError("requote on layer 2"),
            33333,
        ]
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PARTIAL"
        assert result.layer_1_ticket == 11111
        assert result.layer_2_ticket is None
        assert result.layer_3_ticket == 33333
        assert any("Layer 2" in m for m in result.error_messages)

    def test_layer_3_failure_returns_partial(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        mock_mt5.place_limit_order.side_effect = [
            22222,
            RuntimeError("price off layer 3"),
        ]
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PARTIAL"
        assert result.layer_2_ticket == 22222
        assert result.layer_3_ticket is None

    def test_supabase_log_trade_failure_does_not_change_status(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Trade rows fail to write but orders are on broker — best-effort.
        mock_supabase.log_trade.side_effect = RuntimeError("DB blip")
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        # Status stays PLACED — the broker has the orders, the bookkeeping
        # just failed and is logged for reconciliation.
        assert result.status == "PLACED"
        assert any("trade record" in m for m in result.error_messages)


# --------------------------------------------------------------------------- #
# Gap-through detection
# --------------------------------------------------------------------------- #


class TestGapThrough:
    def test_buy_filled_below_zone_bottom_triggers_skip(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Zone is [1895, 1900]; tolerance default 0.05; threshold 1894.95.
        # Filled at 1893 → way below → gap-through.
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
        assert result.layer_1_filled_price == 1893.0
        assert result.layer_2_ticket is None
        assert result.layer_3_ticket is None
        # Layer 1 was closed.
        mock_mt5.close_position.assert_called_once_with(11111)
        # No limit orders attempted.
        mock_mt5.place_limit_order.assert_not_called()
        # Event logged for the dashboard.
        mock_supabase.log_event.assert_called()

    def test_sell_filled_above_zone_top_triggers_skip(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Zone is [1900, 1905]; threshold 1905.05.
        # Filled at 1908 → above → gap-through.
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

    def test_filled_within_tolerance_does_not_trigger_skip(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Filled exactly at zone.bottom — within tolerance.
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1895.0},
        ]
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        # Inside zone — proceeds normally.
        assert result.status == "PLACED"

    def test_unable_to_resolve_filled_price_skips_gap_check(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # If positions_get returns empty (e.g. position closed instantly
        # by SL), we can't compute fill price — skip gap check, continue.
        mock_mt5.get_open_positions.return_value = []
        zone = make_imbalance_zone()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        # No gap-through triggered (we couldn't check).
        assert result.status == "PLACED"
        mock_mt5.close_position.assert_not_called()


# --------------------------------------------------------------------------- #
# Layer prices and TP1
# --------------------------------------------------------------------------- #


class TestLayerPrices:
    def test_buy_layer_prices(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        # Layer 2 = midpoint = 1897.5; Layer 3 = bottom = 1895.
        limit_calls = mock_mt5.place_limit_order.call_args_list
        assert limit_calls[0].kwargs["price"] == 1897.5  # L2
        assert limit_calls[1].kwargs["price"] == 1895.0  # L3

    def test_sell_layer_prices(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1900.00},
        ]
        zone = make_imbalance_zone(direction="SELL", top=1905, bottom=1900)
        place_layered_orders(
            zone, zone_id, 0.01, 1925.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        limit_calls = mock_mt5.place_limit_order.call_args_list
        # Layer 2 = midpoint 1902.5; Layer 3 = far edge = top = 1905.
        assert limit_calls[0].kwargs["price"] == 1902.5
        assert limit_calls[1].kwargs["price"] == 1905.0

    def test_tp1_buy(self, mock_mt5, mock_supabase, zone_id) -> None:
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.tp1_price == 1904.0  # top + default 4

    def test_tp1_sell(self, mock_mt5, mock_supabase, zone_id) -> None:
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1900.00},
        ]
        zone = make_imbalance_zone(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1925.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.tp1_price == 1896.0  # bottom - default 4

    def test_no_tp_set_on_mt5_orders(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Per design: TP1 is bot-managed, never on the broker order.
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        market_call = mock_mt5.place_market_order.call_args
        assert market_call.kwargs["tp"] is None
        for limit_call in mock_mt5.place_limit_order.call_args_list:
            assert limit_call.kwargs["tp"] is None


# --------------------------------------------------------------------------- #
# Comments include short setup_id
# --------------------------------------------------------------------------- #


class TestComments:
    def test_market_order_comment_includes_setup_id(
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

    def test_limit_order_comments_include_layer_and_setup_id(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
    ) -> None:
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        comments = [
            c.kwargs["comment"]
            for c in mock_mt5.place_limit_order.call_args_list
        ]
        assert comments[0].startswith("bot:L2:s=")
        assert comments[1].startswith("bot:L3:s=")
        for c in comments:
            assert str(setup_id)[:8] in c
        # All comments are <= MT5's 31-char limit.
        for c in comments:
            assert len(c) <= 31


# --------------------------------------------------------------------------- #
# Entry-mode and Supabase records
# --------------------------------------------------------------------------- #


class TestSupabaseRecords:
    def test_imbalance_zone_uses_imbalance_entry_mode(
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
        zone = make_imbalance_zone(
            is_imbalance=False, is_strong_point=True
        )
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        setup_input = mock_supabase.log_setup.call_args.args[0]
        assert setup_input.entry_mode == "STRONG_POINT_FIRST_TOUCH"

    def test_three_trade_records_written_on_happy_path(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert mock_supabase.log_trade.call_count == 3
        # Layer 1 → FILLED with entry_price; Layers 2/3 → PENDING with no entry.
        trade_inputs = [
            call.args[0] for call in mock_supabase.log_trade.call_args_list
        ]
        statuses = [t.status for t in trade_inputs]
        order_types = [t.order_type for t in trade_inputs]
        assert statuses == ["FILLED", "PENDING", "PENDING"]
        assert order_types == ["MARKET", "LIMIT", "LIMIT"]
        # Layer 1 has entry_price set; Layers 2/3 don't.
        assert trade_inputs[0].entry_price is not None
        assert trade_inputs[1].entry_price is None
        assert trade_inputs[2].entry_price is None

    def test_setup_record_planned_prices(
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

    def test_custom_gap_tolerance(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Filled $0.50 below zone — fails default $0.05 tolerance, but
        # passes if we widen to $1.00.
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1894.50},
        ]
        zone = make_imbalance_zone(direction="BUY", top=1900, bottom=1895)
        result_strict = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result_strict.status == "SKIPPED"

        # Reset side_effect for fresh limit orders.
        mock_mt5.place_limit_order.side_effect = [22222, 33333]
        mock_mt5.place_market_order.return_value = 11111
        mock_mt5.close_position.reset_mock()

        result_lenient = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
            config=OrderManagerConfig(gap_tolerance_dollars=1.0),
        )
        assert result_lenient.status == "PLACED"
