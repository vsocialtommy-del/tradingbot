"""Tests for ``bot.execution.order_manager``.

Two methodology updates land in this test file:

* PR #15 — Layers 2/3 are no longer placed at the broker; they're
  written as Supabase rows with status ``WAITING`` and fired later
  by ``entry_trigger``.
* Loosened-rules PR (May 2026) — ``place_layered_orders`` now takes
  ``tp1_price`` as a required argument. The TP1 source moved upstream
  to ``main._try_place_setup`` (via :mod:`bot.strategy.tp1_target`),
  and the BOS_LEVEL / FIXED_DISTANCE machinery in this module is
  gone. The pre-checks now also enforce that ``tp1_price`` is on the
  favourable side of Layer 1's entry — a defensive sanity check, not
  a distance filter.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.execution.mt5_connector import MT5Connector
from bot.execution.order_manager import (
    OrderManagerConfig,
    place_layered_orders,
)
from bot.logging.supabase_logger import SupabaseLogger
from bot.strategy.pattern_detection import (
    Base,
    Impulse,
    Pattern,
    PatternType,
)
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Helpers — synthetic ValidatedZone construction
# --------------------------------------------------------------------------- #


def make_zone_for_test(
    *,
    direction: str = "BUY",
    top: float = 1900.0,
    bottom: float = 1895.0,
    is_strong_point: bool = True,
    is_tradeable: bool = True,
    rejection_reason: str | None = None,
) -> ValidatedZone:
    """Build a ValidatedZone for tests.

    Under the loosened rules, ``broken_swing`` / ``broken_at`` /
    ``sl_anchor_swing`` are always None on a real ValidatedZone — so
    we set them None here too. ``is_strong_point`` defaults to True;
    tests that exercise pre-check failure modes flip ``is_tradeable``
    or ``is_strong_point``.
    """
    ts = pd.Timestamp("2026-05-08T12:00:00Z")
    impulse = Impulse(
        direction="RALLY" if direction == "BUY" else "DROP",
        start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=5.0, largest_body=5.0, candle_count=1,
    )
    base = Base(
        start_index=1, end_index=1, candle_count=1,
        top=top, bottom=bottom, range_size=top - bottom, largest_body=0.5,
    )
    pattern = Pattern(
        pattern_type=PatternType.RBR if direction == "BUY" else PatternType.DBD,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction=direction,  # type: ignore[arg-type]
        formed_at=ts,
    )
    zone = Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
    )
    refined = RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=zone,
    )
    return ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        refined_zone=refined,
        is_strong_point=is_strong_point,
        validation_failures=[],
        broken_swing=None,
        broken_at=None,
        sl_anchor_swing=None,
    )


# Default TP1 prices that match the default zone (1895-1900 BUY / 1900-1906 SELL).
DEFAULT_BUY_TP1 = 1910.0   # above zone top
DEFAULT_SELL_TP1 = 1890.0  # below zone bottom


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def setup_id() -> UUID:
    return UUID("11111111-2222-3333-4444-555555555555")


@pytest.fixture
def zone_id() -> UUID:
    return uuid4()


@pytest.fixture
def layer_2_trade_id() -> UUID:
    return UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


@pytest.fixture
def layer_3_trade_id() -> UUID:
    return UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=MT5Connector)
    m.place_market_order.return_value = 11111  # ticket
    m.get_open_positions.return_value = [
        {"ticket": 11111, "price_open": 1900.00},
    ]
    return m


@pytest.fixture
def mock_supabase(
    mocker: MockerFixture,
    setup_id: UUID,
    layer_2_trade_id: UUID,
    layer_3_trade_id: UUID,
) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.log_setup.return_value = {"id": str(setup_id)}
    m.log_trade.side_effect = [
        {"id": str(uuid4())},
        {"id": str(layer_2_trade_id)},
        {"id": str(layer_3_trade_id)},
    ]
    return m


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


class TestHappyPath:
    def test_buy_layer_1_places_and_l2_l3_are_waiting_rows(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
        layer_2_trade_id, layer_3_trade_id,
    ) -> None:
        zone = make_zone_for_test(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, lot_size=0.01,
            sl_price=1880.0, tp1_price=DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.setup_id == setup_id
        assert result.layer_1_ticket == 11111
        assert result.layer_2_trade_id == layer_2_trade_id
        assert result.layer_3_trade_id == layer_3_trade_id
        assert result.layer_1_filled_price == 1900.00
        assert result.error_messages == []
        assert result.tp1_price == DEFAULT_BUY_TP1

    def test_sell_layer_1_places_and_l2_l3_are_waiting_rows(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
        layer_2_trade_id, layer_3_trade_id,
    ) -> None:
        # SELL: fill below zone top, TP1 below zone bottom.
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1905.00},
        ]
        zone = make_zone_for_test(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, lot_size=0.01,
            sl_price=1925.0, tp1_price=DEFAULT_SELL_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.layer_1_ticket == 11111
        assert result.tp1_price == DEFAULT_SELL_TP1


class TestLayerOrderToBroker:
    """The headline behaviour: only Layer 1 hits MT5."""

    def test_layer_2_and_3_NOT_sent_to_mt5(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        mock_mt5.place_limit_order.assert_not_called()
        assert mock_mt5.place_market_order.call_count == 1

    def test_only_layer_1_gets_a_broker_comment(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
    ) -> None:
        zone = make_zone_for_test()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
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
        zone = make_zone_for_test()
        parent = mocker.MagicMock()
        parent.attach_mock(mock_supabase.log_setup, "log_setup")
        parent.attach_mock(mock_mt5.place_market_order, "place_market")
        parent.attach_mock(mock_mt5.place_limit_order, "place_limit")
        parent.attach_mock(mock_supabase.log_trade, "log_trade")

        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )

        names = [c[0] for c in parent.mock_calls]
        assert names[0] == "log_setup"
        assert "place_market" in names
        assert "log_trade" in names
        market_idx = names.index("place_market")
        first_log_trade_idx = names.index("log_trade")
        assert market_idx < first_log_trade_idx
        assert "place_limit" not in names


# --------------------------------------------------------------------------- #
# Pre-checks — fail before any side effects
# --------------------------------------------------------------------------- #


class TestPreChecks:
    def test_zone_not_tradeable_returns_failed_no_mt5_calls(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test(
            is_tradeable=False, rejection_reason="ZONE_TOO_NARROW",
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert result.setup_id is None
        mock_mt5.place_market_order.assert_not_called()
        mock_supabase.log_setup.assert_not_called()
        assert any("not tradeable" in m for m in result.error_messages)

    def test_not_strong_point_returns_failed(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test(is_strong_point=False)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        mock_mt5.place_market_order.assert_not_called()

    def test_lot_size_zero_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test()
        result = place_layered_orders(
            zone, zone_id, 0.0, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        mock_mt5.place_market_order.assert_not_called()

    def test_sl_above_zone_top_for_buy_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1910.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"

    def test_sl_below_zone_bottom_for_sell_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1895.0, DEFAULT_SELL_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"

    def test_tp1_below_zone_top_for_buy_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Loosened-rules pre-check: TP1 must be above zone.top for BUY.
        # Caller (main._try_place_setup) skips zones with no qualifying
        # peak before reaching here, so this guards against bugs in the
        # caller.
        zone = make_zone_for_test(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0, tp1_price=1895.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert any("TP1" in m for m in result.error_messages)
        mock_mt5.place_market_order.assert_not_called()

    def test_tp1_above_zone_bottom_for_sell_rejected(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1925.0, tp1_price=1905.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert any("TP1" in m for m in result.error_messages)


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #


class TestFailureModes:
    def test_layer_1_market_failure_returns_failed(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
    ) -> None:
        mock_mt5.place_market_order.side_effect = RuntimeError("broker error")
        zone = make_zone_for_test()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert result.setup_id == setup_id
        mock_supabase.log_trade.assert_not_called()

    def test_supabase_log_setup_failure_no_mt5_calls(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        mock_supabase.log_setup.side_effect = RuntimeError("DB down")
        zone = make_zone_for_test()
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
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
        zone = make_zone_for_test(direction="BUY", top=1900, bottom=1895)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "SKIPPED"
        assert result.layer_1_ticket == 11111
        assert result.layer_2_trade_id is None
        assert result.layer_3_trade_id is None
        mock_mt5.close_position.assert_called_once_with(11111)
        mock_supabase.log_trade.assert_not_called()
        mock_supabase.log_event.assert_called()

    def test_sell_filled_above_zone_top_triggers_skip(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1908.0},
        ]
        zone = make_zone_for_test(direction="SELL", top=1905, bottom=1900)
        result = place_layered_orders(
            zone, zone_id, 0.01, 1925.0, DEFAULT_SELL_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "SKIPPED"
        mock_mt5.close_position.assert_called_once_with(11111)


# --------------------------------------------------------------------------- #
# Trade-row contents — the WAITING semantics
# --------------------------------------------------------------------------- #


class TestTradeRowContents:
    def test_layer_1_row_is_filled_with_ticket_and_entry_price(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        layer_1_input = mock_supabase.log_trade.call_args_list[0].args[0]
        assert layer_1_input.layer_number == 1
        assert layer_1_input.status == "FILLED"
        assert layer_1_input.order_type == "MARKET"
        assert layer_1_input.mt5_ticket == 11111
        assert layer_1_input.entry_price is not None
        assert float(layer_1_input.entry_price) == 1900.00

    def test_layer_2_and_3_rows_are_waiting_with_no_ticket(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
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
        zone = make_zone_for_test()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        kwargs = mock_mt5.place_market_order.call_args.kwargs
        assert kwargs["tp"] is None
        assert kwargs["sl"] == 1880.0


# --------------------------------------------------------------------------- #
# Setup record
# --------------------------------------------------------------------------- #


class TestSetupRecord:
    def test_strong_point_uses_strong_point_entry_mode(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test()
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, DEFAULT_BUY_TP1,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        setup_input = mock_supabase.log_setup.call_args.args[0]
        assert setup_input.entry_mode == "STRONG_POINT_FIRST_TOUCH"

    def test_setup_record_planned_prices_match_zone_geometry(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_zone_for_test(direction="BUY", top=1900, bottom=1895)
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0, tp1_price=1908.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        s = mock_supabase.log_setup.call_args.args[0]
        assert float(s.planned_layer1_price) == 1900.0  # zone top
        assert float(s.planned_layer2_price) == 1897.5  # midpoint
        assert float(s.planned_layer3_price) == 1895.0  # zone bottom
        assert float(s.planned_sl_price) == 1880.0
        assert float(s.planned_tp1_price) == 1908.0
        assert s.status == "PENDING"

    def test_tp1_passes_through_to_setup_row(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Different TP1 each time → setup row reflects it. No internal
        # computation that could overwrite the caller's choice.
        for tp1 in (1905.0, 1925.5, 1962.0):
            mock_supabase.reset_mock()
            mock_supabase.log_setup.return_value = {"id": str(uuid4())}
            mock_supabase.log_trade.side_effect = [
                {"id": str(uuid4())} for _ in range(3)
            ]
            zone = make_zone_for_test(direction="BUY", top=1900, bottom=1895)
            place_layered_orders(
                zone, zone_id, 0.01, 1880.0, tp1_price=tp1,
                mt5=mock_mt5, supabase=mock_supabase,
            )
            s = mock_supabase.log_setup.call_args.args[0]
            assert float(s.planned_tp1_price) == tp1


# --------------------------------------------------------------------------- #
# Config — what survives the deletion
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_default_config_has_no_tp1_method(self) -> None:
        # TP1Method / tp1_distance_dollars are gone; sanity-check the
        # config surface is the minimum still required.
        cfg = OrderManagerConfig()
        assert cfg.symbol == "XAUUSD"
        assert cfg.gap_tolerance_dollars == pytest.approx(0.05)
        assert not hasattr(cfg, "tp1_method")
        assert not hasattr(cfg, "tp1_distance_dollars")
