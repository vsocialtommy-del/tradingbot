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
from bot.strategy.pattern_detection import (
    Base,
    Impulse,
    Pattern,
    PatternType,
)
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Helpers — synthetic ValidatedZone construction (PR #31)
# --------------------------------------------------------------------------- #


def make_imbalance_zone(
    *,
    direction: str = "BUY",
    top: float = 1900.0,
    bottom: float = 1895.0,
    is_strong_point: bool = True,
    is_tradeable: bool = True,
    rejection_reason: str | None = None,
    bos_broken_level: float | None = None,
    is_imbalance: bool | None = None,  # accepted for back-compat; ignored
) -> ValidatedZone:
    """Build a ValidatedZone for tests.

    Name retained from the pre-PR-31 helper so the call sites in this
    file don't churn — returns a ``ValidatedZone`` (the post-PR-31 type)
    rather than the old ``ImbalanceZone``. The ``is_imbalance`` kwarg
    is accepted (and ignored) for backward signature compat.

    ``bos_broken_level`` controls the broken_swing price (the
    structural high/low the Strong Point body-closed past). Default:
    ``top + 5`` for BUY / ``bottom - 5`` for SELL. Tests that need
    a specific BOS level pass an explicit value.
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
    if bos_broken_level is None:
        bos_broken_level = top + 5.0 if direction == "BUY" else bottom - 5.0
    broken = Swing(
        index=12, time=ts, price=bos_broken_level,
        kind="HIGH" if direction == "BUY" else "LOW",
    )
    anchor = Swing(
        index=11, time=ts,
        price=bottom - 5.0 if direction == "BUY" else top + 5.0,
        kind="LOW" if direction == "BUY" else "HIGH",
    )
    return ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        refined_zone=refined,
        is_strong_point=is_strong_point,
        validation_failures=[],
        broken_swing=broken if is_strong_point else None,
        broken_at=ts if is_strong_point else None,
        sl_anchor_swing=anchor,
    )


def make_imbalance_zone_no_bos(**kwargs: Any) -> ValidatedZone:
    """ValidatedZone with broken_swing=None.

    Used only for the defensive test that the BOS_LEVEL pre-check
    fails cleanly when validation has been bypassed. In production
    every zone reaching ``order_manager`` has a broken_swing because
    ``strong_point`` guarantees it; this exists purely to verify the
    guard.
    """
    z = make_imbalance_zone(**kwargs)
    return ValidatedZone(
        direction=z.direction, top=z.top, bottom=z.bottom,
        formed_at=z.formed_at, source_pattern=z.source_pattern,
        refined_zone=z.refined_zone,
        is_strong_point=z.is_strong_point,
        validation_failures=z.validation_failures,
        broken_swing=None,
        broken_at=None,
        sl_anchor_swing=z.sl_anchor_swing,
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
        zone = make_imbalance_zone(
            direction="BUY", top=1900, bottom=1895,
            bos_broken_level=1907.0,  # the swing high broken by the impulse
        )
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
        # BOS_LEVEL (default): TP1 = bos_event.broken_level.
        assert result.tp1_price == 1907.0

    def test_sell_layer_1_places_and_l2_l3_are_waiting_rows(
        self, mock_mt5, mock_supabase, zone_id, setup_id,
        layer_2_trade_id, layer_3_trade_id,
    ) -> None:
        zone = make_imbalance_zone(
            direction="SELL", top=1905, bottom=1900,
            bos_broken_level=1893.0,  # the swing low broken by the impulse
        )
        result = place_layered_orders(
            zone, zone_id, lot_size=0.01, sl_price=1925.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.layer_1_ticket == 11111
        assert result.layer_2_trade_id == layer_2_trade_id
        assert result.layer_3_trade_id == layer_3_trade_id
        # BOS_LEVEL (default): TP1 = bos_event.broken_level.
        assert result.tp1_price == 1893.0


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
    # NOTE: pre-PR-31 this file had a
    # ``test_imbalance_uses_imbalance_entry_mode`` test verifying that
    # an Imbalance-flagged zone produced ``IMBALANCE_FIRST_TOUCH`` as
    # the entry_mode on the setup record. PR #31 removed the Imbalance
    # setup from v1 — entry_mode is now always ``STRONG_POINT_FIRST_TOUCH``.
    # When Setup 4 (Imbalance) is rebuilt later we'll add a discriminator
    # field on the validated zone and re-introduce the dispatch test.

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
        zone = make_imbalance_zone(
            direction="BUY", top=1900, bottom=1895,
            bos_broken_level=1908.0,
        )
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        s = mock_supabase.log_setup.call_args.args[0]
        assert float(s.planned_layer1_price) == 1900.0  # zone top
        assert float(s.planned_layer2_price) == 1897.5  # midpoint
        assert float(s.planned_layer3_price) == 1895.0  # zone bottom
        assert float(s.planned_sl_price) == 1880.0
        # BOS_LEVEL (default): TP1 = bos_event.broken_level.
        assert float(s.planned_tp1_price) == 1908.0
        assert s.status == "PENDING"


# --------------------------------------------------------------------------- #
# Custom config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_default_method_is_bos_level(self) -> None:
        # Sanity: default constructor selects BOS_LEVEL, the strategy default.
        cfg = OrderManagerConfig()
        assert cfg.tp1_method == "BOS_LEVEL"

    def test_fixed_distance_uses_legacy_calculation(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # FIXED_DISTANCE remains usable for backtests. It ignores
        # bos_event entirely — even with a BoS at 1907, the legacy path
        # uses zone_top + tp1_distance_dollars.
        zone = make_imbalance_zone(
            direction="BUY", top=1900, bottom=1895,
            bos_broken_level=1907.0,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
            config=OrderManagerConfig(
                tp1_method="FIXED_DISTANCE", tp1_distance_dollars=10.0,
            ),
        )
        # 1900 (top) + 10 = 1910 — the BoS at 1907 is intentionally ignored.
        assert result.tp1_price == 1910.0

    def test_fixed_distance_buy_default_4(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(
            direction="BUY", top=1900, bottom=1895,
            bos_broken_level=1907.0,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
            config=OrderManagerConfig(tp1_method="FIXED_DISTANCE"),
        )
        assert result.tp1_price == 1904.0  # top + 4 default

    def test_fixed_distance_sell_default_4(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        zone = make_imbalance_zone(
            direction="SELL", top=1905, bottom=1900,
            bos_broken_level=1893.0,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1925.0,
            mt5=mock_mt5, supabase=mock_supabase,
            config=OrderManagerConfig(tp1_method="FIXED_DISTANCE"),
        )
        assert result.tp1_price == 1896.0  # bottom - 4 default


# --------------------------------------------------------------------------- #
# TP1 method = BOS_LEVEL — strategy default
# --------------------------------------------------------------------------- #


class TestTP1MethodBosLevel:
    """May 2026 strategy refinement: TP1 = the swing high/low broken by
    the impulse before the zone formed (spec Section 6.1)."""

    def test_buy_uses_bos_event_broken_level(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Mirrors Tommy's screenshot example shape (zone ~6 pts wide,
        # BoS ~7 pts above zone top) with mock-fixture-compatible prices.
        # The actual numbers in the screenshot were 4704/4710/4717.478.
        zone = make_imbalance_zone(
            direction="BUY", top=1900.0, bottom=1894.0,
            bos_broken_level=1907.478,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, sl_price=1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.tp1_price == pytest.approx(1907.478)

    def test_sell_uses_bos_event_broken_level(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # SELL mirror. Zone 1900-1906; BoS at 1885 (broken swing low
        # below the zone, the natural retracement target downward).
        # Mock fixture's price_open=1900.00 lands exactly at zone bottom
        # (zone.top=1906, zone.bottom=1900 for SELL → entry at top means
        # fill ≤ top + tolerance), so we override it.
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 11111, "price_open": 1906.00},
        ]
        zone = make_imbalance_zone(
            direction="SELL", top=1906.0, bottom=1900.0,
            bos_broken_level=1885.0,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, sl_price=1920.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.tp1_price == pytest.approx(1885.0)

    def test_bos_close_to_zone_still_used(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Edge case: BoS just 1.5 pts above zone top — tiny TP, but spec
        # is "no filter, take all valid Strong Points".
        zone = make_imbalance_zone(
            direction="BUY", top=1900.0, bottom=1895.0,
            bos_broken_level=1901.5,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.tp1_price == pytest.approx(1901.5)

    def test_bos_far_from_zone_still_used(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Edge case: BoS 60+ pts above zone — wide TP. Same rule.
        zone = make_imbalance_zone(
            direction="BUY", top=1900.0, bottom=1895.0,
            bos_broken_level=1962.0,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "PLACED"
        assert result.tp1_price == pytest.approx(1962.0)

    def test_planned_tp1_price_in_setup_row_matches_bos_level(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Verify the value persists into the Supabase setup row, not
        # just the OrderPlacementResult.
        zone = make_imbalance_zone(
            direction="BUY", top=1900.0, bottom=1895.0,
            bos_broken_level=1912.5,
        )
        place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        s = mock_supabase.log_setup.call_args.args[0]
        assert float(s.planned_tp1_price) == pytest.approx(1912.5)

    def test_missing_bos_event_fails_pre_check_no_setup_written(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Defensive: a zone reaching order_manager without a bos_event
        # means validation was bypassed somewhere upstream. Hard-fail
        # with a clear error rather than crashing on None-deref.
        zone = make_imbalance_zone_no_bos(
            direction="BUY", top=1900.0, bottom=1895.0,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
        )
        assert result.status == "FAILED"
        assert result.setup_id is None
        # Pre-check fails before any side effects.
        mock_supabase.log_setup.assert_not_called()
        mock_mt5.place_market_order.assert_not_called()
        # Error message is actionable.
        assert any("broken_swing" in m for m in result.error_messages)
        assert any("BOS_LEVEL" in m for m in result.error_messages)

    def test_missing_bos_event_fixed_distance_still_works(
        self, mock_mt5, mock_supabase, zone_id,
    ) -> None:
        # Inverse of the previous test: with FIXED_DISTANCE, no bos_event
        # required — the legacy path is the rollback escape hatch.
        zone = make_imbalance_zone_no_bos(
            direction="BUY", top=1900.0, bottom=1895.0,
        )
        result = place_layered_orders(
            zone, zone_id, 0.01, 1880.0,
            mt5=mock_mt5, supabase=mock_supabase,
            config=OrderManagerConfig(tp1_method="FIXED_DISTANCE"),
        )
        assert result.status == "PLACED"
        assert result.tp1_price == 1904.0  # top + 4 (default)
