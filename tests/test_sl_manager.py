"""Tests for ``bot.exits.sl_manager``.

Same pattern as test_tp1_manager / test_entry_trigger:
``MagicMock(spec=...)`` for dependencies, helper builders for the
typed Setup / Trade / Zone fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.execution.mt5_connector import MT5Connector
from bot.exits.sl_manager import (
    SLCalculation,
    SLManager,
    SLManagerConfig,
    SLValidation,
    _resolve_buy_reference,
    _resolve_sell_reference,
)
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade
from bot.strategy.imbalance import ImbalanceZone
from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.structure import BosEvent, Swing
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
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
        planned_tp1_price=Decimal("1907"),
        status=status,  # type: ignore[arg-type]
        skip_reason=None,
        activated_at=NOW,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_trade(
    *,
    id: UUID | None = None,
    setup_id: UUID | None = None,
    layer_number: int = 1,
    status: str = "FILLED",
    mt5_ticket: int | None = 11111,
) -> Trade:
    return Trade(
        id=id or uuid4(),
        setup_id=setup_id or uuid4(),
        layer_number=layer_number,
        direction="BUY",
        order_type="MARKET",
        mt5_ticket=mt5_ticket,
        entry_price=Decimal("1900.0"),
        exit_price=None,
        lot_size=Decimal("0.01"),
        sl_price=Decimal("1880.0"),
        tp_price=None,
        status=status,  # type: ignore[arg-type]
        pnl=None,
        commission=Decimal("0"),
        swap=Decimal("0"),
        close_reason=None,
        filled_at=NOW,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_zone(
    *,
    direction: str = "BUY",
    top: float = 1900.0,
    bottom: float = 1895.0,
) -> ImbalanceZone:
    """Tiny ImbalanceZone — only direction matters for sl_manager.

    bos_event is set so the zone could in principle reach order_manager;
    sl_manager itself doesn't care. Keep it conservative anyway."""
    ts = pd.Timestamp("2026-05-08T12:00:00Z")
    if direction == "BUY":
        pattern: Any = WPattern(
            low1=Swing(index=2, time=ts, price=bottom, kind="LOW"),
            low2=Swing(index=9, time=ts, price=bottom, kind="LOW"),
            peak_index=6, peak_time=ts, peak_price=top + 5,
            formed_at=ts, completed=True,
        )
    else:
        pattern = MPattern(
            high1=Swing(index=2, time=ts, price=top, kind="HIGH"),
            high2=Swing(index=9, time=ts, price=top, kind="HIGH"),
            trough_index=6, trough_time=ts, trough_price=bottom - 5,
            formed_at=ts, completed=True,
        )
    initial = Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
    )
    refined = RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=True, rejection_reason=None,
        original_zone=initial,
    )
    bos = BosEvent(
        bar_index=12, time=ts,
        direction="UP" if direction == "BUY" else "DOWN",
        broken_swing_index=4,
        broken_level=top + 5 if direction == "BUY" else bottom - 5,
        break_close=top + 6 if direction == "BUY" else bottom - 6,
    )
    validated = ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=True, rejection_reason=None,
        original_zone=initial, refined_zone=refined,
        is_strong_point=True, validation_failures=[], bos_event=bos,
    )
    return ImbalanceZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=True, rejection_reason=None,
        original_zone=initial, refined_zone=refined,
        is_strong_point=True, validation_failures=[], bos_event=bos,
        validated_zone=validated,
        approach_count=2, is_imbalance=True, approach_events=[],
        qualified_at=ts, is_tapped=False, tapped_at=None,
    )


def make_ohlc_with_swing_low(
    *,
    swing_low_price: float,
    swing_low_index: int = 10,
    n_bars: int = 25,
    base_price: float = 4520.0,
    start: str = "2026-05-08T08:00:00Z",
) -> pd.DataFrame:
    """Build OHLC where bar at ``swing_low_index`` has the lowest close.

    The dip is wide enough (3 bars on each side at ``base_price``) for
    swing_strength=3 detection to fire on it.
    """
    closes = [base_price] * n_bars
    closes[swing_low_index] = swing_low_price
    times = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [100] * n_bars,
        },
        index=times,
    )


def make_ohlc_with_swing_high(
    *,
    swing_high_price: float,
    swing_high_index: int = 10,
    n_bars: int = 25,
    base_price: float = 4480.0,
    start: str = "2026-05-08T08:00:00Z",
) -> pd.DataFrame:
    closes = [base_price] * n_bars
    closes[swing_high_index] = swing_high_price
    times = pd.date_range(start=start, periods=n_bars, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [100] * n_bars,
        },
        index=times,
    )


def make_flat_ohlc(
    *, n_bars: int = 25, base_price: float = 4500.0,
) -> pd.DataFrame:
    """All bars identical → swing detector finds nothing."""
    times = pd.date_range(
        start="2026-05-08T08:00:00Z", periods=n_bars, freq="5min", tz="UTC",
    )
    return pd.DataFrame(
        {
            "open": [base_price] * n_bars,
            "high": [base_price + 0.5] * n_bars,
            "low": [base_price - 0.5] * n_bars,
            "close": [base_price] * n_bars,
            "volume": [100] * n_bars,
        },
        index=times,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    return mocker.MagicMock(spec=MT5Connector)


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.get_trades_for_setup.return_value = []
    return m


@pytest.fixture
def manager(
    mock_mt5: MagicMock, mock_supabase: MagicMock,
) -> SLManager:
    return SLManager(mt5=mock_mt5, supabase=mock_supabase)


# --------------------------------------------------------------------------- #
# calculate_initial_sl
# --------------------------------------------------------------------------- #


class TestCalculateInitialSL:
    def test_buy_uses_lowest_swing_low_minus_buffer(
        self, manager: SLManager,
    ) -> None:
        # Single swing low at 4500 inside the lookback window.
        zone = make_zone(direction="BUY", top=4520.0, bottom=4515.0)
        df = make_ohlc_with_swing_low(
            swing_low_price=4500.0, swing_low_index=10, n_bars=25,
            base_price=4520.0,
        )
        result = manager.calculate_initial_sl(zone, df)

        assert result.direction == "BUY"
        assert result.reference_swing_price == 4500.0
        assert result.buffer_used == 17.5
        assert result.fallback_used is False
        # 4500 - 17.5 = 4482.5
        assert result.sl_price == pytest.approx(4482.5)

    def test_sell_uses_highest_swing_high_plus_buffer(
        self, manager: SLManager,
    ) -> None:
        zone = make_zone(direction="SELL", top=4520.0, bottom=4515.0)
        df = make_ohlc_with_swing_high(
            swing_high_price=4540.0, swing_high_index=10, n_bars=25,
            base_price=4520.0,
        )
        result = manager.calculate_initial_sl(zone, df)

        assert result.direction == "SELL"
        assert result.reference_swing_price == 4540.0
        # 4540 + 17.5 = 4557.5
        assert result.sl_price == pytest.approx(4557.5)
        assert result.fallback_used is False

    def test_buy_picks_lowest_among_multiple_swing_lows(
        self, manager: SLManager,
    ) -> None:
        # Two swing lows: one at 4505, one (deeper) at 4495.
        zone = make_zone(direction="BUY")
        n = 30
        closes = [4520.0] * n
        closes[8] = 4505.0
        closes[18] = 4495.0
        times = pd.date_range(
            start="2026-05-08T08:00:00Z", periods=n, freq="5min", tz="UTC",
        )
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [100] * n,
            },
            index=times,
        )

        result = manager.calculate_initial_sl(zone, df)

        # Lowest of {4505, 4495} = 4495.
        assert result.reference_swing_price == 4495.0
        assert result.sl_price == pytest.approx(4495.0 - 17.5)
        assert result.fallback_used is False

    def test_sell_picks_highest_among_multiple_swing_highs(
        self, manager: SLManager,
    ) -> None:
        zone = make_zone(direction="SELL")
        n = 30
        closes = [4480.0] * n
        closes[8] = 4495.0
        closes[18] = 4505.0
        times = pd.date_range(
            start="2026-05-08T08:00:00Z", periods=n, freq="5min", tz="UTC",
        )
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 0.5 for c in closes],
                "low": [c - 0.5 for c in closes],
                "close": closes,
                "volume": [100] * n,
            },
            index=times,
        )

        result = manager.calculate_initial_sl(zone, df)
        assert result.reference_swing_price == 4505.0
        assert result.sl_price == pytest.approx(4505.0 + 17.5)

    def test_buy_no_swings_in_window_falls_back_to_lowest_low(
        self, manager: SLManager,
    ) -> None:
        # Flat closes → swing detector finds nothing. Fallback uses
        # the lowest bar low across the lookback window.
        zone = make_zone(direction="BUY")
        df = make_flat_ohlc(n_bars=25, base_price=4500.0)
        # Lowest low = base_price - 0.5 = 4499.5
        result = manager.calculate_initial_sl(zone, df)

        assert result.fallback_used is True
        assert result.reference_swing_price == pytest.approx(4499.5)
        assert result.sl_price == pytest.approx(4499.5 - 17.5)

    def test_sell_no_swings_in_window_falls_back_to_highest_high(
        self, manager: SLManager,
    ) -> None:
        zone = make_zone(direction="SELL")
        df = make_flat_ohlc(n_bars=25, base_price=4500.0)
        # Highest high = base_price + 0.5 = 4500.5
        result = manager.calculate_initial_sl(zone, df)

        assert result.fallback_used is True
        assert result.reference_swing_price == pytest.approx(4500.5)
        assert result.sl_price == pytest.approx(4500.5 + 17.5)

    def test_lookback_capped_at_available_bars(
        self, manager: SLManager,
    ) -> None:
        # Only 10 bars available; lookback config is 20. Should use 10.
        zone = make_zone(direction="BUY")
        df = make_ohlc_with_swing_low(
            swing_low_price=4500.0, swing_low_index=5, n_bars=10,
            base_price=4520.0,
        )
        result = manager.calculate_initial_sl(zone, df)
        assert result.lookback_used == 10
        # Swing detection still works on 10 bars (need 2*strength+1 = 7
        # at minimum; strength=3 → bar at idx 5 has full shoulders 2..8).
        assert result.reference_swing_price == 4500.0

    def test_custom_buffer_via_config(
        self, mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        mgr = SLManager(
            mt5=mock_mt5, supabase=mock_supabase,
            config=SLManagerConfig(sl_buffer_points=20.0),
        )
        zone = make_zone(direction="BUY")
        df = make_ohlc_with_swing_low(
            swing_low_price=4500.0, swing_low_index=10, n_bars=25,
            base_price=4520.0,
        )
        result = mgr.calculate_initial_sl(zone, df)
        assert result.buffer_used == 20.0
        assert result.sl_price == pytest.approx(4500.0 - 20.0)

    def test_missing_columns_raises(self, manager: SLManager) -> None:
        zone = make_zone(direction="BUY")
        df = pd.DataFrame({"close": [4500.0] * 10})  # no high / low
        with pytest.raises(ValueError, match="must have"):
            manager.calculate_initial_sl(zone, df)

    def test_empty_ohlc_raises(self, manager: SLManager) -> None:
        # Use a config with min lookback so the empty check fires.
        mgr = SLManager(
            mt5=MagicMock(spec=MT5Connector),
            supabase=MagicMock(spec=SupabaseLogger),
            config=SLManagerConfig(recent_swing_lookback=20),
        )
        zone = make_zone(direction="BUY")
        df = pd.DataFrame({"open": [], "high": [], "low": [], "close": []})
        with pytest.raises(ValueError, match="empty"):
            mgr.calculate_initial_sl(zone, df)


# --------------------------------------------------------------------------- #
# Module-level reference resolvers
# --------------------------------------------------------------------------- #


class TestReferenceResolvers:
    def test_buy_resolver_uses_min_swing_low_no_fallback(self) -> None:
        ts = pd.Timestamp("2026-05-08T12:00:00Z")
        swings = [
            Swing(index=5, time=ts, price=4505.0, kind="LOW"),
            Swing(index=10, time=ts, price=4495.0, kind="LOW"),
            Swing(index=15, time=ts, price=4520.0, kind="HIGH"),  # ignored
        ]
        window = pd.DataFrame({"low": [4480.0]})
        ref, fallback = _resolve_buy_reference(swings, window)
        assert ref == 4495.0
        assert fallback is False

    def test_buy_resolver_falls_back_when_no_low_swings(self) -> None:
        ts = pd.Timestamp("2026-05-08T12:00:00Z")
        # Only HIGH swings present → no LOW for BUY.
        swings = [Swing(index=5, time=ts, price=4520.0, kind="HIGH")]
        window = pd.DataFrame({"low": [4480.0, 4485.0, 4475.0]})
        ref, fallback = _resolve_buy_reference(swings, window)
        assert ref == 4475.0  # min of the lows column
        assert fallback is True

    def test_sell_resolver_uses_max_swing_high_no_fallback(self) -> None:
        ts = pd.Timestamp("2026-05-08T12:00:00Z")
        swings = [
            Swing(index=5, time=ts, price=4505.0, kind="HIGH"),
            Swing(index=10, time=ts, price=4515.0, kind="HIGH"),
            Swing(index=15, time=ts, price=4480.0, kind="LOW"),  # ignored
        ]
        window = pd.DataFrame({"high": [4540.0]})
        ref, fallback = _resolve_sell_reference(swings, window)
        assert ref == 4515.0
        assert fallback is False

    def test_sell_resolver_falls_back_when_no_high_swings(self) -> None:
        ts = pd.Timestamp("2026-05-08T12:00:00Z")
        swings = [Swing(index=5, time=ts, price=4480.0, kind="LOW")]
        window = pd.DataFrame({"high": [4500.0, 4510.0, 4505.0]})
        ref, fallback = _resolve_sell_reference(swings, window)
        assert ref == 4510.0  # max of the highs column
        assert fallback is True


# --------------------------------------------------------------------------- #
# validate_sl_distance
# --------------------------------------------------------------------------- #


class TestValidateSLDistance:
    def test_buy_sl_below_entry_within_band_valid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4520.0, sl_price=4500.0, direction="BUY",
        )
        assert result.is_valid is True
        assert result.distance_points == pytest.approx(20.0)
        assert result.is_too_close is False
        assert result.is_too_far is False
        assert result.error is None

    def test_sell_sl_above_entry_within_band_valid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4480.0, sl_price=4500.0, direction="SELL",
        )
        assert result.is_valid is True
        assert result.distance_points == pytest.approx(20.0)

    def test_buy_sl_at_or_above_entry_invalid(
        self, manager: SLManager,
    ) -> None:
        # Equal: SL must be strictly below entry for BUY.
        equal = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4500.0, direction="BUY",
        )
        assert equal.is_valid is False
        assert equal.error is not None
        assert "must be below" in equal.error

        above = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4510.0, direction="BUY",
        )
        assert above.is_valid is False

    def test_sell_sl_at_or_below_entry_invalid(
        self, manager: SLManager,
    ) -> None:
        equal = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4500.0, direction="SELL",
        )
        assert equal.is_valid is False
        assert "must be above" in (equal.error or "")

        below = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4490.0, direction="SELL",
        )
        assert below.is_valid is False

    def test_too_close_flagged(self, manager: SLManager) -> None:
        # Default min = 5. Distance 3 → too close.
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4497.0, direction="BUY",
        )
        assert result.is_valid is False
        assert result.is_too_close is True
        assert result.is_too_far is False
        assert result.distance_points == pytest.approx(3.0)
        assert "below minimum" in (result.error or "")

    def test_too_far_flagged(self, manager: SLManager) -> None:
        # Default max = 200. Distance 250 → too far.
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4250.0, direction="BUY",
        )
        assert result.is_valid is False
        assert result.is_too_far is True
        assert result.is_too_close is False
        assert result.distance_points == pytest.approx(250.0)
        assert "above maximum" in (result.error or "")

    def test_boundary_at_min_distance_valid(
        self, manager: SLManager,
    ) -> None:
        # Distance exactly equals min — boundary inclusive.
        result = manager.validate_sl_distance(
            entry_price=4505.0, sl_price=4500.0, direction="BUY",
        )
        # 5.0 == min_sl_distance_points (5.0) → valid.
        assert result.distance_points == pytest.approx(5.0)
        assert result.is_too_close is False
        assert result.is_valid is True

    def test_boundary_at_max_distance_valid(
        self, manager: SLManager,
    ) -> None:
        # Distance exactly equals max — boundary inclusive.
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4300.0, direction="BUY",
        )
        # 200.0 == max_sl_distance_points (200.0) → valid.
        assert result.distance_points == pytest.approx(200.0)
        assert result.is_too_far is False
        assert result.is_valid is True

    def test_unknown_direction_invalid(
        self, manager: SLManager,
    ) -> None:
        result = manager.validate_sl_distance(
            entry_price=4500.0, sl_price=4490.0, direction="X",
        )
        assert result.is_valid is False
        assert result.error is not None
        assert "unknown direction" in result.error

    def test_custom_min_max_via_config(
        self, mock_mt5: MagicMock, mock_supabase: MagicMock,
    ) -> None:
        mgr = SLManager(
            mt5=mock_mt5, supabase=mock_supabase,
            config=SLManagerConfig(
                min_sl_distance_points=10.0,
                max_sl_distance_points=50.0,
            ),
        )
        # Distance 8 — under custom min of 10.
        r = mgr.validate_sl_distance(4500.0, 4492.0, "BUY")
        assert r.is_too_close is True
        # Distance 60 — above custom max of 50.
        r = mgr.validate_sl_distance(4500.0, 4440.0, "BUY")
        assert r.is_too_far is True


# --------------------------------------------------------------------------- #
# apply_sl_to_setup
# --------------------------------------------------------------------------- #


class TestApplySLToSetup:
    def test_three_open_positions_all_succeed_returns_true(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=1, mt5_ticket=11111),
            make_trade(setup_id=setup.id, layer_number=2, mt5_ticket=22222),
            make_trade(setup_id=setup.id, layer_number=3, mt5_ticket=33333),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is True
        assert mock_mt5.modify_order.call_count == 3
        # Each call had sl=4485.0
        for call in mock_mt5.modify_order.call_args_list:
            assert call.kwargs["sl"] == 4485.0
        # Each trade row sl_price was synced.
        assert mock_supabase.update_trade.call_count == 3
        # Three INFO log_event calls, no ERROR.
        log_levels = [c.kwargs.get("level") for c in mock_supabase.log_event.call_args_list]
        assert log_levels.count("INFO") == 3
        assert "ERROR" not in log_levels

    def test_one_of_three_fails_returns_false_others_still_attempted(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        # Fail-soft: continue on failure; return False overall.
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=1, mt5_ticket=11111),
            make_trade(setup_id=setup.id, layer_number=2, mt5_ticket=22222),
            make_trade(setup_id=setup.id, layer_number=3, mt5_ticket=33333),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        # Layer 2 fails; Layers 1 and 3 succeed.
        mock_mt5.modify_order.side_effect = [None, RuntimeError("busy"), None]

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is False
        # All three were attempted (fail-soft).
        assert mock_mt5.modify_order.call_count == 3
        # The two successes synced their trade rows; the failure didn't.
        assert mock_supabase.update_trade.call_count == 2
        # CRITICAL log written for the failure.
        log_levels = [c.kwargs.get("level") for c in mock_supabase.log_event.call_args_list]
        assert "ERROR" in log_levels

    def test_all_three_fail_returns_false_all_attempted(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=i, mt5_ticket=10000 + i)
            for i in (1, 2, 3)
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_mt5.modify_order.side_effect = RuntimeError("broker down")

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is False
        assert mock_mt5.modify_order.call_count == 3

    def test_no_open_positions_returns_true_no_modify_calls(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        # Setup has only WAITING trades — no broker positions to modify.
        setup = make_setup()
        trades = [
            make_trade(
                setup_id=setup.id, layer_number=2,
                status="WAITING", mt5_ticket=None,
            ),
            make_trade(
                setup_id=setup.id, layer_number=3,
                status="WAITING", mt5_ticket=None,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is True
        mock_mt5.modify_order.assert_not_called()

    def test_setup_with_no_trade_rows_returns_true(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        # Edge case: setup row exists but no trade rows yet (e.g. the
        # call landed between log_setup and the first log_trade write).
        setup = make_setup()
        mock_supabase.get_trades_for_setup.return_value = []

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is True
        mock_mt5.modify_order.assert_not_called()

    def test_terminal_trades_skipped(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        # CLOSED / CANCELLED trades have no live position — ignore them.
        setup = make_setup()
        trades = [
            make_trade(
                setup_id=setup.id, layer_number=1,
                status="FILLED", mt5_ticket=11111,
            ),
            make_trade(
                setup_id=setup.id, layer_number=2,
                status="CLOSED", mt5_ticket=22222,
            ),
            make_trade(
                setup_id=setup.id, layer_number=3,
                status="CANCELLED", mt5_ticket=None,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)

        assert ok is True
        # Only Layer 1 modified.
        assert mock_mt5.modify_order.call_count == 1
        assert mock_mt5.modify_order.call_args.args[0] == 11111

    def test_partially_closed_trades_modified_too(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        # PARTIALLY_CLOSED (post-TP1 runner) still has an open position.
        setup = make_setup()
        trades = [
            make_trade(
                setup_id=setup.id, layer_number=1,
                status="PARTIALLY_CLOSED", mt5_ticket=11111,
            ),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)
        assert ok is True
        mock_mt5.modify_order.assert_called_once()

    def test_supabase_sync_failure_does_not_fail_overall(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        # Broker accepts the SL change; Supabase row update fails.
        # Broker is the source of truth — overall result is still True.
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=1, mt5_ticket=11111),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_supabase.update_trade.side_effect = RuntimeError("DB blip")

        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)
        assert ok is True

    def test_log_event_failure_is_swallowed(
        self, manager: SLManager, mock_mt5: MagicMock,
        mock_supabase: MagicMock,
    ) -> None:
        # bot_logs writes are best-effort — never raise.
        setup = make_setup()
        trades = [
            make_trade(setup_id=setup.id, layer_number=1, mt5_ticket=11111),
        ]
        mock_supabase.get_trades_for_setup.return_value = trades
        mock_supabase.log_event.side_effect = RuntimeError("DB down")

        # Doesn't raise.
        ok = manager.apply_sl_to_setup(setup, sl_price=4485.0)
        assert ok is True


# --------------------------------------------------------------------------- #
# Result + config dataclass shape
# --------------------------------------------------------------------------- #


class TestDataclassDefaults:
    def test_sl_calculation_default_fallback_is_false(self) -> None:
        c = SLCalculation(
            sl_price=4482.5, reference_swing_price=4500.0,
            buffer_used=17.5, lookback_used=20, direction="BUY",
        )
        assert c.fallback_used is False

    def test_sl_validation_default_error_is_none(self) -> None:
        v = SLValidation(
            is_valid=True, distance_points=20.0,
            is_too_close=False, is_too_far=False,
        )
        assert v.error is None

    def test_config_defaults_match_spec(self) -> None:
        cfg = SLManagerConfig()
        assert cfg.symbol == "XAUUSD"
        assert cfg.swing_strength == 3
        assert cfg.recent_swing_lookback == 20
        assert cfg.sl_buffer_points == 17.5
        assert cfg.min_sl_distance_points == 5.0
        assert cfg.max_sl_distance_points == 200.0
