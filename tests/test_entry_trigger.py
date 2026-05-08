"""Tests for ``bot.execution.entry_trigger``.

Same pytest-mock pattern as test_order_manager and test_position_tracker:
``MagicMock(spec=...)`` for dependencies, helper builders for the
typed Setup/Trade fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.execution.entry_trigger import (
    EntryTrigger,
    EntryTriggerConfig,
    FiredTrigger,
    _bars_since_activation,
    _trigger_met_history,
    _trigger_met_live,
    _trigger_price,
)
from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import PositionTracker
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
    layer_1_price: float = 1900.0,
    layer_2_price: float = 1897.5,
    layer_3_price: float = 1895.0,
    activated_at: datetime | None = NOW,
) -> Setup:
    return Setup(
        id=id or uuid4(),
        zone_id=uuid4(),
        direction=direction,  # type: ignore[arg-type]
        entry_mode="STRONG_POINT_FIRST_TOUCH",
        planned_layer1_price=Decimal(str(layer_1_price)),
        planned_layer2_price=Decimal(str(layer_2_price)),
        planned_layer3_price=Decimal(str(layer_3_price)),
        planned_sl_price=Decimal("1880"),
        planned_tp1_price=Decimal("1904"),
        status=status,  # type: ignore[arg-type]
        skip_reason=None,
        activated_at=activated_at,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_trade(
    *,
    id: UUID | None = None,
    setup_id: UUID | None = None,
    layer_number: int = 2,
    status: str = "WAITING",
    mt5_ticket: int | None = None,
) -> Trade:
    return Trade(
        id=id or uuid4(),
        setup_id=setup_id or uuid4(),
        layer_number=layer_number,
        direction="BUY",
        order_type="MARKET",
        mt5_ticket=mt5_ticket,
        entry_price=None,
        exit_price=None,
        lot_size=Decimal("0.01"),
        sl_price=Decimal("1880"),
        tp_price=None,
        status=status,  # type: ignore[arg-type]
        pnl=None,
        commission=Decimal("0"),
        swap=Decimal("0"),
        close_reason=None,
        filled_at=None,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_ohlc(
    closes: list[float],
    *,
    lows: list[float] | None = None,
    highs: list[float] | None = None,
    start: str = "2026-05-08T12:00:00Z",
) -> pd.DataFrame:
    n = len(closes)
    if lows is None:
        lows = list(closes)
    if highs is None:
        highs = list(closes)
    times = pd.date_range(start=start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": [100] * n},
        index=times,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=MT5Connector)
    m.place_market_order.return_value = 44444
    m.get_open_positions.return_value = [
        {"ticket": 44444, "price_open": 1897.50},
    ]
    return m


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.get_trades_for_setup.return_value = []
    m.get_trade_by_id.return_value = None
    return m


@pytest.fixture
def mock_tracker(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=PositionTracker)
    m.get_active_setups.return_value = []
    return m


@pytest.fixture
def trigger(
    mock_mt5: MagicMock,
    mock_supabase: MagicMock,
    mock_tracker: MagicMock,
) -> EntryTrigger:
    return EntryTrigger(
        mt5=mock_mt5,
        supabase=mock_supabase,
        position_tracker=mock_tracker,
    )


# --------------------------------------------------------------------------- #
# Pure helpers — _trigger_price, _trigger_met_live, _trigger_met_history
# --------------------------------------------------------------------------- #


class TestTriggerPriceMapping:
    def test_layer_1_returns_layer1_price(self) -> None:
        s = make_setup(layer_1_price=1900.0)
        t = make_trade(layer_number=1)
        assert _trigger_price(s, t) == 1900.0

    def test_layer_2_returns_midpoint(self) -> None:
        s = make_setup(layer_2_price=1897.5)
        t = make_trade(layer_number=2)
        assert _trigger_price(s, t) == 1897.5

    def test_layer_3_returns_far_edge(self) -> None:
        s = make_setup(layer_3_price=1895.0)
        t = make_trade(layer_number=3)
        assert _trigger_price(s, t) == 1895.0

    def test_unknown_layer_raises(self) -> None:
        s = make_setup()
        t = make_trade(layer_number=99)
        with pytest.raises(ValueError):
            _trigger_price(s, t)


class TestTriggerMetLive:
    def test_buy_fires_when_bid_at_or_below_trigger(self) -> None:
        s = make_setup(direction="BUY")
        # bid=1897.5, trigger=1897.5 → inclusive boundary
        assert _trigger_met_live(s, 1897.5, bid=1897.5, ask=1897.6) is True
        # bid=1897.49 → past trigger, also fires
        assert _trigger_met_live(s, 1897.5, bid=1897.49, ask=1897.6) is True
        # bid=1897.51 → above, doesn't fire
        assert _trigger_met_live(s, 1897.5, bid=1897.51, ask=1897.6) is False

    def test_sell_fires_when_ask_at_or_above_trigger(self) -> None:
        s = make_setup(direction="SELL")
        assert _trigger_met_live(s, 1902.5, bid=1902.4, ask=1902.5) is True
        assert _trigger_met_live(s, 1902.5, bid=1902.4, ask=1902.51) is True
        assert _trigger_met_live(s, 1902.5, bid=1902.4, ask=1902.49) is False


class TestTriggerMetHistory:
    def test_buy_fires_when_any_bar_low_at_or_below_trigger(self) -> None:
        s = make_setup(direction="BUY")
        # All bar lows above trigger 1897.5 → no fire.
        df = make_ohlc(
            closes=[1900, 1900, 1899],
            lows=[1899, 1898, 1898],
        )
        assert _trigger_met_history(s, 1897.5, df) is False

        # One bar's low at 1897 → fires (1897 <= 1897.5).
        df = make_ohlc(
            closes=[1900, 1898, 1899],
            lows=[1899, 1897, 1898],
        )
        assert _trigger_met_history(s, 1897.5, df) is True

    def test_sell_fires_when_any_bar_high_at_or_above_trigger(self) -> None:
        s = make_setup(direction="SELL")
        df = make_ohlc(
            closes=[1900, 1901, 1900],
            highs=[1901, 1902, 1901],
        )
        assert _trigger_met_history(s, 1902.5, df) is False

        df = make_ohlc(
            closes=[1900, 1903, 1901],
            highs=[1901, 1903, 1902],
        )
        assert _trigger_met_history(s, 1902.5, df) is True

    def test_empty_dataframe_returns_false(self) -> None:
        s = make_setup(direction="BUY")
        empty = pd.DataFrame({"low": [], "high": []})
        assert _trigger_met_history(s, 1897.5, empty) is False


class TestBarsSinceActivation:
    def test_returns_full_df_when_no_activated_at(self) -> None:
        s = make_setup(activated_at=None)
        df = make_ohlc([1900, 1901, 1902])
        result = _bars_since_activation(s, df)
        assert len(result) == 3

    def test_filters_to_bars_at_or_after_activation(self) -> None:
        # 5 bars from 12:00 onward (5min freq).
        df = make_ohlc(
            [1900, 1901, 1902, 1903, 1904],
            start="2026-05-08T12:00:00Z",
        )
        # Activate at 12:10 → bars at 12:10, 12:15, 12:20 (indices 2, 3, 4).
        s = make_setup(
            activated_at=datetime(2026, 5, 8, 12, 10, tzinfo=timezone.utc),
        )
        result = _bars_since_activation(s, df)
        assert len(result) == 3
        assert float(result["close"].iloc[0]) == 1902


# --------------------------------------------------------------------------- #
# check_live — per-tick flow
# --------------------------------------------------------------------------- #


class TestCheckLive:
    def test_no_active_setups_no_fires(
        self, trigger, mock_tracker, mock_mt5, mock_supabase,
    ) -> None:
        mock_tracker.get_active_setups.return_value = []
        result = trigger.check_live(bid=1900.0, ask=1900.1)
        assert result == []
        mock_mt5.place_market_order.assert_not_called()

    def test_active_setup_with_no_waiting_trades_no_fires(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        s = make_setup(status="ACTIVE")
        mock_tracker.get_active_setups.return_value = [s]
        # All trades FILLED → none waiting.
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(layer_number=1, status="FILLED", mt5_ticket=11111),
        ]
        result = trigger.check_live(bid=1897.0, ask=1897.1)
        assert result == []
        mock_mt5.place_market_order.assert_not_called()

    def test_pending_setup_does_not_fire_layers(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        # PENDING setups have no Layer 1 fill yet — Layer 2/3 shouldn't fire.
        s = make_setup(status="PENDING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [
            make_trade(layer_number=2, status="WAITING"),
        ]
        result = trigger.check_live(bid=1893.0, ask=1893.1)
        assert result == []
        mock_mt5.place_market_order.assert_not_called()

    def test_buy_layer_2_fires_when_bid_reaches_trigger(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        s = make_setup(direction="BUY", layer_2_price=1897.5)
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]

        result = trigger.check_live(bid=1897.5, ask=1897.6)

        assert len(result) == 1
        fired = result[0]
        assert isinstance(fired, FiredTrigger)
        assert fired.setup_id == s.id
        assert fired.trade_id == layer_2.id
        assert fired.layer_number == 2
        assert fired.mt5_ticket == 44444
        # Verify market order was placed.
        mock_mt5.place_market_order.assert_called_once()
        # Verify trade row was transitioned WAITING → FILLED.
        mock_tracker.update_trade_status.assert_called_once()
        kwargs = mock_tracker.update_trade_status.call_args.kwargs
        assert kwargs["entry_price"] == 1897.50
        assert kwargs["mt5_ticket"] == 44444

    def test_buy_layer_2_does_not_fire_above_trigger(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        s = make_setup(direction="BUY", layer_2_price=1897.5)
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]

        # Bid above trigger → no fire.
        result = trigger.check_live(bid=1898.0, ask=1898.1)
        assert result == []
        mock_mt5.place_market_order.assert_not_called()

    def test_tick_gap_fires_all_waiting_layers_in_one_call(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        # The headline edge case: tick gaps from 1900 to 1893 in one update.
        # Both Layer 2 (1897.5) and Layer 3 (1895) triggers are met.
        s = make_setup(
            direction="BUY", layer_2_price=1897.5, layer_3_price=1895.0,
        )
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        layer_3 = make_trade(setup_id=s.id, layer_number=3, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2, layer_3]
        # Two market orders placed.
        mock_mt5.place_market_order.side_effect = [44444, 55555]
        # MT5 returns both positions.
        mock_mt5.get_open_positions.side_effect = [
            [{"ticket": 44444, "price_open": 1893.0}],
            [
                {"ticket": 44444, "price_open": 1893.0},
                {"ticket": 55555, "price_open": 1893.0},
            ],
        ]

        result = trigger.check_live(bid=1893.0, ask=1893.1)

        assert len(result) == 2
        layer_numbers = {f.layer_number for f in result}
        assert layer_numbers == {2, 3}
        assert mock_mt5.place_market_order.call_count == 2

    def test_sell_layer_2_fires_when_ask_at_trigger(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        s = make_setup(direction="SELL", layer_2_price=1902.5)
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]

        result = trigger.check_live(bid=1902.4, ask=1902.5)
        assert len(result) == 1

    def test_mt5_failure_does_not_transition_trade(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        s = make_setup(direction="BUY", layer_2_price=1897.5)
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]
        mock_mt5.place_market_order.side_effect = RuntimeError("requote")

        result = trigger.check_live(bid=1897.0, ask=1897.1)
        assert result == []
        # Trade stays WAITING — no transition called.
        mock_tracker.update_trade_status.assert_not_called()


# --------------------------------------------------------------------------- #
# check_history — startup catch-up
# --------------------------------------------------------------------------- #


class TestCheckHistory:
    def test_buy_layer_2_fires_when_bar_history_shows_trigger_crossed(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        # Bot was offline, price dipped to 1897 (crossing 1897.5 trigger),
        # bounced back to 1898. On startup, check_history fires Layer 2.
        s = make_setup(
            direction="BUY",
            layer_2_price=1897.5,
            activated_at=datetime(2026, 5, 8, 11, 0, tzinfo=timezone.utc),
        )
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]

        df = make_ohlc(
            closes=[1900, 1898, 1898, 1899, 1898],
            lows=[1899, 1897, 1898, 1898, 1898],
            start="2026-05-08T11:00:00Z",
        )

        result = trigger.check_history(df)
        assert len(result) == 1
        assert result[0].layer_number == 2
        # MT5 placed market order at current price (mocked at 44444 / 1897.50).
        mock_mt5.place_market_order.assert_called_once()

    def test_no_bar_crossed_trigger_no_fire(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        s = make_setup(direction="BUY", layer_2_price=1897.5)
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]

        df = make_ohlc(
            closes=[1900, 1899, 1898],
            lows=[1899, 1898, 1898],  # never reached 1897.5
        )
        result = trigger.check_history(df)
        assert result == []
        mock_mt5.place_market_order.assert_not_called()

    def test_only_bars_after_activated_at_considered(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        # Bar BEFORE activation has low=1897 (would trigger),
        # bars AFTER activation are all above. Should NOT fire.
        s = make_setup(
            direction="BUY",
            layer_2_price=1897.5,
            activated_at=datetime(2026, 5, 8, 12, 5, tzinfo=timezone.utc),
        )
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]

        df = make_ohlc(
            closes=[1900, 1900, 1899, 1898, 1899],
            lows=[1897, 1898, 1898, 1898, 1898],  # bar 0 low = 1897
            start="2026-05-08T12:00:00Z",
        )
        # Bar 0 is at 12:00 (before activation 12:05); bars 1+ are at 12:05+.
        # None of bars 1+ cross 1897.5.
        result = trigger.check_history(df)
        assert result == []

    def test_missing_columns_raises(
        self, trigger, mock_tracker, mock_supabase,
    ) -> None:
        df = pd.DataFrame({"close": [1900]})  # no high/low
        with pytest.raises(ValueError, match="'low' and 'high'"):
            trigger.check_history(df)


# --------------------------------------------------------------------------- #
# Comment format (parity with order_manager Layer 1)
# --------------------------------------------------------------------------- #


class TestComment:
    def test_market_order_comment_includes_layer_and_setup_id(
        self, trigger, mock_tracker, mock_supabase, mock_mt5,
    ) -> None:
        s = make_setup(direction="BUY", layer_2_price=1897.5)
        layer_2 = make_trade(setup_id=s.id, layer_number=2, status="WAITING")
        mock_tracker.get_active_setups.return_value = [s]
        mock_supabase.get_trades_for_setup.return_value = [layer_2]

        trigger.check_live(bid=1897.5, ask=1897.6)

        comment = mock_mt5.place_market_order.call_args.kwargs["comment"]
        # Default prefix "bot:trig", layer 2, first 8 chars of setup id.
        assert comment.startswith("bot:trig:L2:s=")
        assert str(s.id)[:8] in comment
        assert len(comment) <= 31
