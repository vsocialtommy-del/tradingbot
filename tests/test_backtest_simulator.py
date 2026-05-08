"""Tests for ``bot.backtest.simulator``.

The simulator owns all the broker-side semantics — slippage, spread,
SL/TP order, pending fills, partial closes. These tests pin down each
rule explicitly so changes there break the suite loudly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.backtest.simulator import (
    POINT_TO_DOLLARS,
    BacktestBroker,
    BacktestPosition,
    BrokerConfig,
    CloseReason,
    LayerFilled,
    OrderType,
    SLHit,
    Tick,
    TPHit,
    generate_ticks_from_bar,
)


NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def tick_at(mid: float, spread_points: float = 23.0, t: datetime = NOW) -> Tick:
    half = (spread_points * POINT_TO_DOLLARS) / 2.0
    return Tick(time=t, bid=mid - half, ask=mid + half)


# --------------------------------------------------------------------------- #
# generate_ticks_from_bar
# --------------------------------------------------------------------------- #


class TestGenerateTicks:
    def test_bullish_bar_emits_open_low_high_close(self) -> None:
        ticks = generate_ticks_from_bar(
            bar_time=NOW, open_=1900.0, high=1903.0, low=1899.0, close=1902.0,
        )
        assert len(ticks) == 4
        mids = [t.mid for t in ticks]
        assert mids == [1900.0, 1899.0, 1903.0, 1902.0]

    def test_bearish_bar_emits_open_high_low_close(self) -> None:
        ticks = generate_ticks_from_bar(
            bar_time=NOW, open_=1902.0, high=1903.0, low=1898.0, close=1899.0,
        )
        mids = [t.mid for t in ticks]
        assert mids == [1902.0, 1903.0, 1898.0, 1899.0]

    def test_doji_treated_as_bullish(self) -> None:
        # close == open: directional inference falls through to bullish.
        ticks = generate_ticks_from_bar(
            bar_time=NOW, open_=1900.0, high=1901.0, low=1899.0, close=1900.0,
        )
        mids = [t.mid for t in ticks]
        assert mids == [1900.0, 1899.0, 1901.0, 1900.0]

    def test_spread_applied_symmetrically(self) -> None:
        ticks = generate_ticks_from_bar(
            bar_time=NOW, open_=1900.0, high=1901.0, low=1899.0, close=1900.5,
            spread_points=10.0,  # $0.10 spread
        )
        # bid = mid - 0.05; ask = mid + 0.05.
        assert ticks[0].bid == pytest.approx(1899.95)
        assert ticks[0].ask == pytest.approx(1900.05)

    def test_invalid_bar_high_lt_low(self) -> None:
        with pytest.raises(ValueError, match="high"):
            generate_ticks_from_bar(
                bar_time=NOW, open_=1900.0, high=1899.0, low=1900.0, close=1899.5,
            )

    def test_invalid_bar_close_outside_range(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            generate_ticks_from_bar(
                bar_time=NOW, open_=1900.0, high=1901.0, low=1899.0, close=1905.0,
            )

    def test_ticks_are_ordered_in_time(self) -> None:
        ticks = generate_ticks_from_bar(
            bar_time=NOW, open_=1900.0, high=1901.0, low=1899.0, close=1900.5,
        )
        times = [t.time for t in ticks]
        assert times == sorted(times)
        assert times[0] == NOW


# --------------------------------------------------------------------------- #
# Market orders + slippage
# --------------------------------------------------------------------------- #


class TestMarketOrders:
    def test_buy_market_fills_at_ask_plus_slippage(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        # spread=23 ($0.23), slippage=1.5 ($0.015). Mid 1900 → ask 1900.115.
        # BUY fill = ask + slippage = 1900.13.
        t = tick_at(1900.0)
        pos = broker.place_market_order(
            direction="BUY", lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=t,
        )
        assert pos.entry_price == pytest.approx(1900.13)

    def test_sell_market_fills_at_bid_minus_slippage(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        # bid 1899.885 → SELL fill = bid - 0.015 = 1899.87.
        t = tick_at(1900.0)
        pos = broker.place_market_order(
            direction="SELL", lot_size=0.01, sl=1920.0, tp=1893.0,
            setup_id=1, layer=1, tick=t,
        )
        assert pos.entry_price == pytest.approx(1899.87)


# --------------------------------------------------------------------------- #
# Pending orders
# --------------------------------------------------------------------------- #


class TestPendingOrders:
    def test_buy_limit_triggers_when_ask_drops_to_limit(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_pending_order(
            direction="BUY", order_type=OrderType.BUY_LIMIT,
            price=1899.50, lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=1, layer=2, now=NOW,
        )
        # Tick mid 1900 → ask = 1900.115 → above limit → no fill.
        events = broker.process_tick(tick_at(1900.0))
        assert events == []
        assert len(broker.pending) == 1
        # Tick mid 1899.30 → ask = 1899.415 → below limit → fill.
        events = broker.process_tick(tick_at(1899.30))
        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, LayerFilled)
        # Pessimistic fill = limit + slippage.
        assert ev.fill_price == pytest.approx(1899.515)

    def test_sell_limit_triggers_when_bid_rises_to_limit(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_pending_order(
            direction="SELL", order_type=OrderType.SELL_LIMIT,
            price=1900.50, lot_size=0.01, sl=1920.0, tp=1893.0,
            setup_id=1, layer=2, now=NOW,
        )
        events = broker.process_tick(tick_at(1900.0))  # bid 1899.885 — no
        assert events == []
        events = broker.process_tick(tick_at(1900.70))  # bid 1900.585 — yes
        assert len(events) == 1
        # SELL limit fill = limit - slippage.
        assert events[0].fill_price == pytest.approx(1900.485)

    def test_cancel_all_pending_for_setup(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        for layer in (2, 3):
            broker.place_pending_order(
                direction="BUY", order_type=OrderType.BUY_LIMIT,
                price=1899.0, lot_size=0.01, sl=1880.0, tp=1907.0,
                setup_id=42, layer=layer, now=NOW,
            )
        # Different setup, should not be cancelled.
        broker.place_pending_order(
            direction="BUY", order_type=OrderType.BUY_LIMIT,
            price=1899.0, lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=99, layer=2, now=NOW,
        )
        cancelled = broker.cancel_all_pending_for_setup(42)
        assert cancelled == 2
        assert len(broker.pending) == 1
        assert next(iter(broker.pending.values())).setup_id == 99

    def test_pending_order_only_fills_once(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_pending_order(
            direction="BUY", order_type=OrderType.BUY_LIMIT,
            price=1899.50, lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=1, layer=2, now=NOW,
        )
        broker.process_tick(tick_at(1899.30))  # fills
        # Subsequent tick at the same level should not refill.
        events = broker.process_tick(tick_at(1899.20))
        # No layer fills; events are SL/TP-related only.
        assert not any(isinstance(e, LayerFilled) for e in events)


# --------------------------------------------------------------------------- #
# SL / TP behaviour
# --------------------------------------------------------------------------- #


class TestSLTPBehaviour:
    def test_sl_hit_on_buy_fills_below_sl(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        # Open BUY at ~1900.13.
        broker.place_market_order(
            direction="BUY", lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        # Drop to 1880 mid → bid 1879.885 → SL trigger.
        events = broker.process_tick(tick_at(1880.0))
        sl_events = [e for e in events if isinstance(e, SLHit)]
        assert len(sl_events) == 1
        # Pessimistic SL fill = sl - slippage = 1880 - 0.015 = 1879.985.
        assert sl_events[0].fill_price == pytest.approx(1879.985)

    def test_tp_hit_on_buy_fills_below_tp(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_market_order(
            direction="BUY", lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        events = broker.process_tick(tick_at(1907.5))  # bid 1907.385
        tp_events = [e for e in events if isinstance(e, TPHit)]
        assert len(tp_events) == 1
        # Pessimistic TP fill = tp - slippage = 1907 - 0.015 = 1906.985.
        assert tp_events[0].fill_price == pytest.approx(1906.985)

    def test_sl_hit_on_sell_fills_above_sl(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_market_order(
            direction="SELL", lot_size=0.01, sl=1920.0, tp=1893.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        events = broker.process_tick(tick_at(1920.0))  # ask 1920.115
        sl_events = [e for e in events if isinstance(e, SLHit)]
        assert len(sl_events) == 1
        assert sl_events[0].fill_price == pytest.approx(1920.015)

    def test_tp_hit_partial_close_50_percent(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_market_order(
            direction="BUY", lot_size=0.10, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        events = broker.process_tick(tick_at(1907.5))
        tp_events = [e for e in events if isinstance(e, TPHit)]
        assert len(tp_events) == 1
        assert tp_events[0].closed_lots == pytest.approx(0.05)
        # Position still open with remaining 0.05 lots.
        remaining = list(broker.positions.values())
        assert len(remaining) == 1
        assert remaining[0].lot_size == pytest.approx(0.05)
        assert remaining[0].status == "PARTIAL"

    def test_modify_position_sl_updates(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        pos = broker.place_market_order(
            direction="BUY", lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        broker.modify_position(pos.ticket, sl=pos.entry_price)
        assert broker.positions[pos.ticket].sl == pytest.approx(pos.entry_price)


class TestSLTPSameBarConflict:
    def test_sl_wins_when_same_tick_crosses_both(self) -> None:
        # Configure a position where the very next tick happens to
        # straddle both SL and TP — engineered via tight band.
        broker = BacktestBroker(starting_balance=10_000.0)
        pos = broker.place_market_order(
            direction="BUY", lot_size=0.01, sl=1899.0, tp=1901.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        # Tick that crosses both (synthetic; in practice it'd be split
        # across multiple ticks of an OHLC walk, but engine could ask
        # us to evaluate a tick that does both — pessimism still applies):
        # bid 1898.5 → SL hit; ask 1901.5 → TP would hit.
        # process_tick calls _check_sl_hits FIRST, closes the position.
        # By the time _check_tp_hits runs, the position is no longer
        # in self.positions, so TP doesn't fire.
        events = broker.process_tick(Tick(
            time=NOW, bid=1898.5, ask=1901.5,
        ))
        kinds = [type(e).__name__ for e in events]
        assert kinds == ["SLHit"]
        # And the position is gone.
        assert pos.ticket not in broker.positions


# --------------------------------------------------------------------------- #
# Process-tick order: SL → TP → Pending
# --------------------------------------------------------------------------- #


class TestProcessTickOrder:
    def test_sl_fires_before_layer_trigger_in_same_tick(self) -> None:
        # Layer-1 BUY position with SL at 1885.
        # Pending BUY_LIMIT for layer 2 at 1890.
        # A tick that crashes to 1880 → both SL and pending limit cross.
        # But SL must fire first (we don't open layer 2 into a stopped-out setup).
        broker = BacktestBroker(starting_balance=10_000.0)
        pos = broker.place_market_order(
            direction="BUY", lot_size=0.01, sl=1885.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        broker.place_pending_order(
            direction="BUY", order_type=OrderType.BUY_LIMIT,
            price=1890.0, lot_size=0.01, sl=1885.0, tp=1907.0,
            setup_id=1, layer=2, now=NOW,
        )
        events = broker.process_tick(tick_at(1880.0))
        kinds = [type(e).__name__ for e in events]
        # SL hit must come first; the layer fills *after* (the broker
        # doesn't know about the cascade-cancel rule — engine handles
        # that). What we assert here is the ORDER.
        assert kinds[0] == "SLHit"


# --------------------------------------------------------------------------- #
# P&L and balance accounting
# --------------------------------------------------------------------------- #


class TestPnL:
    def test_buy_winner_pnl_correct(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        # Open BUY at ~1900.13 (mid 1900 + half spread + slippage).
        broker.place_market_order(
            direction="BUY", lot_size=0.10, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        # TP at 1907 → fill ~1906.985 (TP minus slippage).
        events = broker.process_tick(tick_at(1907.5))
        tp = [e for e in events if isinstance(e, TPHit)][0]
        # Closed 0.05 lots at $1906.985 from $1900.13 entry =
        #   delta * 100 * 0.05 = 6.855 * 5 = 34.275 gross
        #   - commission 3.50 * 0.05 = 0.175
        #   = 34.10 net
        assert tp.pnl == pytest.approx(34.10, abs=0.05)

    def test_sl_loser_pnl_correct(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_market_order(
            direction="BUY", lot_size=0.10, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        events = broker.process_tick(tick_at(1879.5))  # SL hit
        sl = [e for e in events if isinstance(e, SLHit)][0]
        # Entry ~1900.13, SL fill ~1879.985.
        # gross = (1879.985 - 1900.13) * 0.10 * 100 = -201.45
        # commission = $3.50 per lot * 0.10 = $0.35
        # net = -201.80
        assert sl.pnl == pytest.approx(-201.80, abs=0.05)

    def test_balance_updates_on_close(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_market_order(
            direction="BUY", lot_size=0.01, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        starting = broker.balance
        broker.process_tick(tick_at(1907.5))  # TP1 partial
        # Balance should differ from starting after the partial.
        assert broker.balance != starting

    def test_equity_marks_unrealised_pnl(self) -> None:
        broker = BacktestBroker(starting_balance=10_000.0)
        broker.place_market_order(
            direction="BUY", lot_size=0.10, sl=1880.0, tp=1907.0,
            setup_id=1, layer=1, tick=tick_at(1900.0),
        )
        # Mid moves up; equity should rise above balance.
        broker.process_tick(tick_at(1905.0))
        assert broker.equity > broker.balance


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


class TestDefaults:
    def test_broker_config_defaults(self) -> None:
        c = BrokerConfig()
        assert c.spread_points == 23.0
        assert c.slippage_points == 1.5
        assert c.commission_per_lot == 3.50
        assert c.sl_tp_conflict_sl_wins is True
        assert c.contract_size == 100.0
        assert c.tp1_close_fraction == 0.5
