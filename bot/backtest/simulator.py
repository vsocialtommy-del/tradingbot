"""Synthetic broker + tick generator for the backtest engine.

The :class:`BacktestBroker` is the only I/O surface the engine talks
to — it owns the open positions, pending orders, balance, and equity
curve. It is **not** a stand-in for ``MT5Connector``; it has its own
explicit API tailored to backtest needs (e.g. ``process_tick`` returns
a list of events the engine can consume directly).

Tick generation
---------------

Real-tick data isn't available without paying. We approximate intra-bar
behaviour with the **OHLC walk method** (industry-standard MT4/MT5
strategy-tester fallback):

* Bullish bar (``close >= open``) → ``open → low → high → close``
* Bearish bar (``close < open``)  → ``open → high → low → close``

Four ticks per M5 bar. This minimises false TP1 hits in bearish bars
and false SL hits in bullish bars; it matches what intra-bar action
typically looks like at 5-min resolution. It is **approximate** — real
tick paths can U-turn arbitrarily within a bar — but is the standard
fallback when only OHLC is available.

Pessimism layer
---------------

Documented in the spec discussion. Every fill is biased one slippage
unit *worse for the trader* than the theoretical price:

* BUY entry / BUY pending fill / SL on BUY (price moved down)
  / TP on BUY (price moved up):
  add slippage_dollars to a level that's protective (BUY entry, BUY TP)
  or subtract from one that's adverse (BUY SL → fills below the SL).
* SELL entries / fills are mirrored.

In practice all of this collapses to a simple rule: **slippage moves
the fill price in the same direction price was already moving**. SL
fills below SL on a BUY (worse loss); TP fills below TP on a BUY
(less profit).

Same-bar SL+TP conflict
-----------------------

When a bar's range covers both the SL and the TP1 levels of a
position, **SL wins** (pessimism). Configurable via
``BrokerConfig.sl_tp_conflict_sl_wins=False`` for sensitivity testing.

Per-tick check order
--------------------

Inside :meth:`BacktestBroker.process_tick`:

  1. SL hits on open positions          (close losing trades first)
  2. TP1 hits on open positions          (lock in profits next)
  3. Pending order triggers             (open new layers last)

Reasoning: if a single tick crosses both an SL level and a layer-trigger
level, SL must fire first (we don't open a fresh layer into a position
that's about to be stopped out). Similarly, if a tick crosses TP1 and
a layer-trigger level, the existing position takes the partial close
before any new layer is opened.

Units
-----

* ``spread_points`` and ``slippage_points`` use the **broker
  convention** (1 point = $0.01 for XAUUSD). Default 23 = $0.23
  matches Vantage Gold raw spread; 1.5 = $0.015 a typical slippage.
* ``sl_buffer_points``, ``layer_*_offset_points``, etc. (in
  :class:`BacktestConfig`) use **price units** — 17.5 means $17.50.
  Inherited from the live ``bot_config`` convention; see the engine
  docstring for the unfortunate dual convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Iterator, Literal


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# 1 broker point = $0.01 for XAUUSD. Used to convert spread/slippage.
POINT_TO_DOLLARS: float = 0.01

# Price epsilon for boundary comparisons. ½ a broker point = $0.005.
PRICE_EPSILON: float = 0.005


# --------------------------------------------------------------------------- #
# Enums + dataclasses
# --------------------------------------------------------------------------- #


class OrderType(str, Enum):
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class CloseReason(str, Enum):
    SL = "SL"
    TP1 = "TP1"
    MANUAL = "MANUAL"
    END_OF_DATA = "END_OF_DATA"


@dataclass(frozen=True)
class Tick:
    time: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class BacktestPosition:
    """An open or closed market position."""

    ticket: int
    setup_id: int
    layer: int
    direction: Literal["BUY", "SELL"]
    entry_price: float
    lot_size: float
    sl: float
    tp: float | None
    opened_at: datetime
    # Mutable as the position evolves.
    status: Literal["OPEN", "CLOSED", "PARTIAL"] = "OPEN"
    closed_lots: float = 0.0  # for partial closes (TP1 = 50%)
    exit_price: float | None = None
    exit_time: datetime | None = None
    close_reason: CloseReason | None = None
    realised_pnl: float = 0.0  # cumulative realised P&L (commission-net)
    commission_paid: float = 0.0


@dataclass
class PendingOrder:
    ticket: int
    setup_id: int
    layer: int
    direction: Literal["BUY", "SELL"]
    order_type: OrderType
    price: float  # trigger price
    sl: float
    tp: float | None
    lot_size: float
    created_at: datetime
    status: Literal["PENDING", "FILLED", "CANCELLED"] = "PENDING"


# Events returned by process_tick — engine consumes these.

@dataclass(frozen=True)
class SLHit:
    ticket: int
    setup_id: int
    layer: int
    fill_price: float
    pnl: float


@dataclass(frozen=True)
class TPHit:
    ticket: int
    setup_id: int
    layer: int
    fill_price: float
    closed_lots: float
    pnl: float


@dataclass(frozen=True)
class LayerFilled:
    ticket: int
    setup_id: int
    layer: int
    fill_price: float


BrokerEvent = SLHit | TPHit | LayerFilled


@dataclass(frozen=True)
class BrokerConfig:
    spread_points: float = 23.0
    slippage_points: float = 1.5
    commission_per_lot: float = 3.50  # $ per lot, charged on close
    sl_tp_conflict_sl_wins: bool = True
    """Same-bar SL+TP conflict: SL wins by default (pessimistic).
    Flip to False for sensitivity testing."""
    contract_size: float = 100.0  # XAUUSD: 100 oz per lot
    """For P&L calc: ``pnl = (exit - entry) * direction * lots * 100``."""
    tp1_close_fraction: float = 0.5  # close 50% of lot on TP1


# --------------------------------------------------------------------------- #
# Tick generator
# --------------------------------------------------------------------------- #


def generate_ticks_from_bar(
    bar_time: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    *,
    spread_points: float = 23.0,
    bar_duration_minutes: int = 5,
) -> list[Tick]:
    """Emit 4 ticks per bar via OHLC walk with directional inference.

    See module docstring. Ticks are evenly spaced inside the bar at
    minutes 0, 1, 2, 3 (last reserved by callers as the bar-close
    boundary).
    """
    if high < low:
        raise ValueError(f"bar at {bar_time}: high {high} < low {low}")
    if not (low <= open_ <= high) or not (low <= close <= high):
        raise ValueError(
            f"bar at {bar_time}: open/close outside [low, high]"
        )

    # Directional inference.
    if close >= open_:
        path = [open_, low, high, close]  # bullish: dip then rally
    else:
        path = [open_, high, low, close]  # bearish: bounce then drop

    half_spread = (spread_points * POINT_TO_DOLLARS) / 2.0
    step = timedelta(minutes=bar_duration_minutes // 4 or 1)
    return [
        Tick(
            time=bar_time + step * i,
            bid=mid - half_spread,
            ask=mid + half_spread,
        )
        for i, mid in enumerate(path)
    ]


# --------------------------------------------------------------------------- #
# Broker
# --------------------------------------------------------------------------- #


class BacktestBroker:
    """Synthetic broker. Owns positions, pending orders, balance, equity."""

    def __init__(
        self,
        starting_balance: float,
        config: BrokerConfig | None = None,
    ) -> None:
        self.config = config or BrokerConfig()
        self.balance: float = starting_balance
        self.equity: float = starting_balance
        self.positions: dict[int, BacktestPosition] = {}
        self.pending: dict[int, PendingOrder] = {}
        self.closed_positions: list[BacktestPosition] = []
        self._next_ticket: int = 100_000
        self._slippage_d: float = (
            self.config.slippage_points * POINT_TO_DOLLARS
        )

    # ------------------------------------------------------------------ #
    # Order placement
    # ------------------------------------------------------------------ #

    def place_market_order(
        self,
        *,
        direction: Literal["BUY", "SELL"],
        lot_size: float,
        sl: float,
        tp: float | None,
        setup_id: int,
        layer: int,
        tick: Tick,
    ) -> BacktestPosition:
        """Open a position immediately at the current tick.

        Pessimistic fill: BUY at ``ask + slippage``; SELL at ``bid - slippage``.
        """
        if direction == "BUY":
            fill = tick.ask + self._slippage_d
        else:
            fill = tick.bid - self._slippage_d
        ticket = self._issue_ticket()
        pos = BacktestPosition(
            ticket=ticket,
            setup_id=setup_id,
            layer=layer,
            direction=direction,
            entry_price=fill,
            lot_size=lot_size,
            sl=sl,
            tp=tp,
            opened_at=tick.time,
        )
        self.positions[ticket] = pos
        return pos

    def place_pending_order(
        self,
        *,
        direction: Literal["BUY", "SELL"],
        order_type: OrderType,
        price: float,
        lot_size: float,
        sl: float,
        tp: float | None,
        setup_id: int,
        layer: int,
        now: datetime,
    ) -> PendingOrder:
        ticket = self._issue_ticket()
        order = PendingOrder(
            ticket=ticket,
            setup_id=setup_id,
            layer=layer,
            direction=direction,
            order_type=order_type,
            price=price,
            sl=sl,
            tp=tp,
            lot_size=lot_size,
            created_at=now,
        )
        self.pending[ticket] = order
        return order

    def cancel_pending(self, ticket: int) -> None:
        order = self.pending.get(ticket)
        if order is None:
            return
        order.status = "CANCELLED"
        del self.pending[ticket]

    def cancel_all_pending_for_setup(self, setup_id: int) -> int:
        """Cancel every pending order belonging to ``setup_id``. Returns count."""
        tickets = [
            t for t, o in self.pending.items()
            if o.setup_id == setup_id and o.status == "PENDING"
        ]
        for t in tickets:
            self.cancel_pending(t)
        return len(tickets)

    # ------------------------------------------------------------------ #
    # Modify / close
    # ------------------------------------------------------------------ #

    def modify_position(
        self, ticket: int, *, sl: float | None = None, tp: float | None = None,
    ) -> None:
        pos = self.positions.get(ticket)
        if pos is None:
            return
        if sl is not None:
            pos.sl = sl
        if tp is not None:
            pos.tp = tp

    def close_position(
        self,
        ticket: int,
        *,
        tick: Tick,
        fraction: float = 1.0,
        reason: CloseReason = CloseReason.MANUAL,
    ) -> tuple[BacktestPosition, float]:
        """Close all (or a fraction) of a position at the current tick.

        Returns the position (mutated) and the realised P&L from this
        close. Commission is charged on the closed lots.
        """
        pos = self.positions.get(ticket)
        if pos is None:
            raise ValueError(f"unknown ticket: {ticket}")
        if not 0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]: {fraction}")
        # Pessimistic close: BUY closes at bid - slippage; SELL at ask + slippage.
        if pos.direction == "BUY":
            exit_price = tick.bid - self._slippage_d
        else:
            exit_price = tick.ask + self._slippage_d
        return self._close_at(
            pos, exit_price=exit_price, exit_time=tick.time,
            fraction=fraction, reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Per-tick processing
    # ------------------------------------------------------------------ #

    def process_tick(self, tick: Tick) -> list[BrokerEvent]:
        """Step the broker forward by one tick.

        Order:
          1. SL hits         (close losing trades first)
          2. Cascade-cancel pending orders for any setup whose SL hit
          3. TP hits         (lock in profits)
          4. Cascade-cancel pending orders for any setup whose TP hit
          5. Pending fills   (only those not cascade-cancelled above)

        The two cascade steps are baked into the broker (not the engine
        event handler) because if the SL and a deeper-layer pending fill
        both trigger on the same tick, the cascade has to win the race.
        Doing it in the engine handler runs after the pending fills
        already happened. See PR description.
        """
        events: list[BrokerEvent] = []
        # 1. SL hits.
        sl_events = self._check_sl_hits(tick)
        events.extend(sl_events)
        # 2. Cascade-cancel pending for setups whose SL hit.
        for e in sl_events:
            self.cancel_all_pending_for_setup(e.setup_id)
        # 3. TP hits (positions that survived SL).
        tp_events = self._check_tp_hits(tick)
        events.extend(tp_events)
        # 4. Cascade-cancel pending for setups whose TP hit.
        for e in tp_events:
            self.cancel_all_pending_for_setup(e.setup_id)
        # 5. Remaining pending order triggers.
        events.extend(self._check_pending_fills(tick))
        # Recompute equity at tick close.
        self._recompute_equity(tick)
        return events

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _issue_ticket(self) -> int:
        t = self._next_ticket
        self._next_ticket += 1
        return t

    def _check_sl_hits(self, tick: Tick) -> list[BrokerEvent]:
        events: list[BrokerEvent] = []
        # Snapshot — close_at mutates self.positions.
        for ticket in list(self.positions.keys()):
            pos = self.positions[ticket]
            if pos.status == "CLOSED":
                continue
            hit, fill = self._sl_hit(pos, tick)
            if not hit:
                continue
            _, pnl = self._close_at(
                pos, exit_price=fill, exit_time=tick.time,
                fraction=1.0, reason=CloseReason.SL,
            )
            events.append(SLHit(
                ticket=pos.ticket, setup_id=pos.setup_id, layer=pos.layer,
                fill_price=fill, pnl=pnl,
            ))
        return events

    def _check_tp_hits(self, tick: Tick) -> list[BrokerEvent]:
        events: list[BrokerEvent] = []
        for ticket in list(self.positions.keys()):
            pos = self.positions[ticket]
            if pos.status != "OPEN" or pos.tp is None:
                continue
            hit, fill = self._tp_hit(pos, tick)
            if not hit:
                continue
            # Capture lot_size BEFORE _close_at mutates it on partial.
            closed_lots = pos.lot_size * self.config.tp1_close_fraction
            _, pnl = self._close_at(
                pos, exit_price=fill, exit_time=tick.time,
                fraction=self.config.tp1_close_fraction,
                reason=CloseReason.TP1,
            )
            events.append(TPHit(
                ticket=pos.ticket, setup_id=pos.setup_id, layer=pos.layer,
                fill_price=fill,
                closed_lots=closed_lots,
                pnl=pnl,
            ))
        return events

    def _check_pending_fills(self, tick: Tick) -> list[BrokerEvent]:
        events: list[BrokerEvent] = []
        for ticket in list(self.pending.keys()):
            order = self.pending[ticket]
            if order.status != "PENDING":
                continue
            triggered, fill = self._pending_triggered(order, tick)
            if not triggered:
                continue
            order.status = "FILLED"
            del self.pending[ticket]
            new_ticket = self._issue_ticket()
            pos = BacktestPosition(
                ticket=new_ticket,
                setup_id=order.setup_id,
                layer=order.layer,
                direction=order.direction,
                entry_price=fill,
                lot_size=order.lot_size,
                sl=order.sl,
                tp=order.tp,
                opened_at=tick.time,
            )
            self.positions[new_ticket] = pos
            events.append(LayerFilled(
                ticket=new_ticket, setup_id=order.setup_id,
                layer=order.layer, fill_price=fill,
            ))
        return events

    def _sl_hit(self, pos: BacktestPosition, tick: Tick) -> tuple[bool, float]:
        """Detect SL trigger at this tick.

        BUY SL = below entry; hits when bid <= sl. Pessimistic fill at
        ``sl - slippage`` (further loss).
        SELL SL = above entry; hits when ask >= sl. Pessimistic fill at
        ``sl + slippage`` (further loss).
        """
        if pos.direction == "BUY":
            if tick.bid <= pos.sl + PRICE_EPSILON:
                return True, pos.sl - self._slippage_d
        else:
            if tick.ask >= pos.sl - PRICE_EPSILON:
                return True, pos.sl + self._slippage_d
        return False, 0.0

    def _tp_hit(self, pos: BacktestPosition, tick: Tick) -> tuple[bool, float]:
        """Detect TP trigger at this tick.

        BUY TP = above entry; hits when bid >= tp. Pessimistic fill at
        ``tp - slippage`` (smaller profit).
        SELL TP = below entry; hits when ask <= tp. Pessimistic fill at
        ``tp + slippage`` (smaller profit).

        SL+TP same-tick conflict: handled by the order in
        :meth:`process_tick` — SL is checked first, so any position whose
        SL hit this tick is already CLOSED before we get here. Cross-bar
        same-tick conflict resolution flag preserved here for future use.
        """
        assert pos.tp is not None  # filtered upstream
        if pos.direction == "BUY":
            if tick.bid >= pos.tp - PRICE_EPSILON:
                return True, pos.tp - self._slippage_d
        else:
            if tick.ask <= pos.tp + PRICE_EPSILON:
                return True, pos.tp + self._slippage_d
        return False, 0.0

    def _pending_triggered(
        self, order: PendingOrder, tick: Tick,
    ) -> tuple[bool, float]:
        """Detect pending-order trigger at this tick.

        Pessimistic fill: trigger price + slippage in the adverse
        direction (so BUY fills slightly above the limit, SELL slightly
        below).
        """
        if order.order_type == OrderType.BUY_LIMIT:
            # Fills when ask drops to the limit.
            if tick.ask <= order.price + PRICE_EPSILON:
                return True, order.price + self._slippage_d
        elif order.order_type == OrderType.SELL_LIMIT:
            if tick.bid >= order.price - PRICE_EPSILON:
                return True, order.price - self._slippage_d
        elif order.order_type == OrderType.BUY_STOP:
            if tick.ask >= order.price - PRICE_EPSILON:
                return True, order.price + self._slippage_d
        elif order.order_type == OrderType.SELL_STOP:
            if tick.bid <= order.price + PRICE_EPSILON:
                return True, order.price - self._slippage_d
        return False, 0.0

    def _close_at(
        self,
        pos: BacktestPosition,
        *,
        exit_price: float,
        exit_time: datetime,
        fraction: float,
        reason: CloseReason,
    ) -> tuple[BacktestPosition, float]:
        """Apply a (partial) close, update balance, return realised P&L.

        Partial closes record a *shadow* snapshot in
        ``closed_positions`` (a frozen copy of the closed portion).
        This way every realised lot — including the TP1 50% — appears
        as a discrete entry that metrics can count. The original
        position stays open with the remaining lot_size; its
        ``realised_pnl`` resets so the final close reports only its
        own contribution (the partial's P&L is already on the shadow).
        """
        closed_lots = pos.lot_size * fraction
        sign = 1.0 if pos.direction == "BUY" else -1.0
        gross_pnl = (
            (exit_price - pos.entry_price) * sign
            * closed_lots * self.config.contract_size
        )
        commission = self.config.commission_per_lot * closed_lots
        net_pnl = gross_pnl - commission
        self.balance += net_pnl

        if fraction >= 1.0 - 1e-9:
            pos.status = "CLOSED"
            pos.exit_price = exit_price
            pos.exit_time = exit_time
            pos.close_reason = reason
            pos.realised_pnl = net_pnl
            pos.commission_paid = commission
            pos.closed_lots = closed_lots
            self.closed_positions.append(pos)
            del self.positions[pos.ticket]
            return pos, net_pnl

        # Partial close: shadow + reduce original.
        shadow = BacktestPosition(
            ticket=pos.ticket,
            setup_id=pos.setup_id,
            layer=pos.layer,
            direction=pos.direction,
            entry_price=pos.entry_price,
            lot_size=closed_lots,
            sl=pos.sl,
            tp=pos.tp,
            opened_at=pos.opened_at,
            status="CLOSED",
            closed_lots=closed_lots,
            exit_price=exit_price,
            exit_time=exit_time,
            close_reason=reason,
            realised_pnl=net_pnl,
            commission_paid=commission,
        )
        self.closed_positions.append(shadow)
        pos.status = "PARTIAL"
        pos.lot_size -= closed_lots
        return pos, net_pnl

    def _recompute_equity(self, tick: Tick) -> None:
        """Equity = balance + unrealised P&L on open positions at this tick."""
        unreal = 0.0
        for pos in self.positions.values():
            if pos.status != "OPEN" and pos.status != "PARTIAL":
                continue
            if pos.direction == "BUY":
                # Mark to bid (as if closing).
                m2m = (tick.bid - pos.entry_price) * pos.lot_size * (
                    self.config.contract_size
                )
            else:
                m2m = (pos.entry_price - tick.ask) * pos.lot_size * (
                    self.config.contract_size
                )
            unreal += m2m
        self.equity = self.balance + unreal
