"""XAUTISM layer 2: broker stop order at the zone edge (Nov 2026, step 6).

Layer 1 (0.01) goes in on first touch of the zone. Layer 2 (the rest of
the size) goes in as price comes back *out* of the zone, through the
same edge. To make that fill even if the bot lags, layer 2 lives at the
broker as a pending order:

* BUY zone:  BUY STOP  at the zone top.
* SELL zone: SELL STOP at the zone bottom.

A buy stop has to sit above the current ask (a sell stop below the bid),
so it can only be placed once price is inside the zone by at least the
minimum distance. If price just tags the edge and leaves, layer 2 is
never placed and only the 0.01 runs.

Layer 2 carries the same SL and TP as layer 1, both at the broker.

Per-tick state machine for a WAITING ``order_type='STOP'`` layer 2:

=====================  ==============================================
mt5_ticket is None     Not placed yet. Place it once price is deep
                       enough. Cancel (CASCADE_CANCELLED) if layer 1
                       has already closed (TP or SL hit first).
ticket in pending      Waiting at the broker. Cancel at the broker and
                       mark CANCELLED if layer 1 has closed.
ticket in positions    Triggered. Mark FILLED with the fill price.
ticket nowhere         Left the book between ticks. History says
                       FILLED → FILLED then CLOSED (it already hit SL
                       or TP); otherwise CANCELLED.
=====================  ==============================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from loguru import logger

from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import OPEN_TRADE_STATUSES, PositionTracker
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


@dataclass(frozen=True)
class XautismEntryConfig:
    """XAUTISM entry + management switches."""

    enabled: bool = True
    """0.01 first touch + layer 2 stop at the edge, SL and TP at the
    broker. False = the legacy 3-layer bot-fired model."""

    first_touch_lots: float = 0.01
    lot_step: float = 0.01

    layer2_min_distance: float = 0.10
    """Price must be at least this far inside the zone (on the right
    side of the stop) before layer 2 is sent. The broker's own
    ``trade_stops_level`` is added on top when it's larger."""

    m5_close_exit: bool = True
    """Close every layer (and cancel a pending layer 2) when an M5 candle
    closes beyond the far edge of the zone. Extra exit on top of the
    broker SL."""


@dataclass(frozen=True)
class Layer2Event:
    setup_id: UUID
    trade_id: UUID
    action: str  # PLACED / FILLED / CANCELLED / FILLED_AND_CLOSED
    ticket: int | None = None
    price: float | None = None


class Layer2StopManager:
    """Places, tracks and cancels the layer 2 stop orders."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        tracker: PositionTracker,
        config: XautismEntryConfig | None = None,
        *,
        symbol: str = "XAUUSD",
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        self._tracker = tracker
        self._config = config or XautismEntryConfig()
        self._symbol = symbol

    def check_live(
        self, bid: float, ask: float,
        config: XautismEntryConfig | None = None,
    ) -> list[Layer2Event]:
        if config is not None:
            self._config = config
        events: list[Layer2Event] = []
        positions: list[dict[str, Any]] | None = None
        pending: list[dict[str, Any]] | None = None

        for setup in self._tracker.get_active_setups():
            if setup.status != "ACTIVE":
                continue
            try:
                trades = self._supabase.get_trades_for_setup(setup.id)
            except Exception:
                logger.exception(f"layer2: get_trades failed for {setup.id}")
                continue
            l2 = next(
                (t for t in trades
                 if t.layer_number == 2 and t.order_type == "STOP"
                 and t.status == "WAITING"),
                None,
            )
            if l2 is None:
                continue
            l1 = next((t for t in trades if t.layer_number == 1), None)

            # Lazy broker reads: only when a setup actually needs them.
            if positions is None:
                positions = self._safe(self._mt5.get_open_positions, self._symbol)
                pending = self._safe(self._mt5.get_pending_orders, self._symbol)
            if positions is None or pending is None:
                return events  # broker read failed; retry next tick

            pos_by_ticket = {p["ticket"]: p for p in positions}
            pending_tickets = {o["ticket"] for o in pending}
            l1_open = (
                l1 is not None
                and l1.status in OPEN_TRADE_STATUSES
                and l1.mt5_ticket is not None
                and l1.mt5_ticket in pos_by_ticket
            )

            ev = self._handle(setup, l1_open, l2, pos_by_ticket,
                              pending_tickets, bid, ask)
            if ev is not None:
                events.append(ev)
        return events

    # ------------------------------------------------------------------ #

    def _handle(
        self,
        setup: Setup,
        l1_open: bool,
        l2: Trade,
        pos_by_ticket: dict[int, dict[str, Any]],
        pending_tickets: set[int],
        bid: float,
        ask: float,
    ) -> Layer2Event | None:
        ticket = l2.mt5_ticket

        # Not yet at the broker.
        if ticket is None:
            if not l1_open:
                return self._cancel(setup, l2, reason="CASCADE_CANCELLED")
            return self._maybe_place(setup, l2, bid, ask)

        # Triggered.
        if ticket in pos_by_ticket:
            price = pos_by_ticket[ticket].get("price_open")
            self._tracker.update_trade_status(
                l2.id, "FILLED",
                entry_price=float(price) if price is not None else None,
            )
            logger.info(f"layer2 FILLED: setup={setup.id} ticket={ticket} @ {price}")
            return Layer2Event(setup.id, l2.id, "FILLED", ticket,
                               float(price) if price is not None else None)

        # Still pending at the broker.
        if ticket in pending_tickets:
            if not l1_open:
                try:
                    self._mt5.cancel_pending_order(ticket)
                except Exception:
                    logger.exception(f"layer2: cancel of pending {ticket} failed")
                    return None  # retry next tick
                return self._cancel(setup, l2, reason="CASCADE_CANCELLED")
            return None

        # Left the book between ticks.
        state = self._safe(self._mt5.get_order_history_state, ticket)
        if state == "FILLED":
            # Filled and already closed by SL / TP before we saw it.
            self._tracker.update_trade_status(l2.id, "FILLED")
            self._tracker.update_trade_status(
                l2.id, "CLOSED", close_reason="MANUAL_CLOSE",
            )
            return Layer2Event(setup.id, l2.id, "FILLED_AND_CLOSED", ticket)
        return self._cancel(setup, l2, reason="MANUAL_CLOSE")

    def _maybe_place(
        self, setup: Setup, l2: Trade, bid: float, ask: float,
    ) -> Layer2Event | None:
        stop_price = float(setup.planned_layer2_price)
        broker_min = self._safe(self._mt5.get_min_stop_distance, self._symbol) or 0.0
        min_dist = max(self._config.layer2_min_distance, broker_min)

        if setup.direction == "BUY":
            ready = ask <= stop_price - min_dist
        else:
            ready = bid >= stop_price + min_dist
        if not ready:
            return None

        try:
            ticket = self._mt5.place_stop_order(
                symbol=self._symbol,
                direction=setup.direction,
                lot_size=float(l2.lot_size),
                price=stop_price,
                sl=float(l2.sl_price),
                tp=float(l2.tp_price) if l2.tp_price is not None else None,
                comment=f"bot:L2stop:s={str(setup.id)[:8]}",
            )
        except Exception:
            logger.exception(
                f"layer2: stop order failed for setup {setup.id}; retry next tick"
            )
            return None
        try:
            self._supabase.update_trade(l2.id, mt5_ticket=ticket)
        except Exception:
            logger.exception(
                f"layer2: stop {ticket} placed but trade row not updated "
                f"(setup {setup.id}); reconciliation needed"
            )
        logger.info(
            f"layer2 STOP placed: setup={setup.id} {setup.direction} "
            f"{l2.lot_size} lots @ {stop_price} ticket={ticket}"
        )
        return Layer2Event(setup.id, l2.id, "PLACED", ticket, stop_price)

    def _cancel(self, setup: Setup, l2: Trade, *, reason: str) -> Layer2Event | None:
        try:
            self._tracker.update_trade_status(l2.id, "CANCELLED", close_reason=reason)
        except Exception:
            logger.exception(f"layer2: cancel row update failed for {l2.id}")
            return None
        logger.info(f"layer2 CANCELLED ({reason}): setup={setup.id}")
        return Layer2Event(setup.id, l2.id, "CANCELLED", l2.mt5_ticket)

    @staticmethod
    def _safe(fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception:
            logger.exception(f"layer2: broker call {getattr(fn, '__name__', fn)} failed")
            return None


def cancel_layer2_at_broker(mt5: MT5Connector, trade: Trade) -> None:
    """Best-effort removal of a placed layer 2 stop (used by exits)."""
    if trade.order_type == "STOP" and trade.mt5_ticket is not None:
        try:
            mt5.cancel_pending_order(trade.mt5_ticket)
        except Exception:
            logger.exception(f"cancel of pending layer 2 {trade.mt5_ticket} failed")
