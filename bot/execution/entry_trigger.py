"""Entry trigger — fires Layer 2/3 market orders when price reaches the trigger.

For each ACTIVE setup with WAITING trades:

* **BUY zones**: WAITING trade fires when current ``bid <= trigger``.
* **SELL zones**: WAITING trade fires when current ``ask >= trigger``.

The trigger price for a layer is the setup's ``planned_layerN_price``
(set by ``order_manager`` at setup creation). Layer 2 = midpoint;
Layer 3 = far zone edge.

Two entry points
----------------

:meth:`EntryTrigger.check_live`
    Called per tick by the main loop. Reads ``current bid/ask``, fires
    any WAITING trade whose trigger condition is currently met.

:meth:`EntryTrigger.check_history`
    Called once at startup with the OHLC bars covering the bot's
    downtime. Catches up any WAITING trade whose trigger crossed
    during downtime by inspecting bar lows/highs since
    ``setup.activated_at``.

    **Catch-up fires at the current market price**, not the historical
    trigger price. Documented in PR #15 — the trade-off is a possibly
    worse fill than the trigger called for, accepted in exchange for
    not missing legitimate triggers when the bot was offline.

Both paths share :meth:`EntryTrigger._fire_layer` — places a market
order via ``mt5_connector``, transitions the trade ``WAITING → FILLED``
through ``position_tracker`` (which validates the state transition).

Design decisions called out in PR #15
-------------------------------------

1. **Trigger condition is symmetric live vs history.** Live uses
   ``bid <= trigger`` (BUY); history uses ``min(bar.low) <= trigger``.
   Same condition, different data source.

2. **No special "missed it, don't chase" logic** between layers within
   a single setup. The user trades layered entries — they want all
   layers filled if the price reaches them, even if it's a fast move
   through multiple triggers in one tick.

3. **MT5 failures do not transition the trade.** If
   ``place_market_order`` raises, the trade stays WAITING and the
   error is logged. Next tick will retry. This matches the bot's
   "best-effort, eventual consistency" philosophy.

4. **Cancellation is handled by ``position_tracker.update_setup_status``.**
   When the parent setup transitions to a terminal state (STOPPED_OUT,
   CLOSED, SKIPPED), the cascade-cancel logic flips WAITING trades to
   CANCELLED. ``EntryTrigger`` doesn't need to check setup terminality
   on every tick — it only operates on ACTIVE setups, and once a setup
   leaves ACTIVE its WAITING trades are already cancelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import pandas as pd
from loguru import logger

from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import PositionTracker
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


@dataclass(frozen=True)
class FiredTrigger:
    """Output for each layer that just fired."""

    setup_id: UUID
    trade_id: UUID
    layer_number: int
    mt5_ticket: int
    fill_price: float


@dataclass(frozen=True)
class EntryTriggerConfig:
    symbol: str = "XAUUSD"
    comment_prefix: str = "bot:trig"


class EntryTrigger:
    """Fires Layer 2/3 market orders when their trigger price is reached."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        position_tracker: PositionTracker,
        config: EntryTriggerConfig | None = None,
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        self._tracker = position_tracker
        self._config = config or EntryTriggerConfig()

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #

    def check_live(self, bid: float, ask: float) -> list[FiredTrigger]:
        """Check all ACTIVE setups for triggered layers at the current tick."""
        fired: list[FiredTrigger] = []
        for setup in self._tracker.get_active_setups():
            if setup.status != "ACTIVE":
                # PENDING setups have no Layer 1 fill yet (so Layer 2/3
                # triggers shouldn't fire); TP1_HIT means we're past
                # entry phase.
                continue
            for trade in self._supabase.get_trades_for_setup(setup.id):
                if trade.status != "WAITING":
                    continue
                trigger = _trigger_price(setup, trade)
                if _trigger_met_live(setup, trigger, bid, ask):
                    result = self._fire_layer(setup, trade)
                    if result is not None:
                        fired.append(result)
        return fired

    def check_history(self, df: pd.DataFrame) -> list[FiredTrigger]:
        """Startup catch-up. Fires WAITING trades whose trigger crossed
        during downtime, inferred from OHLC bar lows/highs.

        Fires at *current* market price (read via MT5), not the
        historical trigger.
        """
        if "low" not in df.columns or "high" not in df.columns:
            raise ValueError("df must have 'low' and 'high' columns")

        fired: list[FiredTrigger] = []
        for setup in self._tracker.get_active_setups():
            if setup.status != "ACTIVE":
                continue
            bars = _bars_since_activation(setup, df)
            for trade in self._supabase.get_trades_for_setup(setup.id):
                if trade.status != "WAITING":
                    continue
                trigger = _trigger_price(setup, trade)
                if _trigger_met_history(setup, trigger, bars):
                    result = self._fire_layer(setup, trade)
                    if result is not None:
                        fired.append(result)
        return fired

    # ----------------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------------- #

    def _fire_layer(
        self, setup: Setup, trade: Trade
    ) -> FiredTrigger | None:
        """Place market order for the layer, update trade row to FILLED."""
        comment = (
            f"{self._config.comment_prefix}:L{trade.layer_number}"
            f":s={str(setup.id)[:8]}"
        )
        try:
            ticket = self._mt5.place_market_order(
                symbol=self._config.symbol,
                direction=setup.direction,
                lot_size=float(trade.lot_size),
                sl=float(trade.sl_price),
                tp=None,  # bot-managed TP1
                comment=comment,
            )
        except Exception:
            logger.exception(
                f"entry_trigger: failed to fire layer {trade.layer_number} "
                f"for setup {setup.id}"
            )
            return None

        fill_price = self._resolve_fill_price(ticket)

        try:
            self._tracker.update_trade_status(
                trade.id,
                "FILLED",
                entry_price=fill_price,
                mt5_ticket=ticket,
            )
        except Exception:
            logger.exception(
                f"entry_trigger: trade row update failed for {trade.id} "
                f"(broker has the position but Supabase doesn't show FILLED)"
            )
            # Don't return None — the layer DID fire on the broker.
            # Reconciliation will fix the bookkeeping.

        logger.info(
            f"entry_trigger fired: setup={setup.id} layer={trade.layer_number} "
            f"ticket={ticket} fill={fill_price}"
        )
        return FiredTrigger(
            setup_id=setup.id,
            trade_id=trade.id,
            layer_number=trade.layer_number,
            mt5_ticket=ticket,
            fill_price=fill_price if fill_price is not None else 0.0,
        )

    def _resolve_fill_price(self, ticket: int) -> float | None:
        try:
            positions = self._mt5.get_open_positions(symbol=self._config.symbol)
        except Exception:
            logger.exception("entry_trigger: get_open_positions failed")
            return None
        for p in positions:
            if p.get("ticket") == ticket:
                price = p.get("price_open")
                if price is not None:
                    return float(price)
        return None


# --------------------------------------------------------------------------- #
# Module-level helpers (testable without an EntryTrigger instance)
# --------------------------------------------------------------------------- #


def _trigger_price(setup: Setup, trade: Trade) -> float:
    """Map a trade's layer_number to its trigger price on the setup."""
    if trade.layer_number == 1:
        return float(setup.planned_layer1_price)
    if trade.layer_number == 2:
        return float(setup.planned_layer2_price)
    if trade.layer_number == 3:
        return float(setup.planned_layer3_price)
    raise ValueError(f"unknown layer_number: {trade.layer_number}")


def _trigger_met_live(
    setup: Setup, trigger: float, bid: float, ask: float
) -> bool:
    """BUY fires when bid <= trigger; SELL when ask >= trigger.

    Inclusive at the trigger price (boundary fires).
    """
    if setup.direction == "BUY":
        return bid <= trigger
    return ask >= trigger


def _trigger_met_history(
    setup: Setup, trigger: float, bars: pd.DataFrame
) -> bool:
    """Bar-history catch-up: any bar's low <= trigger (BUY) / high >= trigger (SELL)."""
    if len(bars) == 0:
        return False
    if setup.direction == "BUY":
        return bool((bars["low"] <= trigger).any())
    return bool((bars["high"] >= trigger).any())


def _bars_since_activation(setup: Setup, df: pd.DataFrame) -> pd.DataFrame:
    """Slice df to bars at-or-after setup.activated_at; or all bars if no timestamp."""
    if setup.activated_at is None:
        return df
    try:
        return df[df.index >= setup.activated_at]
    except Exception:
        # Index might not be timezone-comparable to activated_at; fall back.
        logger.warning(
            f"entry_trigger: couldn't slice bars by activated_at "
            f"({setup.activated_at}); using full df"
        )
        return df
