"""Zone-exit BE trigger — PR #47.

Fires on M5 close when the just-closed bar's body confirms the trade
direction by closing **out of the originating zone** past the L1 entry:

* BUY  setups → close > L1 entry  (= zone.top — price left going up)
* SELL setups → close < L1 entry  (= zone.bottom — price left going down)

Action on fire (per-setup, idempotent):

1. Close the **shallowest still-FILLED** layer (typically L1; if L1 is
   already terminal pick L2, etc.) at the current bid/ask.
   ``close_reason='ZONE_EXIT'`` (new — migration 012).
2. Modify SL to entry on every **remaining FILLED** layer (status
   stays FILLED, just the SL moves to the layer's own entry price —
   "break-even" protection).
3. Cancel every still-WAITING layer with
   ``close_reason='ZONE_EXIT_CANCELLED'``. Rationale: once price has
   confirmed direction by leaving the zone, the original "wait for
   deeper retest" thesis is invalid — those fills would be at worse
   prices and against the now-confirmed move.

Special case: **only one layer filled.** If exactly one layer is
FILLED at the trigger, we BE that layer (modify SL to its entry) but
do NOT close it. A close would exit the trade entirely; BE keeps it
running toward TP1 with downside protection. The user explicitly
requested this branching when designing the PR.

Idempotency
-----------
We detect "already done" from live trade state instead of persisting
a flag (no new column on ``setups`` needed):

* If the shallowest FILLED layer's ``sl_price`` already equals its
  ``entry_price`` (within 0.01 tolerance), the BE move has already
  been applied → skip the whole trigger.
* If every layer is already terminal (CLOSED / CANCELLED), nothing to
  do → skip.

On bot restart this re-evaluates cleanly: the BE move is a no-op
modify_order (broker accepts setting the SL to its current value);
the close on the shallowest layer would be a no-op too because it's
already terminal in the DB.

Relationship to TP cascade (``tp_manager``)
-------------------------------------------
The TP cascade (PR #41 / #43) and the zone-exit trigger are
independent and compose cleanly:

* Zone-exit fires first (it's M5-close-cadenced and fires earlier in
  most setups' lifetimes than any TP).
* After zone-exit, remaining FILLED layers have SL at their own
  entry (BE for that layer).
* If TP1 fires later on (say) layer 2, ``tp_manager`` cascades a
  fresh SL onto layer 3 = layer 2's entry. Layer 3's SL moves from
  its own entry (BE) to layer 2's entry (deeper into profit) —
  always a tighter SL, never looser, so no double-protection issue.

State written
-------------
* The closed layer's trade row: ``status: FILLED → CLOSED``,
  ``close_reason='ZONE_EXIT'``, ``exit_price=bid/ask``.
* Each remaining FILLED layer: broker ``modify_order(ticket, sl=entry)``
  and the row's ``sl_price`` patched to the layer's own entry price.
* Each remaining WAITING layer: ``status: WAITING → CANCELLED``,
  ``close_reason='ZONE_EXIT_CANCELLED'``.
* Setup status: nothing here. The completion hook in
  ``position_tracker.update_trade_status`` flips it to ``CLOSED``
  when the last trade row goes terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from uuid import UUID

from loguru import logger

from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import PositionTracker
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Result + config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ZoneExitResult:
    """One zone-exit trigger firing on a setup."""

    setup_id: UUID
    close_price: float
    closed_trade_id: UUID | None
    """The shallowest FILLED layer's trade row id, if a close was
    actually performed. ``None`` if only-one-layer-filled (BE-only
    branch) or no FILLED layers at all."""
    closed_layer: int | None
    be_layer_count: int
    """Number of remaining FILLED layers that had SL modified to
    entry (== break-even for that layer)."""
    cancelled_waiting_count: int
    """Number of WAITING layers cancelled."""
    error: str | None = None


@dataclass(frozen=True)
class ZoneExitConfig:
    symbol: str = "XAUUSD"
    """Reserved for future use (broker routing). Not read today."""

    be_tolerance_points: float = 0.01
    """Tolerance for the "is SL already at entry?" idempotency check.
    XAUUSD typically prices to 2 dp ($0.01); broker fills can vary by
    a small fraction. 0.01 catches "already done" without false
    positives from real cascaded-SL moves."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class ZoneExitManager:
    """Body-close-out-of-zone detection and BE orchestration."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        position_tracker: PositionTracker,
        config: ZoneExitConfig | None = None,
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        self._tracker = position_tracker
        self._config = config or ZoneExitConfig()

    def check(
        self, setup: Setup, last_close: float, bid: float, ask: float,
    ) -> ZoneExitResult | None:
        """Evaluate one setup against the last M5 bar's close.

        Returns ``None`` if the trigger doesn't fire (either the
        body-close condition isn't met OR idempotency says already
        done). Returns a :class:`ZoneExitResult` on a real firing.
        """
        if setup.status != "ACTIVE":
            return None

        # 1. Body-close trigger. L1 == the zone edge price entered from.
        l1 = float(setup.planned_layer1_price)
        if setup.direction == "BUY":
            if last_close <= l1:
                return None
        else:  # SELL
            if last_close >= l1:
                return None

        # 2. Load trades, sort by layer number ascending (1, 2, 3).
        try:
            trades = self._supabase.get_trades_for_setup(setup.id)
        except Exception:
            logger.exception(
                f"zone_exit_manager: get_trades_for_setup failed for {setup.id}"
            )
            return None
        trades = sorted(trades, key=lambda t: t.layer_number)

        filled = [t for t in trades if t.status == "FILLED"]
        waiting = [t for t in trades if t.status == "WAITING"]

        if not filled and not waiting:
            return None  # nothing to do — every layer terminal

        # 3. Idempotency: if the shallowest FILLED layer already has
        # SL at its own entry price, the BE move was already applied.
        if filled and self._is_be_already_done(filled[0]):
            return None

        # 4. Decide branch: ≥2 filled → close shallowest + BE rest.
        # Exactly 1 filled → BE only (don't close, keep trade alive).
        close_price = bid if setup.direction == "BUY" else ask
        closed_trade_id: UUID | None = None
        closed_layer: int | None = None
        be_layer_count = 0
        errors: list[str] = []

        if len(filled) >= 2:
            shallowest = filled[0]
            err = self._close_layer(shallowest, close_price)
            if err is not None:
                errors.append(err)
            else:
                closed_trade_id = shallowest.id
                closed_layer = shallowest.layer_number
            remaining_filled = filled[1:]
        else:
            remaining_filled = filled

        # 5. Move SL to entry on each remaining FILLED layer.
        for trade in remaining_filled:
            err = self._move_to_be(trade)
            if err is not None:
                errors.append(err)
            else:
                be_layer_count += 1

        # 6. Cancel WAITING layers.
        cancelled_waiting_count = 0
        for trade in waiting:
            err = self._cancel_waiting(trade)
            if err is not None:
                errors.append(err)
            else:
                cancelled_waiting_count += 1

        return ZoneExitResult(
            setup_id=setup.id,
            close_price=close_price,
            closed_trade_id=closed_trade_id,
            closed_layer=closed_layer,
            be_layer_count=be_layer_count,
            cancelled_waiting_count=cancelled_waiting_count,
            error="; ".join(errors) if errors else None,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _is_be_already_done(self, shallowest_filled: Trade) -> bool:
        """True iff the shallowest FILLED layer's SL is at its own entry.

        We use the shallowest because that's the layer most likely to
        have been BE'd already (multi-layer cases will all have moved
        together; single-layer is just this one).
        """
        if shallowest_filled.entry_price is None:
            # Shouldn't happen for FILLED, but be defensive.
            return False
        sl = float(shallowest_filled.sl_price)
        entry = float(shallowest_filled.entry_price)
        return abs(sl - entry) <= self._config.be_tolerance_points

    def _close_layer(self, trade: Trade, close_price: float) -> str | None:
        layer_num = trade.layer_number
        if trade.mt5_ticket is None:
            msg = (
                f"layer {layer_num} of setup {trade.setup_id} is FILLED "
                f"but has no mt5_ticket; cannot close"
            )
            logger.error(f"zone_exit_manager: {msg}")
            return msg
        try:
            self._mt5.close_position(trade.mt5_ticket)
        except Exception as e:
            msg = (
                f"close_position failed for shallowest layer {layer_num} "
                f"ticket={trade.mt5_ticket}: {e}"
            )
            logger.exception(f"zone_exit_manager: {msg}")
            self._safe_log_event(
                "ERROR",
                f"ZONE_EXIT close failed on layer {layer_num}",
                context={
                    "setup_id": str(trade.setup_id),
                    "trade_id": str(trade.id),
                    "ticket": trade.mt5_ticket,
                    "close_price": close_price,
                    "exception": str(e),
                },
                setup_id=trade.setup_id, trade_id=trade.id,
            )
            return msg
        try:
            self._tracker.update_trade_status(
                trade.id,
                "CLOSED",
                close_reason="ZONE_EXIT",
                exit_price=close_price,
            )
        except Exception as e:
            msg = (
                f"trade row update failed after ZONE_EXIT close on "
                f"layer {layer_num} (broker position is closed; "
                f"bookkeeping desync): {e}"
            )
            logger.exception(f"zone_exit_manager: {msg}")
            return msg
        return None

    def _move_to_be(self, trade: Trade) -> str | None:
        """Modify the layer's SL to its own entry price."""
        layer_num = trade.layer_number
        if trade.entry_price is None:
            msg = (
                f"FILLED layer {layer_num} has no entry_price "
                f"(unexpected); cannot move to BE"
            )
            logger.error(f"zone_exit_manager: {msg}")
            return msg
        if trade.mt5_ticket is None:
            msg = (
                f"FILLED layer {layer_num} has no mt5_ticket "
                f"(unexpected); cannot modify SL"
            )
            logger.error(f"zone_exit_manager: {msg}")
            return msg
        new_sl = float(trade.entry_price)
        try:
            self._mt5.modify_order(trade.mt5_ticket, sl=new_sl)
        except Exception as e:
            msg = (
                f"modify_order to BE failed on layer {layer_num} "
                f"ticket={trade.mt5_ticket}: {e}"
            )
            logger.exception(f"zone_exit_manager: {msg}")
            self._safe_log_event(
                "ERROR",
                f"ZONE_EXIT BE move failed on layer {layer_num}",
                context={
                    "setup_id": str(trade.setup_id),
                    "trade_id": str(trade.id),
                    "ticket": trade.mt5_ticket,
                    "attempted_sl": new_sl,
                    "exception": str(e),
                },
                setup_id=trade.setup_id, trade_id=trade.id,
            )
            return msg
        try:
            self._supabase.update_trade(
                trade.id, sl_price=Decimal(str(new_sl)),
            )
        except Exception as e:
            msg = (
                f"sl_price row update failed for layer {layer_num} "
                f"after BE modify_order success: {e}"
            )
            logger.exception(f"zone_exit_manager: {msg}")
            return msg
        return None

    def _cancel_waiting(self, trade: Trade) -> str | None:
        """Cancel a WAITING layer after zone-exit confirmation."""
        try:
            self._tracker.update_trade_status(
                trade.id,
                "CANCELLED",
                close_reason="ZONE_EXIT_CANCELLED",
            )
        except Exception as e:
            msg = (
                f"WAITING layer {trade.layer_number} cancellation failed "
                f"after zone-exit confirmation: {e}"
            )
            logger.exception(f"zone_exit_manager: {msg}")
            return msg
        return None

    def _safe_log_event(self, *args, **kwargs) -> None:
        try:
            self._supabase.log_event(*args, **kwargs)
        except Exception:
            logger.exception(
                "zone_exit_manager: log_event itself failed (non-fatal)"
            )
