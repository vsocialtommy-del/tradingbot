"""Per-layer take-profit with cascading SL protection (PR #41).

Replaces the pre-PR-41 ``tp1_manager``, which closed 50 % of every
filled layer and moved all SLs to break-even. That model didn't work
at 0.01 lots — the broker rejects partial closes below the lot step,
and the fallback "move all SLs to BE" reduced every trade to
break-even-or-stop. Replaced with:

* Each layer has its own TP price (``setups.planned_tp{1,2,3}_price``).
* When layer N's TP price is crossed by the live tick, the layer's
  full position closes (no partial). ``close_reason='TP{N}'``.
* SL on every remaining FILLED layer is cascaded to the closed
  layer's entry price. ``modify_order`` per ticket. SL on every
  remaining WAITING layer is updated in the trade row
  (``trades.sl_price``) so that ``entry_trigger._fire_layer`` uses
  the cascaded value when it eventually fires.
* WAITING layers are **not** cancelled (Q-A decision: lets the
  cascaded SL do its job if price retraces).
* The setup itself stays ``ACTIVE`` while any trade is FILLED or
  WAITING. ``position_tracker._check_setup_complete`` (PR #41 hook)
  transitions it to ``CLOSED`` once every trade row is terminal.

TP recompute (Q-C decision: only when NULL)
-------------------------------------------
This module never recomputes TPs. When a layer closes and the next
TP slot is NULL on the setup row, the orchestrator
(``main._maybe_recompute_next_tp``) calls
``tp_target.find_nearest_local_peak`` against the current df and
patches the setup row. tp_manager just publishes
:class:`LayerCloseResult` for the caller to act on.

State written
-------------
* The closed layer's trade row: ``status: FILLED → CLOSED``,
  ``close_reason='TP{N}'``, ``exit_price=bid/ask``.
* Each remaining FILLED layer: broker ``modify_order(ticket, sl=…)``
  and the row's ``sl_price`` patched.
* Each remaining WAITING layer: ``sl_price`` patched only (no broker
  call — there's no ticket yet).
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
class LayerCloseResult:
    """One layer closure event (per-layer TP hit)."""

    setup_id: UUID
    trade_id: UUID
    layer_number: int
    tp_price: float
    close_price: float
    cascaded_sl: float | None
    """The new SL applied to remaining layers (= closed layer's
    entry_price). ``None`` only if cascade was skipped (no remaining
    open layers, or entry_price unavailable + no fallback)."""
    needs_next_tp_recompute: bool
    """``True`` iff the next layer's TP slot is NULL on the setup row
    and the caller should attempt recomputation against current bars.
    Always ``False`` after layer 3 closes."""
    error: str | None = None


@dataclass(frozen=True)
class TPManagerConfig:
    symbol: str = "XAUUSD"
    comment_prefix: str = "bot:tp"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class TPManager:
    """Per-layer TP detection and close orchestration."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        position_tracker: PositionTracker,
        config: TPManagerConfig | None = None,
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        self._tracker = position_tracker
        self._config = config or TPManagerConfig()

    def check(
        self, setup: Setup, bid: float, ask: float,
    ) -> list[LayerCloseResult]:
        """Check every FILLED layer's TP against the current tick.

        Returns the list of closures triggered this tick (possibly
        empty). Layers are processed in order 1 → 2 → 3; multiple
        layers can fire on the same tick (e.g. a spike past several
        levels), and each close cascades the SL on layers after it.
        """
        if setup.status != "ACTIVE":
            return []

        try:
            trades = self._supabase.get_trades_for_setup(setup.id)
        except Exception:
            logger.exception(
                f"tp_manager: get_trades_for_setup failed for {setup.id}"
            )
            return []

        by_num: dict[int, Trade] = {t.layer_number: t for t in trades}
        results: list[LayerCloseResult] = []

        for layer_num in (1, 2, 3):
            trade = by_num.get(layer_num)
            if trade is None or trade.status != "FILLED":
                continue
            tp_price = _layer_tp(setup, layer_num)
            if tp_price is None:
                # Layer has no TP (NULL on setup row). Rides on
                # cascaded SL until external close. Per-PR-41 Q-B.
                continue
            if not _trigger_met(setup.direction, tp_price, bid, ask):
                continue

            close_price = bid if setup.direction == "BUY" else ask
            result = self._close_layer(
                setup=setup,
                trade=trade,
                tp_price=tp_price,
                close_price=close_price,
                remaining_trades=[
                    t for t in trades
                    if t.layer_number > layer_num
                ],
            )
            results.append(result)

            # Mutate local view so a subsequent iteration sees the
            # now-terminal status (defensive — the loop top already
            # filters on status != "FILLED").
            by_num[layer_num] = trade.model_copy(
                update={"status": "CLOSED"},
            )

        return results

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _close_layer(
        self,
        *,
        setup: Setup,
        trade: Trade,
        tp_price: float,
        close_price: float,
        remaining_trades: list[Trade],
    ) -> LayerCloseResult:
        """Close one layer's ticket + cascade SL to layers after it."""
        layer_num = trade.layer_number
        errors: list[str] = []

        # 1. Close the broker position.
        if trade.mt5_ticket is None:
            msg = (
                f"layer {layer_num} of setup {setup.id} is FILLED "
                f"but has no mt5_ticket; cannot close"
            )
            logger.error(f"tp_manager: {msg}")
            return LayerCloseResult(
                setup_id=setup.id,
                trade_id=trade.id,
                layer_number=layer_num,
                tp_price=tp_price,
                close_price=close_price,
                cascaded_sl=None,
                needs_next_tp_recompute=False,
                error=msg,
            )
        try:
            self._mt5.close_position(trade.mt5_ticket)
        except Exception as e:
            msg = (
                f"close_position failed for layer {layer_num} "
                f"ticket={trade.mt5_ticket}: {e}"
            )
            logger.exception(f"tp_manager: {msg}")
            self._safe_log_event(
                "ERROR",
                f"TP{layer_num} close failed",
                context={
                    "setup_id": str(setup.id),
                    "trade_id": str(trade.id),
                    "ticket": trade.mt5_ticket,
                    "tp_price": tp_price,
                    "exception": str(e),
                },
                setup_id=setup.id, trade_id=trade.id,
            )
            return LayerCloseResult(
                setup_id=setup.id,
                trade_id=trade.id,
                layer_number=layer_num,
                tp_price=tp_price,
                close_price=close_price,
                cascaded_sl=None,
                needs_next_tp_recompute=False,
                error=msg,
            )

        # 2. Mark the trade CLOSED. The setup-completion hook in
        # ``position_tracker.update_trade_status`` fires here — when
        # this is the last filled layer, the setup transitions to
        # CLOSED automatically.
        close_reason = f"TP{layer_num}"
        try:
            self._tracker.update_trade_status(
                trade.id,
                "CLOSED",
                close_reason=close_reason,
                exit_price=close_price,
            )
        except Exception as e:
            msg = (
                f"trade row update failed after TP{layer_num} close "
                f"(broker position is closed; bookkeeping desync): {e}"
            )
            logger.exception(f"tp_manager: {msg}")
            errors.append(msg)

        # 3. Cascade SL to the closed layer's entry price on every
        # subsequent layer (FILLED → modify_order + row; WAITING →
        # row only).
        cascade_sl = _resolve_cascade_sl(trade, setup)
        if cascade_sl is None:
            errors.append(
                f"could not resolve cascade SL for layer {layer_num} "
                f"(entry_price NULL and no planned_layer{layer_num}_price)"
            )
        elif remaining_trades:
            errors.extend(
                self._cascade_sl_to_remaining(
                    remaining_trades, new_sl=cascade_sl,
                )
            )

        # 4. Hint the caller whether the next TP slot needs recompute.
        needs_recompute = False
        if layer_num < 3:
            next_slot = _layer_tp(setup, layer_num + 1)
            needs_recompute = next_slot is None

        return LayerCloseResult(
            setup_id=setup.id,
            trade_id=trade.id,
            layer_number=layer_num,
            tp_price=tp_price,
            close_price=close_price,
            cascaded_sl=cascade_sl,
            needs_next_tp_recompute=needs_recompute,
            error="; ".join(errors) if errors else None,
        )

    def _cascade_sl_to_remaining(
        self,
        remaining: list[Trade],
        *,
        new_sl: float,
    ) -> list[str]:
        """Apply ``new_sl`` to every remaining trade.

        * FILLED → broker ``modify_order(ticket, sl=new_sl)`` AND
          update ``trades.sl_price``.
        * WAITING → update ``trades.sl_price`` only.
        * Terminal (CLOSED/CANCELLED) → no-op.

        Per-trade failures are logged + appended to the error list
        but don't stop the cascade.
        """
        errors: list[str] = []
        for trade in remaining:
            if trade.status == "FILLED":
                err = self._cascade_filled(trade, new_sl)
                if err is not None:
                    errors.append(err)
            elif trade.status == "WAITING":
                err = self._cascade_waiting(trade, new_sl)
                if err is not None:
                    errors.append(err)
        return errors

    def _cascade_filled(self, trade: Trade, new_sl: float) -> str | None:
        if trade.mt5_ticket is None:
            return (
                f"FILLED layer {trade.layer_number} has no mt5_ticket "
                f"(unexpected); cannot modify SL"
            )
        try:
            self._mt5.modify_order(trade.mt5_ticket, sl=new_sl)
        except Exception as e:
            msg = (
                f"modify_order SL cascade failed for layer "
                f"{trade.layer_number} ticket={trade.mt5_ticket}: {e}"
            )
            logger.exception(f"tp_manager: {msg}")
            self._safe_log_event(
                "ERROR",
                f"SL cascade failed on layer {trade.layer_number}",
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
                f"sl_price row update failed for layer "
                f"{trade.layer_number} after modify_order success: {e}"
            )
            logger.exception(f"tp_manager: {msg}")
            return msg
        return None

    def _cascade_waiting(self, trade: Trade, new_sl: float) -> str | None:
        try:
            self._supabase.update_trade(
                trade.id, sl_price=Decimal(str(new_sl)),
            )
        except Exception as e:
            msg = (
                f"sl_price row update failed for WAITING layer "
                f"{trade.layer_number}: {e}"
            )
            logger.exception(f"tp_manager: {msg}")
            return msg
        return None

    def _safe_log_event(self, *args, **kwargs) -> None:
        try:
            self._supabase.log_event(*args, **kwargs)
        except Exception:
            logger.exception(
                "tp_manager: log_event itself failed (non-fatal)"
            )


# --------------------------------------------------------------------------- #
# Module-level helpers (testable without a TPManager instance)
# --------------------------------------------------------------------------- #


def _trigger_met(
    direction: str, tp_price: float, bid: float, ask: float,
) -> bool:
    """True iff price has reached the layer's TP.

    * BUY:  bid >= tp_price (we close BUY at the bid).
    * SELL: ask <= tp_price (we close SELL at the ask).
    Inclusive at the trigger price.
    """
    if direction == "BUY":
        return bid >= tp_price
    return ask <= tp_price


def _layer_tp(setup: Setup, layer_num: int) -> float | None:
    """Return the planned TP price for layer N, or None if NULL."""
    if layer_num == 1:
        return float(setup.planned_tp1_price)
    if layer_num == 2:
        return (
            float(setup.planned_tp2_price)
            if setup.planned_tp2_price is not None else None
        )
    if layer_num == 3:
        return (
            float(setup.planned_tp3_price)
            if setup.planned_tp3_price is not None else None
        )
    raise ValueError(f"unknown layer_number: {layer_num}")


def _resolve_cascade_sl(trade: Trade, setup: Setup) -> float | None:
    """The SL value to cascade to remaining layers.

    Primary: the closing layer's ``entry_price`` (where it actually
    filled on the broker). Fallback: the layer's planned price (set
    at setup creation). Returns ``None`` only if neither is
    available — defensive guard, shouldn't happen for a FILLED trade.
    """
    if trade.entry_price is not None:
        return float(trade.entry_price)
    planned = None
    if trade.layer_number == 1:
        planned = setup.planned_layer1_price
    elif trade.layer_number == 2:
        planned = setup.planned_layer2_price
    elif trade.layer_number == 3:
        planned = setup.planned_layer3_price
    if planned is not None:
        return float(planned)
    return None
