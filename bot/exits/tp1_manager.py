"""TP1 partial-take + break-even move (spec Section 6.1).

When a setup's planned TP1 is touched, this module:

1. Closes 50 % of each filled layer, **rounded down** to the broker's
   minimum lot step. When the rounded amount falls below the lot step
   (e.g. 50 % of 0.01 lots = 0.005 < 0.01 step), the partial close is
   **skipped** for that layer — the runner stays open and SL still moves
   to break-even. This is Option B from the v1 lot-rounding decision:
   in v1 with fixed 0.01 lots every TP1 takes this path. v1.1 risk-based
   sizing will produce larger lots where the 50 % rounds to a non-zero
   value and partial closes execute normally.

2. Moves the SL on the remaining position to break-even — the
   **lot-weighted average** of filled entry prices (forward-compatible
   with v1.1 where layers have different sizes). modify_order failures
   here are flagged ``sl_modify_pending=True`` for the main loop to
   retry; the original structural SL is still active so the position is
   not uncovered, and aborting the runner over a transient network blip
   would defeat the strategy. Setup still transitions to TP1_HIT — the
   partial close (if any) succeeded.

3. Cascade-cancels any still-WAITING Layer 2 / 3 trades. Implemented in
   :class:`PositionTracker.update_setup_status` (see CASCADE_CANCEL_…)
   so any path transitioning a setup to TP1_HIT gets the same behaviour.

TP1 trigger price source
------------------------
Reads :attr:`Setup.planned_tp1_price`, set at setup creation by
``order_manager`` from zone bounds + the ``bot_config.tp1_distance_dollars``
value at that moment. This is a deliberate snapshot: re-fetching the
config value per tick would let mid-trade dashboard edits shift the
trigger on in-flight setups, which is almost never what the operator
wants. Same pattern as ``planned_sl_price`` and ``planned_layer*_price``.

State transitions written
-------------------------
* Per layer that DID have a partial close:
  ``trades.status: FILLED → PARTIALLY_CLOSED``, ``close_reason="TP1"``,
  ``sl_price → BE``.
* Per layer that did NOT (lot-rounding skip): ``status`` stays FILLED,
  ``sl_price → BE``.
* Setup: ``status: ACTIVE → TP1_HIT`` (always, when the trigger fires
  and there is at least one filled layer; the cascade in
  ``position_tracker`` then flips any WAITING trades to CANCELLED).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from loguru import logger

from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import PositionTracker
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade


# --------------------------------------------------------------------------- #
# Result + config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TP1Result:
    """Outcome of a TP1 check."""

    triggered: bool
    tp1_price: float = 0.0
    closed_lots: float = 0.0
    new_sl_price: float = 0.0
    error: str | None = None
    sl_modify_pending: bool = False
    """True iff partial close succeeded (or was skipped) but BE SL move
    failed. Setup is in TP1_HIT, original SL still active, main loop
    should retry the SL modify next iteration."""


@dataclass(frozen=True)
class TP1ManagerConfig:
    symbol: str = "XAUUSD"
    lot_step: float = 0.01
    """Broker's minimum volume increment for the symbol. Vantage XAUUSD = 0.01."""
    comment_prefix: str = "bot:tp1"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class TP1Manager:
    """Detects TP1 hits and runs the partial-close + BE move sequence."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        position_tracker: PositionTracker,
        config: TP1ManagerConfig | None = None,
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        self._tracker = position_tracker
        self._config = config or TP1ManagerConfig()

    def check(self, setup: Setup, bid: float, ask: float) -> TP1Result:
        """Check whether ``setup`` has hit TP1 and act accordingly."""
        tp1_price = float(setup.planned_tp1_price)

        # Idempotent: already-TP1_HIT setup should never re-fire.
        if setup.status == "TP1_HIT":
            return TP1Result(triggered=False, tp1_price=tp1_price)

        # Defence: only ACTIVE setups can transition to TP1_HIT.
        if setup.status != "ACTIVE":
            return TP1Result(triggered=False, tp1_price=tp1_price)

        if not _trigger_met(setup, tp1_price, bid, ask):
            return TP1Result(triggered=False, tp1_price=tp1_price)

        trades = self._supabase.get_trades_for_setup(setup.id)
        filled = [t for t in trades if t.status == "FILLED"]
        if not filled:
            msg = f"TP1 trigger met for setup {setup.id} but no FILLED trades"
            logger.error(msg)
            self._supabase.log_event(
                "ERROR", msg,
                context={"setup_id": str(setup.id)},
                setup_id=setup.id,
            )
            return TP1Result(
                triggered=False,
                tp1_price=tp1_price,
                error="no_filled_trades",
            )

        be_price = _compute_be_lot_weighted(filled)
        close_price = bid if setup.direction == "BUY" else ask

        closed_lots, close_errors = self._partial_close_each(
            filled, close_price=close_price, be_price=be_price,
        )

        # Move SL on the remaining position to BE. Even when no partial
        # close ran (Option B path), we still want the runner protected.
        sl_modify_pending, sl_errors = self._move_sl_to_be(
            filled, be_price=be_price,
        )

        # Setup → TP1_HIT regardless of SL outcome. The trigger fired,
        # the partial-close stage finished; SL move is recoverable on
        # the next loop iteration.
        try:
            self._tracker.update_setup_status(setup.id, "TP1_HIT")
        except Exception as e:  # pragma: no cover — defensive
            logger.exception(
                f"tp1_manager: setup status TP1_HIT transition failed for {setup.id}"
            )
            return TP1Result(
                triggered=True,
                tp1_price=tp1_price,
                closed_lots=closed_lots,
                new_sl_price=be_price,
                error=f"setup_transition_failed: {e}",
                sl_modify_pending=sl_modify_pending,
            )

        all_errors = close_errors + sl_errors
        return TP1Result(
            triggered=True,
            tp1_price=tp1_price,
            closed_lots=closed_lots,
            new_sl_price=be_price,
            error="; ".join(all_errors) if all_errors else None,
            sl_modify_pending=sl_modify_pending,
        )

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    def _partial_close_each(
        self,
        filled: list[Trade],
        *,
        close_price: float,
        be_price: float,
    ) -> tuple[float, list[str]]:
        """Close 50% of each filled layer (rounded down to lot step).

        Returns ``(total_closed_lots, errors)``. Errors are per-layer;
        we log each and continue so one layer's failure doesn't block
        the others' partial closes.
        """
        total_closed = 0.0
        errors: list[str] = []
        for trade in filled:
            half = float(trade.lot_size) / 2.0
            partial = _round_down_to_step(half, self._config.lot_step)
            if partial <= 0.0:
                # Option B path — partial below lot step. Skip the close;
                # SL still moves to BE on the full layer.
                logger.info(
                    f"tp1_manager: layer {trade.layer_number} of setup "
                    f"{trade.setup_id} — 50% lot ({half}) below step "
                    f"({self._config.lot_step}); skipping partial close"
                )
                continue
            if trade.mt5_ticket is None:
                msg = (
                    f"layer {trade.layer_number} has no mt5_ticket; "
                    f"cannot partial-close"
                )
                logger.error(f"tp1_manager: {msg}")
                errors.append(msg)
                continue
            try:
                self._mt5.close_position(
                    trade.mt5_ticket, partial_lots=partial,
                )
            except Exception as e:
                msg = (
                    f"close_position failed for layer {trade.layer_number} "
                    f"ticket={trade.mt5_ticket}: {e}"
                )
                logger.exception(f"tp1_manager: {msg}")
                self._supabase.log_event(
                    "ERROR", msg,
                    context={
                        "setup_id": str(trade.setup_id),
                        "trade_id": str(trade.id),
                        "ticket": trade.mt5_ticket,
                        "partial_lots": partial,
                    },
                    setup_id=trade.setup_id,
                    trade_id=trade.id,
                )
                errors.append(msg)
                continue

            # Broker close succeeded — write trade row.
            try:
                self._tracker.update_trade_status(
                    trade.id,
                    "PARTIALLY_CLOSED",
                    close_reason="TP1",
                    exit_price=close_price,
                )
                # Reflect new SL on the remaining lots in Supabase too;
                # the broker move happens in _move_sl_to_be.
                self._supabase.update_trade(
                    trade.id, sl_price=Decimal(str(be_price)),
                )
            except Exception as e:
                # Bookkeeping failure after a successful broker close.
                # Reconciliation will eventually catch up.
                msg = (
                    f"trade row update failed after partial close "
                    f"layer={trade.layer_number}: {e}"
                )
                logger.exception(f"tp1_manager: {msg}")
                errors.append(msg)

            total_closed += partial

        return total_closed, errors

    def _move_sl_to_be(
        self,
        filled: list[Trade],
        *,
        be_price: float,
    ) -> tuple[bool, list[str]]:
        """Move SL → BE on every filled layer's remaining position.

        Returns ``(any_failure, errors)``. ``any_failure=True`` flags the
        result with ``sl_modify_pending`` so the main loop can retry on
        the next iteration.
        """
        any_failure = False
        errors: list[str] = []
        for trade in filled:
            if trade.mt5_ticket is None:
                msg = (
                    f"layer {trade.layer_number} has no mt5_ticket; "
                    f"cannot modify SL"
                )
                logger.error(f"tp1_manager: {msg}")
                errors.append(msg)
                any_failure = True
                continue
            try:
                self._mt5.modify_order(trade.mt5_ticket, sl=be_price)
            except Exception as e:
                msg = (
                    f"modify_order SL→BE failed for layer "
                    f"{trade.layer_number} ticket={trade.mt5_ticket}: {e}"
                )
                logger.exception(f"tp1_manager: CRITICAL {msg}")
                self._supabase.log_event(
                    "ERROR",
                    f"CRITICAL: SL→BE failed; original SL still active",
                    context={
                        "setup_id": str(trade.setup_id),
                        "trade_id": str(trade.id),
                        "ticket": trade.mt5_ticket,
                        "attempted_sl": be_price,
                        "exception": str(e),
                        "retry_recommended": True,
                    },
                    setup_id=trade.setup_id,
                    trade_id=trade.id,
                )
                errors.append(msg)
                any_failure = True
                continue

            # Broker SL move succeeded — sync Supabase. (For PARTIALLY_CLOSED
            # trades this is a no-op duplicate of the update in
            # _partial_close_each, but harmless and keeps both paths consistent
            # for trades that skipped the partial close.)
            try:
                self._supabase.update_trade(
                    trade.id, sl_price=Decimal(str(be_price)),
                )
            except Exception as e:
                msg = (
                    f"trade row sl_price update failed after SL move "
                    f"layer={trade.layer_number}: {e}"
                )
                logger.exception(f"tp1_manager: {msg}")
                errors.append(msg)

        return any_failure, errors


# --------------------------------------------------------------------------- #
# Module-level helpers (testable without a TP1Manager instance)
# --------------------------------------------------------------------------- #


def _trigger_met(
    setup: Setup, tp1_price: float, bid: float, ask: float
) -> bool:
    """BUY fires when bid >= TP1; SELL when ask <= TP1. Inclusive boundary."""
    if setup.direction == "BUY":
        return bid >= tp1_price
    return ask <= tp1_price


def _round_down_to_step(value: float, step: float) -> float:
    """Floor ``value`` to the nearest multiple of ``step``.

    Uses Decimal to avoid float drift (0.01 + 0.01 != 0.02 in float).
    Returns 0.0 when ``value < step``.
    """
    if step <= 0:
        raise ValueError(f"lot step must be positive, got {step}")
    d_value = Decimal(str(value))
    d_step = Decimal(str(step))
    n = int(d_value / d_step)  # truncates toward zero, == floor for positives
    return float(n * d_step)


def _compute_be_lot_weighted(filled: Iterable[Trade]) -> float:
    """Lot-weighted average entry: ``Σ(entry_i × lot_i) / Σ(lot_i)``.

    Equal-weight is the same number when all layers share a lot size
    (v1's fixed 0.01 case), but the lot-weighted form stays correct
    when v1.1 risk-based sizing produces unequal layers.

    Trades with no ``entry_price`` are skipped — should not occur for
    FILLED trades but defensive.
    """
    weighted_sum = Decimal("0")
    total_lots = Decimal("0")
    for t in filled:
        if t.entry_price is None:
            continue
        weighted_sum += Decimal(t.entry_price) * Decimal(t.lot_size)
        total_lots += Decimal(t.lot_size)
    if total_lots == 0:
        raise ValueError("cannot compute BE: no filled trades with entry_price")
    return float(weighted_sum / total_lots)
