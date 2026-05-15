"""Trailing-stop for Layer 1 — locks in progressive profit.

Companion to PR #47's zone-exit BE protection. Zone-exit moves the SL
to entry once the M5 body closes out of the zone, which prevents a
loss but lets ALL intermediate profit evaporate if price reverses
back to BE. This manager locks in a portion of that intermediate
profit as the trade runs.

Scope
-----
**Only when Layer 1 is the ONLY filled layer.** Multi-layer cases
(L1+L2 or L1+L2+L3 FILLED) are already handled by the zone-exit
cascade (PR #47) and the TP cascade (PR #41), both of which produce
tighter SL moves than this trailing logic. Running both on a
multi-layer setup would either be a no-op (existing SLs already
better) or risk a regression, so we just skip.

If layer 1 is terminal (CLOSED via TP1, SL_HIT, or ZONE_EXIT) and L2
or L3 is still FILLED, we also skip — those layers are riding on the
TP1 cascade's SL, which is already locked at L1's entry.

Algorithm (BUY example)
-----------------------
* ``distance_to_tp1 = planned_tp1_price - L1.entry_price``
* ``current_profit  = bid - L1.entry_price``
* ``activation_threshold = activation_pct_of_tp1 × distance_to_tp1``
* If ``current_profit < activation_threshold`` → no action.
* ``trail_distance = trail_pct_of_profit × current_profit``
* ``new_sl = bid - trail_distance``
* Only modify if ``new_sl > current sl_price`` (locks more profit).

SELL is symmetric: distance and profit measured downward, new SL sits
above the current ask by ``trail_distance``, "better" means smaller.

Interaction with other SL movers
--------------------------------
* The trailing SL is monotonic — we only ever tighten (move closer to
  current price). If profit retraces, the SL stays where it was put
  on the previous M5 close. No "untrailing".
* If zone-exit BE fired first on this setup (SL already at L1 entry),
  trailing only kicks in once profit crosses 30 % of TP1 distance —
  at which point the new trailing SL is strictly above L1's entry
  (BUY) or strictly below (SELL), so it's an improvement.
* If TP1 cascade fires (layer 1 closes via TP1), there's no more L1
  to trail — and the "only L1 FILLED" gate skips the trail. The
  cascade's SL on L2/L3 is the canonical protection from that point.

Cadence
-------
Runs on every M5 close by default — same cadence as zone-exit. The
algorithm is pure arithmetic on cached state; broker load comes from
the ``modify_order`` call, which we only emit when the SL would
actually tighten. ``check_on_m5_close`` is provided so a future
caller can opt into per-tick trailing without re-wiring this module.

State written
-------------
* The trailed trade row: ``trades.sl_price`` patched. ``status``
  stays FILLED; no ``close_reason``.
* Broker: ``modify_order(ticket, sl=new_sl)``. The existing TP is
  preserved (the connector defaults the ``tp`` argument to the live
  position's TP when not passed).
* Setup status: untouched. The setup remains ACTIVE.

Failure modes
-------------
* ``modify_order`` raises → log + event + return the error in the
  result. No retry here; the next M5 close re-evaluates and will
  retry naturally if the conditions still hold.
* ``update_trade`` raises after a successful broker modify → log
  the desync but keep the broker move (it's the safer of the two).
  Next M5 close will read the stale ``sl_price`` and treat the
  current trail as if it didn't happen — the comparison will likely
  produce the same ``new_sl`` and re-attempt the patch.
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
class TrailingStopResult:
    """One trailing-stop SL move on a setup's Layer 1."""

    setup_id: UUID
    trade_id: UUID
    old_sl: float
    new_sl: float
    current_price: float
    """The bid (BUY) or ask (SELL) used to compute the new SL."""
    current_profit: float
    """Signed distance from entry to current price (always positive
    when we fire — we don't trail at a loss)."""
    error: str | None = None


@dataclass(frozen=True)
class TrailingStopConfig:
    activation_pct_of_tp1: float = 0.30
    """Trailing activates once current profit reaches this fraction
    of the distance from entry to planned TP1. 0.30 = 30 %."""

    trail_pct_of_profit: float = 0.50
    """New SL sits this fraction of current profit behind the current
    price. 0.50 = lock in half, give back half on reversal."""

    only_when_l1_only: bool = True
    """If True (default), trail only when exactly one layer is FILLED
    and it's layer 1. Multi-layer cases are owned by zone-exit /
    TP-cascade SL moves and would conflict with trailing."""

    check_on_m5_close: bool = True
    """Reserved for future per-tick mode. The manager itself is
    stateless — the caller chooses the cadence by deciding when to
    invoke :meth:`TrailingStopManager.check`. Logged here as the
    operator-visible default so per-tick activation is an explicit
    config flip rather than a silent caller change."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class TrailingStopManager:
    """Layer-1 trailing-stop orchestration."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        position_tracker: PositionTracker,
        config: TrailingStopConfig | None = None,
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        self._tracker = position_tracker
        self._config = config or TrailingStopConfig()

    def check(
        self, setup: Setup, bid: float, ask: float,
    ) -> TrailingStopResult | None:
        """Evaluate one setup for a trailing-stop SL move.

        Returns ``None`` if any of the gates short-circuit (setup not
        active, scope mismatch, below activation threshold, new SL
        not better than current). Returns a :class:`TrailingStopResult`
        when an SL move is attempted (success or broker failure).
        """
        if setup.status != "ACTIVE":
            return None

        try:
            trades = self._supabase.get_trades_for_setup(setup.id)
        except Exception:
            logger.exception(
                f"trailing_stop_manager: get_trades_for_setup failed "
                f"for {setup.id}"
            )
            return None

        l1 = self._select_l1_only_filled(trades)
        if l1 is None:
            return None

        if l1.entry_price is None or l1.mt5_ticket is None:
            # FILLED row missing the prerequisites for trailing.
            # Defensive — shouldn't happen for a real broker fill.
            return None

        entry = float(l1.entry_price)
        tp1 = float(setup.planned_tp1_price)
        old_sl = float(l1.sl_price)

        new_sl = _compute_trailed_sl(
            direction=setup.direction,
            entry=entry,
            tp1=tp1,
            bid=bid,
            ask=ask,
            old_sl=old_sl,
            activation_pct=self._config.activation_pct_of_tp1,
            trail_pct=self._config.trail_pct_of_profit,
        )
        if new_sl is None:
            return None

        current_price = bid if setup.direction == "BUY" else ask
        current_profit = (
            current_price - entry if setup.direction == "BUY"
            else entry - current_price
        )

        err = self._apply_trail(l1, new_sl)
        return TrailingStopResult(
            setup_id=setup.id,
            trade_id=l1.id,
            old_sl=old_sl,
            new_sl=new_sl,
            current_price=current_price,
            current_profit=current_profit,
            error=err,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _select_l1_only_filled(self, trades: list[Trade]) -> Trade | None:
        """Return the L1 trade iff exactly L1 is FILLED.

        ``WAITING`` layers are tolerated (L2/L3 not yet hit). ``CLOSED``
        or ``CANCELLED`` layers are also tolerated as long as L1 is
        the only FILLED row. Returns ``None`` otherwise.
        """
        filled = [t for t in trades if t.status == "FILLED"]
        if len(filled) != 1:
            return None
        only_filled = filled[0]
        if self._config.only_when_l1_only and only_filled.layer_number != 1:
            return None
        return only_filled

    def _apply_trail(self, trade: Trade, new_sl: float) -> str | None:
        layer_num = trade.layer_number
        try:
            self._mt5.modify_order(trade.mt5_ticket, sl=new_sl)
        except Exception as e:
            msg = (
                f"modify_order trailing failed on layer {layer_num} "
                f"ticket={trade.mt5_ticket}: {e}"
            )
            logger.exception(f"trailing_stop_manager: {msg}")
            self._safe_log_event(
                "ERROR",
                f"TRAILING_STOP modify failed on layer {layer_num}",
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
                f"after trailing modify_order success: {e}"
            )
            logger.exception(f"trailing_stop_manager: {msg}")
            return msg
        return None

    def _safe_log_event(self, *args, **kwargs) -> None:
        try:
            self._supabase.log_event(*args, **kwargs)
        except Exception:
            logger.exception(
                "trailing_stop_manager: log_event itself failed (non-fatal)"
            )


# --------------------------------------------------------------------------- #
# Module-level helpers (testable without a TrailingStopManager instance)
# --------------------------------------------------------------------------- #


def _compute_trailed_sl(
    *,
    direction: str,
    entry: float,
    tp1: float,
    bid: float,
    ask: float,
    old_sl: float,
    activation_pct: float,
    trail_pct: float,
) -> float | None:
    """Return the new SL to set, or ``None`` if no action.

    Returns ``None`` when:
    * Current profit is below the activation threshold.
    * The geometry between entry and TP1 is degenerate (TP1 not in
      the profit direction — should never happen for a valid setup
      but guarded defensively).
    * The new SL would not be better than the existing SL (no
      tighten = no broker call).
    * The new SL would sit on the wrong side of the current price
      (broker would reject it).
    """
    if direction == "BUY":
        distance_to_tp1 = tp1 - entry
        if distance_to_tp1 <= 0:
            return None
        current_profit = bid - entry
        if current_profit < activation_pct * distance_to_tp1:
            return None
        trail_distance = trail_pct * current_profit
        new_sl = bid - trail_distance
        # Only tighten (move closer to current price = higher SL).
        if new_sl <= old_sl:
            return None
        # Broker rule: SL for an open BUY must sit below the bid.
        if new_sl >= bid:
            return None
        return new_sl

    if direction == "SELL":
        distance_to_tp1 = entry - tp1
        if distance_to_tp1 <= 0:
            return None
        current_profit = entry - ask
        if current_profit < activation_pct * distance_to_tp1:
            return None
        trail_distance = trail_pct * current_profit
        new_sl = ask + trail_distance
        # Only tighten (move closer to current price = lower SL).
        if new_sl >= old_sl:
            return None
        # Broker rule: SL for an open SELL must sit above the ask.
        if new_sl <= ask:
            return None
        return new_sl

    return None
