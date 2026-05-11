"""Position tracker — source of truth for the bot's current setups.

Reads from Supabase, reconciles with MT5, drives the state machine for
setups and trades.

State machines
--------------

Setup statuses (spec Section 9.2 schema; allowed transitions below)::

    PENDING ───► ACTIVE ───► TP1_HIT ───► CLOSED  (terminal)
       │           ├──► STOPPED_OUT  (terminal)
       │           └──► CLOSED       (terminal)
       └──► SKIPPED   (terminal)

Trade statuses::

    PENDING ───► FILLED ───► PARTIALLY_CLOSED ───► CLOSED  (terminal)
       │           └──► CLOSED            (terminal)
       └──► CANCELLED  (terminal)

Invalid transitions raise :class:`StateTransitionError`. Terminal
statuses have no valid next transitions (stay terminal).

Reconciliation
--------------
:meth:`reconcile_with_mt5` cross-checks the FILLED / PARTIALLY_CLOSED
trades in Supabase against MT5's open positions and reports:

* **ghost tickets** — open in MT5 but no Supabase trade. Logged with
  a warning; **not auto-managed** (we don't know what risk profile
  the operator put on it). Surface for manual investigation.
* **lost tickets** — Supabase says open but MT5 says closed. The bot
  was probably restarted or the operator closed manually. Trade is
  marked ``CLOSED`` with ``close_reason="MANUAL_CLOSE"`` (we don't
  know specifically — could have been SL, manual, or news).

Why not call ``reconcile_with_mt5`` automatically? It does writes
(updates lost trades to CLOSED). Auto-running on every loop iteration
is wasteful; on-demand from main.py at startup and on schedule is
cleaner. The ``main.py`` orchestrator is responsible for invoking it.

Design decisions called out in PR
---------------------------------

1. **Ghost positions: log + ignore.** Auto-managing unknown positions
   is risky — we'd be putting SLs / closing orders on something we
   don't understand. Better to surface to the operator.
2. **Lost positions: auto-mark CLOSED.** The position is gone from
   the broker; clinging to a ``FILLED`` row in Supabase serves no
   purpose. We can't determine the precise reason without querying
   broker history (a future improvement); we use ``MANUAL_CLOSE`` as
   a conservative default.
3. **State validation: explicit error type.** :class:`StateTransitionError`
   not :class:`AssertionError`. Invalid transitions are programming
   bugs OR genuine concurrent-modification races; both deserve to
   bubble up cleanly so callers can decide whether to log + retry or
   crash.
4. **MT5 query failures: degrade gracefully.** A network blip during
   ``get_open_positions()`` shouldn't blow up the entire tracker.
   We log and return empty results (or a no-op reconcile).

The ``EXTERNAL_CLOSE`` reason mentioned in early design isn't in the
schema's ``CloseReason`` Literal (migration 001). For v1 we use
``MANUAL_CLOSE`` for externally-closed positions; adding
``EXTERNAL_CLOSE`` is a follow-up schema change.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from loguru import logger

from bot.execution.mt5_connector import MT5Connector
from bot.logging.supabase_logger import (
    Setup,
    SetupStatus,
    SupabaseLogger,
    Trade,
    TradeStatus,
)


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #

VALID_SETUP_TRANSITIONS: dict[str, frozenset[str]] = {
    "PENDING": frozenset({"ACTIVE", "SKIPPED"}),
    "ACTIVE": frozenset({"TP1_HIT", "STOPPED_OUT", "CLOSED"}),
    "TP1_HIT": frozenset({"CLOSED"}),
    "CLOSED": frozenset(),       # terminal
    "SKIPPED": frozenset(),      # terminal
    "STOPPED_OUT": frozenset(),  # terminal
}

VALID_TRADE_TRANSITIONS: dict[str, frozenset[str]] = {
    "WAITING": frozenset({"FILLED", "CANCELLED"}),
    "FILLED": frozenset({"PARTIALLY_CLOSED", "CLOSED"}),
    "PARTIALLY_CLOSED": frozenset({"CLOSED"}),
    "CLOSED": frozenset(),     # terminal
    "CANCELLED": frozenset(),  # terminal
}

ACTIVE_SETUP_STATUSES: frozenset[str] = frozenset({"PENDING", "ACTIVE", "TP1_HIT"})
"""Statuses where the setup is consuming exposure capacity."""

TERMINAL_SETUP_STATUSES: frozenset[str] = frozenset(
    {"CLOSED", "SKIPPED", "STOPPED_OUT"}
)
"""Statuses where the setup's lifecycle is finished."""

CASCADE_CANCEL_SETUP_STATUSES: frozenset[str] = (
    TERMINAL_SETUP_STATUSES | frozenset({"TP1_HIT"})
)
"""Statuses whose entry into triggers cancellation of WAITING layer trades.

Terminal states obviously cancel pending layers (the setup is done). TP1_HIT
also cancels them: once profit is locked at TP1, scaling deeper into the
zone would just add risk in territory we've already booked. (Spec 6.1.)
"""

OPEN_TRADE_STATUSES: frozenset[str] = frozenset({"FILLED", "PARTIALLY_CLOSED"})
"""Statuses where the trade has a live position on the broker."""


class StateTransitionError(ValueError):
    """Raised when an invalid setup/trade state transition is attempted."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReconcileResult:
    """Output of :meth:`PositionTracker.reconcile_with_mt5`."""

    ghost_tickets: list[int]                 # MT5 has, Supabase doesn't
    lost_trade_ids: list[UUID]               # Supabase has, MT5 doesn't
    closed_externally_count: int             # how many lost trades we marked CLOSED
    matched_count: int                       # both agree


@dataclass(frozen=True)
class ClosedPosition:
    """Output of :meth:`PositionTracker.detect_closed_positions`."""

    trade_id: UUID
    mt5_ticket: int
    close_reason: str  # always "MANUAL_CLOSE" in v1 — see module docstring


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class PositionTracker:
    """Source of truth for setup + trade state. Talks to Supabase + MT5."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        *,
        active_setups_cache_ttl_seconds: float = 5.0,
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        # TTL cache for ``get_active_setups()``. The Bot's run-iteration
        # loop calls this 2-3× per tick (entry_trigger + per-active-setup
        # tp1_manager + safe wrappers). At 1 Hz that's ~3 Supabase
        # queries/sec uncached, which trips Supabase's HTTP/2 max-
        # requests-per-connection limit (~10K) every ~55 min. With a
        # 5-second TTL the rate drops to ~0.6/sec ≈ 10K/4.6 h — well
        # below the threshold while still picking up new setups within
        # one strategy-pipeline cycle. Mutators below invalidate the
        # cache explicitly so we never serve stale "this setup is
        # ACTIVE" data after we ourselves transitioned it.
        self._cache_ttl_seconds = active_setups_cache_ttl_seconds
        self._cached_active: list[Setup] | None = None
        self._cached_at: datetime | None = None

    # ---- queries -----------------------------------------------------------

    def get_active_setups(self, *, force_refresh: bool = False) -> list[Setup]:
        """Return setups in PENDING / ACTIVE / TP1_HIT — i.e. consuming exposure.

        TTL-cached. Pass ``force_refresh=True`` to bypass the cache
        when you need fresh data (e.g. after a reconcile that may
        have updated statuses externally).
        """
        if not force_refresh and self._cache_valid():
            # Return a copy so callers can't mutate the cache by
            # accident (e.g. `setups.append(...)` on the result).
            return list(self._cached_active or [])
        result = self._supabase.get_setups_by_status(list(ACTIVE_SETUP_STATUSES))
        self._cached_active = list(result)
        self._cached_at = datetime.now(tz=timezone.utc)
        return result

    def invalidate_active_setups_cache(self) -> None:
        """Force the next ``get_active_setups()`` call to re-query Supabase.

        Called by mutation methods on this class. External callers
        can use it after writes that go around the tracker
        (rare — discourage this).
        """
        self._cached_at = None

    def _cache_valid(self) -> bool:
        if self._cached_active is None or self._cached_at is None:
            return False
        age = (datetime.now(tz=timezone.utc) - self._cached_at).total_seconds()
        return age < self._cache_ttl_seconds

    def get_setup_by_id(self, setup_id: UUID | str) -> Setup | None:
        return self._supabase.get_setup_by_id(setup_id)

    def get_trades_for_setup(self, setup_id: UUID | str) -> list[Trade]:
        return self._supabase.get_trades_for_setup(setup_id)

    # ---- state transitions -------------------------------------------------

    def update_setup_status(
        self,
        setup_id: UUID | str,
        new_status: SetupStatus,
        *,
        skip_reason: str | None = None,
    ) -> Setup:
        """Transition a setup to ``new_status`` if the transition is valid."""
        current = self._supabase.get_setup_by_id(setup_id)
        if current is None:
            raise ValueError(f"setup {setup_id} not found")

        _validate_setup_transition(current.status, new_status)

        fields: dict[str, Any] = {"status": new_status}
        if skip_reason is not None:
            fields["skip_reason"] = skip_reason

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        if new_status == "ACTIVE" and current.activated_at is None:
            fields["activated_at"] = now_iso
        if new_status in ("CLOSED", "SKIPPED", "STOPPED_OUT"):
            fields["closed_at"] = now_iso

        updated = self._supabase.update_setup(setup_id, **fields)
        # Status changed → active-setups membership may have shifted.
        self.invalidate_active_setups_cache()

        # Cascade: terminating a setup or hitting TP1 cancels WAITING layers.
        # Terminal: ACTIVE → STOPPED_OUT shouldn't leave un-fired Layer 2/3
        #   trades hanging — they'd never trigger but would show as live.
        # TP1_HIT: profit locked, scaling deeper just adds risk in
        #   already-booked territory (spec 6.1).
        if new_status in CASCADE_CANCEL_SETUP_STATUSES:
            self._cascade_cancel_waiting(setup_id)

        logger.info(
            f"setup {setup_id} transitioned: {current.status} → {new_status}"
        )
        return updated

    def _cascade_cancel_waiting(self, setup_id: UUID | str) -> None:
        """Cancel all WAITING trades belonging to a now-terminal setup."""
        trades = self._supabase.get_trades_for_setup(setup_id)
        for trade in trades:
            if trade.status != "WAITING":
                continue
            try:
                self.update_trade_status(trade.id, "CANCELLED")
                logger.info(
                    f"cascade-cancel: trade {trade.id} (layer "
                    f"{trade.layer_number}) WAITING → CANCELLED"
                )
            except (StateTransitionError, ValueError):
                logger.exception(
                    f"cascade-cancel failed for trade {trade.id}"
                )

    def update_trade_status(
        self,
        trade_id: UUID | str,
        new_status: TradeStatus,
        *,
        close_reason: str | None = None,
        exit_price: float | None = None,
        entry_price: float | None = None,
        mt5_ticket: int | None = None,
        pnl: float | None = None,
    ) -> Trade:
        """Transition a trade row, validating the transition.

        Optional fields:
          ``close_reason`` / ``exit_price`` / ``pnl``: set on close.
          ``entry_price`` / ``mt5_ticket``: set when an entry-trigger
            fires a WAITING layer (WAITING → FILLED).
        """
        current = self._supabase.get_trade_by_id(trade_id)
        if current is None:
            raise ValueError(f"trade {trade_id} not found")

        _validate_trade_transition(current.status, new_status)

        fields: dict[str, Any] = {"status": new_status}
        if close_reason is not None:
            fields["close_reason"] = close_reason
        if exit_price is not None:
            fields["exit_price"] = exit_price
        if entry_price is not None:
            fields["entry_price"] = entry_price
        if mt5_ticket is not None:
            fields["mt5_ticket"] = mt5_ticket
        if pnl is not None:
            fields["pnl"] = pnl

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        # Both CLOSED and CANCELLED are terminal — stamp closed_at on
        # either transition for consistent analytics.
        if (
            new_status in ("CLOSED", "CANCELLED")
            and current.closed_at is None
        ):
            fields["closed_at"] = now_iso
        if new_status == "FILLED" and current.filled_at is None:
            fields["filled_at"] = now_iso

        updated = self._supabase.update_trade(trade_id, **fields)
        logger.info(
            f"trade {trade_id} transitioned: {current.status} → {new_status}"
        )
        return updated

    # ---- detection (Supabase ←→ MT5 sync) ----------------------------------

    # Note: ``detect_filled_layers`` from the original design has been
    # removed. It existed to monitor broker-pending limit orders for
    # Layers 2/3, but those are no longer placed at the broker (PR #15
    # strategy change). ``entry_trigger`` now directly transitions
    # WAITING → FILLED when it fires a market order.

    def detect_closed_positions(self, setup: Setup) -> list[ClosedPosition]:
        """For ACTIVE/TP1_HIT setups, find trades whose MT5 position is gone.

        Marks each as ``CLOSED`` with reason ``MANUAL_CLOSE``. Returns the
        list of positions that were closed.
        """
        if setup.status not in ACTIVE_SETUP_STATUSES:
            return []
        if setup.status == "PENDING":
            # PENDING setups have no filled positions yet — nothing to detect.
            return []

        trades = self._supabase.get_trades_for_setup(setup.id)
        positions = self._safe_get_open_positions()
        if positions is None:
            return []
        live_tickets = {p["ticket"] for p in positions}

        closed: list[ClosedPosition] = []
        for trade in trades:
            if trade.status not in OPEN_TRADE_STATUSES:
                continue
            if trade.mt5_ticket is None:
                continue
            if trade.mt5_ticket in live_tickets:
                continue
            # Position is gone from MT5 — close the trade row.
            try:
                self.update_trade_status(
                    trade.id,
                    "CLOSED",
                    close_reason="MANUAL_CLOSE",
                )
                closed.append(ClosedPosition(
                    trade_id=trade.id,
                    mt5_ticket=trade.mt5_ticket,
                    close_reason="MANUAL_CLOSE",
                ))
            except (StateTransitionError, ValueError):
                # Already CLOSED (concurrent update) — fine, skip.
                logger.debug(f"trade {trade.id} already closed; skipping")

        return closed

    # ---- reconciliation ----------------------------------------------------

    def reconcile_with_mt5(self) -> ReconcileResult:
        """Cross-check Supabase open trades against MT5's positions.

        - Ghost tickets (MT5-only): logged warning; not managed.
        - Lost tickets (Supabase-only): trade marked ``CLOSED``.

        Best-effort: an MT5 query failure returns an empty result
        rather than raising.
        """
        positions = self._safe_get_open_positions()
        if positions is None:
            return ReconcileResult([], [], 0, 0)

        # All open trades from active setups.
        active_setups = self._supabase.get_setups_by_status(
            list(ACTIVE_SETUP_STATUSES)
        )
        all_open_trades: list[Trade] = []
        for s in active_setups:
            for t in self._supabase.get_trades_for_setup(s.id):
                if t.status in OPEN_TRADE_STATUSES and t.mt5_ticket is not None:
                    all_open_trades.append(t)

        supabase_tickets: set[int] = {
            t.mt5_ticket for t in all_open_trades if t.mt5_ticket is not None
        }
        mt5_tickets: set[int] = {p["ticket"] for p in positions}

        ghost_tickets = sorted(mt5_tickets - supabase_tickets)
        for ticket in ghost_tickets:
            logger.warning(
                f"GHOST POSITION: MT5 ticket {ticket} has no Supabase trade. "
                f"Not auto-managed; investigate manually."
            )

        lost_tickets = supabase_tickets - mt5_tickets
        lost_trade_ids: list[UUID] = []
        closed_externally_count = 0
        for trade in all_open_trades:
            if trade.mt5_ticket in lost_tickets:
                lost_trade_ids.append(trade.id)
                try:
                    self.update_trade_status(
                        trade.id,
                        "CLOSED",
                        close_reason="MANUAL_CLOSE",
                    )
                    closed_externally_count += 1
                    logger.warning(
                        f"LOST POSITION: trade {trade.id} ticket "
                        f"{trade.mt5_ticket} closed externally; marked CLOSED."
                    )
                except Exception:
                    logger.exception(
                        f"failed to mark lost trade {trade.id} as closed"
                    )

        matched_count = len(supabase_tickets & mt5_tickets)

        return ReconcileResult(
            ghost_tickets=ghost_tickets,
            lost_trade_ids=lost_trade_ids,
            closed_externally_count=closed_externally_count,
            matched_count=matched_count,
        )

    # ---- internals ---------------------------------------------------------

    def _safe_get_open_positions(self) -> list[dict[str, Any]] | None:
        """Wrap mt5.get_open_positions in try/except. None on failure."""
        try:
            return self._mt5.get_open_positions()
        except Exception:
            logger.exception("position_tracker: get_open_positions failed")
            return None


# --------------------------------------------------------------------------- #
# Module-level helpers (so tests can validate transitions without a tracker)
# --------------------------------------------------------------------------- #


def _validate_setup_transition(current: str, new: str) -> None:
    valid = VALID_SETUP_TRANSITIONS.get(current, frozenset())
    if new not in valid:
        raise StateTransitionError(
            f"invalid setup transition: {current} → {new}. "
            f"valid next: {sorted(valid) if valid else 'none (terminal)'}"
        )


def _validate_trade_transition(current: str, new: str) -> None:
    valid = VALID_TRADE_TRANSITIONS.get(current, frozenset())
    if new not in valid:
        raise StateTransitionError(
            f"invalid trade transition: {current} → {new}. "
            f"valid next: {sorted(valid) if valid else 'none (terminal)'}"
        )
