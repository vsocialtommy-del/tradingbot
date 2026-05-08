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
    "PENDING": frozenset({"FILLED", "CANCELLED"}),
    "FILLED": frozenset({"PARTIALLY_CLOSED", "CLOSED"}),
    "PARTIALLY_CLOSED": frozenset({"CLOSED"}),
    "CLOSED": frozenset(),     # terminal
    "CANCELLED": frozenset(),  # terminal
}

ACTIVE_SETUP_STATUSES: frozenset[str] = frozenset({"PENDING", "ACTIVE", "TP1_HIT"})
"""Statuses where the setup is consuming exposure capacity."""

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
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase

    # ---- queries -----------------------------------------------------------

    def get_active_setups(self) -> list[Setup]:
        """Return setups in PENDING / ACTIVE / TP1_HIT — i.e. consuming exposure."""
        return self._supabase.get_setups_by_status(list(ACTIVE_SETUP_STATUSES))

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
        logger.info(
            f"setup {setup_id} transitioned: {current.status} → {new_status}"
        )
        return updated

    def update_trade_status(
        self,
        trade_id: UUID | str,
        new_status: TradeStatus,
        *,
        close_reason: str | None = None,
        exit_price: float | None = None,
        pnl: float | None = None,
    ) -> Trade:
        """Transition a trade row, validating the transition."""
        current = self._supabase.get_trade_by_id(trade_id)
        if current is None:
            raise ValueError(f"trade {trade_id} not found")

        _validate_trade_transition(current.status, new_status)

        fields: dict[str, Any] = {"status": new_status}
        if close_reason is not None:
            fields["close_reason"] = close_reason
        if exit_price is not None:
            fields["exit_price"] = exit_price
        if pnl is not None:
            fields["pnl"] = pnl

        now_iso = datetime.now(tz=timezone.utc).isoformat()
        if new_status in ("CLOSED",) and current.closed_at is None:
            fields["closed_at"] = now_iso
        if new_status == "FILLED" and current.filled_at is None:
            fields["filled_at"] = now_iso

        updated = self._supabase.update_trade(trade_id, **fields)
        logger.info(
            f"trade {trade_id} transitioned: {current.status} → {new_status}"
        )
        return updated

    # ---- detection (Supabase ←→ MT5 sync) ----------------------------------

    def detect_filled_layers(self, setup: Setup) -> list[int]:
        """For PENDING setups, check if any pending limit orders have filled.

        Updates each freshly-filled trade to ``FILLED`` and (if at least
        one filled) transitions the setup to ``ACTIVE``. Returns the
        list of layer numbers that were just newly filled.
        """
        if setup.status != "PENDING":
            return []

        trades = self._supabase.get_trades_for_setup(setup.id)
        positions = self._safe_get_open_positions()
        if positions is None:
            return []
        positions_by_ticket = {p["ticket"]: p for p in positions}

        newly_filled: list[int] = []
        for trade in trades:
            if trade.status != "PENDING":
                continue
            if trade.mt5_ticket is None:
                continue
            position = positions_by_ticket.get(trade.mt5_ticket)
            if position is None:
                continue
            # MT5 reports an open position with this ticket → fill happened.
            now_iso = datetime.now(tz=timezone.utc).isoformat()
            self._supabase.update_trade(
                trade.id,
                status="FILLED",
                entry_price=float(position.get("price_open", 0)),
                filled_at=now_iso,
            )
            newly_filled.append(trade.layer_number)

        if newly_filled and setup.status == "PENDING":
            self.update_setup_status(setup.id, "ACTIVE")

        return sorted(newly_filled)

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
