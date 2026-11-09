"""75/25 runner with a trailing stop (Nov 2026).

Backtested on 12 months of M5 (current bot, £100 risk): +£7,446 vs
+£4,976 taking everything at TP1, max drawdown -£568 vs -£651, no losing
month. Only the exit after TP1 changes; entries, SL, sizing and the M5
close exit are untouched.

How a setup is set up (:func:`bot.execution.order_manager.place_xautism_orders`
with ``runner_keep_frac``):

* Layer 1 (0.01): **no broker TP**.
* Layer 2: split into the main part (broker TP at TP1, ~75% of the whole
  position with layer 1) and the runner part, stored as **layer 3**
  (same stop price, no TP, ~25%).

What happens here, every tick, for each ACTIVE runner-mode setup:

1. **Before TP1**: nothing. The broker SL and the M5 close exit protect
   everything exactly as before.
2. **Price reaches TP1** (bid for a buy, ask for a sell, same as the
   broker TP):

   * Any layer 2 / runner stop still waiting is cancelled.
   * If layer 2 filled: layer 1 is closed at market (``close_reason=TP1``),
     the layer 2 main part closes itself at its broker TP, and the runner
     (layer 3) keeps going.
   * If layer 2 never filled (price tagged the edge and left): the 0.01
     layer 1 itself becomes the runner instead of closing for a tiny win.
   * The runner's broker SL moves to its entry (breakeven).
3. **After TP1**: the runner's broker SL trails ``runner_trail_points``
   behind the best price since TP1, never worse than breakeven, and only
   moves when it improves by ``runner_trail_step``. The broker closes it
   when price comes back to the stop; :meth:`PositionTracker.detect_closed_positions`
   then closes the row and the setup as usual.

Restart-safe: the trail lives in the broker SL. A runner is recognised
after a restart by its SL sitting at or beyond its entry, and step 2 is
idempotent if it ever runs twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from loguru import logger

from bot.execution.layer2_stop import XautismEntryConfig, cancel_layer2_at_broker
from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import OPEN_TRADE_STATUSES, PositionTracker
from bot.logging.supabase_logger import Setup, Trade


@dataclass(frozen=True)
class RunnerEvent:
    setup_id: UUID
    action: str  # STARTED / TRAILED
    detail: str = ""


@dataclass
class _Runner:
    best: float


def _beyond_tp(direction: str, bid: float, ask: float, tp: float) -> bool:
    return bid >= tp if direction == "BUY" else ask <= tp


def _sl_locked(direction: str, sl: float, entry: float) -> bool:
    """SL at breakeven or better (only true once the runner has started)."""
    if not sl:
        return False
    return sl >= entry - 1e-6 if direction == "BUY" else sl <= entry + 1e-6


class RunnerTrailManager:
    """Takes 75% at TP1 and trails the rest. See module docstring."""

    def __init__(
        self,
        mt5: MT5Connector,
        tracker: PositionTracker,
        config: XautismEntryConfig | None = None,
        *,
        symbol: str = "XAUUSD",
    ) -> None:
        self._mt5 = mt5
        self._tracker = tracker
        self._config = config or XautismEntryConfig()
        self._symbol = symbol
        self._runners: dict[UUID, _Runner] = {}

    def check_live(
        self, bid: float, ask: float,
        config: XautismEntryConfig | None = None,
    ) -> list[RunnerEvent]:
        if config is not None:
            self._config = config
        if not self._config.runner_enabled:
            return []
        events: list[RunnerEvent] = []
        positions: dict[int, dict[str, Any]] | None = None
        seen: set[UUID] = set()

        for setup in self._tracker.get_active_setups():
            if setup.status != "ACTIVE" or setup.planned_tp1_price is None:
                continue
            try:
                trades = self._tracker.get_trades_for_setup(setup.id)
            except Exception:
                logger.exception(f"runner: get_trades failed for {setup.id}")
                continue
            if positions is None:
                try:
                    positions = {
                        p["ticket"]: p
                        for p in self._mt5.get_open_positions(self._symbol)
                    }
                except Exception:
                    logger.exception("runner: get_open_positions failed")
                    return events
            open_trades = [
                t for t in trades
                if t.status in OPEN_TRADE_STATUSES and t.mt5_ticket in positions
            ]
            if not open_trades or not self._is_runner_mode(trades, positions):
                continue
            seen.add(setup.id)
            ev = self._handle(setup, trades, open_trades, positions, bid, ask)
            if ev is not None:
                events.append(ev)

        # Forget setups that are gone.
        for sid in list(self._runners):
            if sid not in seen:
                self._runners.pop(sid, None)
        return events

    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_runner_mode(trades: list[Trade], positions: dict[int, dict[str, Any]]) -> bool:
        """Placed with the runner split: a layer 3 row exists, or layer 1
        went in without a broker TP."""
        if any(t.layer_number == 3 for t in trades):
            return True
        l1 = next((t for t in trades if t.layer_number == 1), None)
        if l1 is None or l1.mt5_ticket not in positions:
            return False
        return not positions[l1.mt5_ticket].get("tp")

    def _handle(
        self,
        setup: Setup,
        trades: list[Trade],
        open_trades: list[Trade],
        positions: dict[int, dict[str, Any]],
        bid: float,
        ask: float,
    ) -> RunnerEvent | None:
        direction = setup.direction
        tp = float(setup.planned_tp1_price)  # type: ignore[arg-type]
        price = bid if direction == "BUY" else ask
        state = self._runners.get(setup.id)

        if state is None:
            # Restart recovery: a runner whose SL already sits at / past
            # its entry has started before.
            started = any(
                _sl_locked(direction, float(positions[t.mt5_ticket].get("sl") or 0.0),
                           float(positions[t.mt5_ticket]["price_open"]))
                for t in open_trades
            )
            if started:
                state = self._runners[setup.id] = _Runner(best=price)
            elif _beyond_tp(direction, bid, ask, tp):
                return self._start(setup, trades, open_trades, positions, price)
            else:
                return None

        # Trail.
        state.best = max(state.best, price) if direction == "BUY" else min(state.best, price)
        trail = self._config.runner_trail_points
        step = self._config.runner_trail_step
        moved: list[str] = []
        for t in open_trades:
            if t.layer_number == 2 and positions[t.mt5_ticket].get("tp"):
                continue  # main part of layer 2: its broker TP closes it
            pos = positions[t.mt5_ticket]
            entry = float(pos["price_open"])
            cur = float(pos.get("sl") or 0.0)
            if direction == "BUY":
                new = round(max(entry, state.best - trail), 2)
                better = (not cur) or new >= cur + step
            else:
                new = round(min(entry, state.best + trail), 2)
                better = (not cur) or new <= cur - step
            if not better:
                continue
            try:
                self._mt5.modify_order(t.mt5_ticket, sl=new)  # type: ignore[arg-type]
            except Exception:
                logger.exception(f"runner: SL move to {new} failed for {t.mt5_ticket}")
                continue
            moved.append(f"L{t.layer_number} {cur:.2f}→{new:.2f}")
        if moved:
            logger.info(f"runner trail: setup={str(setup.id)[:8]} best={state.best:.2f} " + ", ".join(moved))
            return RunnerEvent(setup.id, "TRAILED", ", ".join(moved))
        return None

    def _start(
        self,
        setup: Setup,
        trades: list[Trade],
        open_trades: list[Trade],
        positions: dict[int, dict[str, Any]],
        price: float,
    ) -> RunnerEvent:
        direction = setup.direction
        # 1. Waiting layer 2 / runner stops are no longer wanted.
        for t in trades:
            if t.status == "WAITING":
                cancel_layer2_at_broker(self._mt5, t)
                try:
                    self._tracker.update_trade_status(
                        t.id, "CANCELLED", close_reason="CASCADE_CANCELLED",
                    )
                except Exception:
                    logger.exception(f"runner: cancel row failed for {t.id}")

        l2_in = [t for t in open_trades if t.layer_number in (2, 3)]
        runners: list[Trade]
        if l2_in:
            # 2a. Layer 2 in: layer 1 is part of the 75% → close it now.
            for t in open_trades:
                if t.layer_number != 1:
                    continue
                try:
                    self._mt5.close_position(t.mt5_ticket)  # type: ignore[arg-type]
                    self._tracker.update_trade_status(
                        t.id, "CLOSED", close_reason="TP1", exit_price=price,
                    )
                except Exception:
                    logger.exception(f"runner: layer 1 close at TP1 failed ({t.mt5_ticket})")
            runners = [t for t in l2_in if t.layer_number == 3]
        else:
            # 2b. Only the 0.01 is in: it becomes the runner.
            runners = [t for t in open_trades if t.layer_number == 1]

        # 3. Runner SL to breakeven.
        for t in runners:
            entry = float(positions[t.mt5_ticket]["price_open"])
            try:
                self._mt5.modify_order(t.mt5_ticket, sl=round(entry, 2))  # type: ignore[arg-type]
            except Exception:
                logger.exception(f"runner: breakeven move failed for {t.mt5_ticket}")

        self._runners[setup.id] = _Runner(best=price)
        who = ", ".join(f"L{t.layer_number} {t.lot_size} lots" for t in runners) or "none"
        logger.info(
            f"runner STARTED: setup={str(setup.id)[:8]} {direction} TP1 "
            f"{float(setup.planned_tp1_price):.2f} reached @ {price:.2f}; "  # type: ignore[arg-type]
            f"{'layer 1 closed, ' if l2_in else 'layer 2 never filled, '}"
            f"trailing {who} {self._config.runner_trail_points} behind, SL at breakeven"
        )
        return RunnerEvent(setup.id, "STARTED", who)
