"""Real-time tick polling for Layer 1 entry detection.

Approach (and why it's polling, not streaming)
----------------------------------------------
The ``MetaTrader5`` Python library does **not** expose an event-driven
tick subscription — there's no ``on_tick`` callback. The supported
pattern is to call ``mt5.symbol_info_tick(symbol)`` in a tight loop and
detect changes by comparing ``time_msc`` (millisecond timestamp). That
matches what the spec calls "real-time tick monitoring" (Section 4.2):
not literally every tick from the broker, but every tick our terminal
records, polled at sub-second cadence.

Default poll interval: 100 ms. Gold ticks ~5-20 times per second
during active sessions, so 100 ms catches every distinct tick without
hammering the terminal. Faster polling is cheap CPU-wise but doesn't
buy more granularity.

Listener contract
-----------------
A listener is ``Callable[[tick, prev_tick], None]`` — receives the new
tick plus the previous one (or ``None`` for the first tick observed).
The previous tick is what the order manager uses for the gap-protection
check (spec Section 4.3): if Layer 1 fires when a single tick has
already crossed the entire zone, the setup is skipped.

Listeners run synchronously inside the polling loop. Anything slow or
fallible (Supabase writes, broker calls) belongs in a background task,
not in the listener body — otherwise we delay the next tick read.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from loguru import logger

from bot.execution.mt5_connector import MT5Connector

TickListener = Callable[[dict[str, Any], dict[str, Any] | None], None]


class TickHandler:
    """Polling tick stream for a single symbol."""

    def __init__(
        self,
        connector: MT5Connector,
        symbol: str,
        poll_interval_ms: int = 100,
    ) -> None:
        self._connector = connector
        self._symbol = symbol
        self._poll_interval_sec = poll_interval_ms / 1000.0
        self._listeners: list[TickListener] = []
        self._last_tick: dict[str, Any] | None = None
        self._running = False

    def add_listener(self, listener: TickListener) -> None:
        """Register a callback. Listeners are called in registration order."""
        self._listeners.append(listener)

    def remove_listener(self, listener: TickListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def run(self) -> None:
        """Block, polling and dispatching until ``stop()`` is called.

        Catches and logs listener exceptions — a buggy listener
        shouldn't kill the polling loop. Network/broker errors from
        ``get_current_price`` propagate; the main loop decides to
        reconnect or halt.
        """
        self._running = True
        logger.info(
            f"TickHandler started for {self._symbol} "
            f"(poll={int(self._poll_interval_sec * 1000)}ms)"
        )
        while self._running:
            tick = self._connector.get_current_price(self._symbol)
            if self._is_new_tick(tick):
                for listener in self._listeners:
                    try:
                        listener(tick, self._last_tick)
                    except Exception as e:
                        logger.exception(f"tick listener raised: {e}")
                self._last_tick = tick
            time.sleep(self._poll_interval_sec)
        logger.info(f"TickHandler stopped for {self._symbol}")

    def stop(self) -> None:
        """Signal ``run()`` to exit after the current iteration."""
        self._running = False

    def _is_new_tick(self, tick: dict[str, Any]) -> bool:
        if self._last_tick is None:
            return True
        # time_msc is millisecond resolution — any change means a new tick.
        return tick["time_msc"] != self._last_tick["time_msc"]
