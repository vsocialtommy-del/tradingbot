"""XAUTISM M5 close exit (Nov 2026, step 6).

Extra exit on top of the broker SL: if an M5 candle **closes** beyond the
far edge of the zone (below the bottom for a buy, above the top for a
sell), the zone has been closed through. Every open layer is closed at
market and a pending layer 2 stop is cancelled, with
``close_reason='M5_CLOSE_EXIT'``.

Only candles that closed after the setup went live count.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

import pandas as pd
from loguru import logger

from bot.execution.layer2_stop import cancel_layer2_at_broker
from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import OPEN_TRADE_STATUSES, PositionTracker
from bot.logging.supabase_logger import Setup

M5 = timedelta(minutes=5)


@dataclass(frozen=True)
class M5CloseExit:
    setup_id: UUID
    close_price: float
    closed_layers: int
    cancelled_layers: int


def closed_beyond_far_edge(
    direction: str, close: float, zone_top: float, zone_bottom: float,
) -> bool:
    if direction == "BUY":
        return close < zone_bottom
    return close > zone_top


def run_m5_close_exit(
    setup: Setup,
    last_closed_bar: pd.Series,
    bar_open_time: pd.Timestamp,
    *,
    zone_top: float,
    zone_bottom: float,
    mt5: MT5Connector,
    tracker: PositionTracker,
) -> M5CloseExit | None:
    """Apply the exit to one setup for one just-closed M5 bar."""
    if setup.status != "ACTIVE":
        return None
    bar_close_time = pd.Timestamp(bar_open_time) + M5
    if setup.activated_at is not None and bar_close_time <= pd.Timestamp(setup.activated_at):
        return None  # candle closed before we were in

    close = float(last_closed_bar["close"])
    if not closed_beyond_far_edge(setup.direction, close, zone_top, zone_bottom):
        return None

    closed = cancelled = 0
    for trade in tracker.get_trades_for_setup(setup.id):
        if trade.status in OPEN_TRADE_STATUSES and trade.mt5_ticket is not None:
            try:
                mt5.close_position(trade.mt5_ticket)
            except Exception:
                logger.exception(
                    f"m5 close exit: close of {trade.mt5_ticket} failed "
                    f"(setup {setup.id}); broker SL still protects it"
                )
                continue
            tracker.update_trade_status(
                trade.id, "CLOSED", close_reason="M5_CLOSE_EXIT",
            )
            closed += 1
        elif trade.status == "WAITING":
            cancel_layer2_at_broker(mt5, trade)
            tracker.update_trade_status(
                trade.id, "CANCELLED", close_reason="M5_CLOSE_EXIT",
            )
            cancelled += 1

    logger.info(
        f"M5_CLOSE_EXIT: setup={setup.id} {setup.direction} candle closed "
        f"{close:.2f} beyond zone {zone_bottom:.2f}-{zone_top:.2f}; "
        f"closed {closed}, cancelled {cancelled}"
    )
    return M5CloseExit(setup.id, close, closed, cancelled)
