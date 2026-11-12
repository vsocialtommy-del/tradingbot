"""Dashboard bridge (Nov 2026).

Everything the phone dashboard needs from a running bot, kept OUT of the
trading path:

* :func:`build_closed_trades` turns MT5 deal history into one row per
  closed position (for the monthly calendar).
* :func:`pick_closest_zones` picks the closest live buy and sell zone.
* :class:`DashboardBridge` writes the status row and the closed trades to
  the bot's own Supabase project.

The bridge must never stop the bot trading. Every write is wrapped; if the
dashboard tables don't exist yet (SQL 016 not run), it says so once and
quietly retries every few minutes.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from loguru import logger

from bot.risk.daily_halt import current_trading_date

# MT5 deal constants (numeric so tests don't need MT5).
DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_ENTRY_INOUT = 2
DEAL_ENTRY_OUT_BY = 3
DEAL_TYPE_BUY = 0
DEAL_TYPE_SELL = 1

_SETUP_TAG = re.compile(r"s=([0-9a-f]{8})")
_OUT_ENTRIES = (DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY)


def calendar_day(t: datetime) -> date:
    """The calendar day a time counts toward on the dashboard.

    The bot's trading day runs 17:00 New York to 17:00 New York (22:00 UK
    most of the year). It's labelled by the day it mostly covers, the same
    as MT5's server date: a trade at 09:00 UK on the 11th is the 11th, one
    at 22:30 UK on the 11th is already the 12th."""
    return current_trading_date(t) + timedelta(days=1)


def _deal_pnl(d: dict[str, Any]) -> float:
    return (
        float(d.get("profit", 0.0) or 0.0) + float(d.get("commission", 0.0) or 0.0)
        + float(d.get("swap", 0.0) or 0.0) + float(d.get("fee", 0.0) or 0.0)
    )


def build_closed_trades(
    deals: Iterable[dict[str, Any]],
    *,
    bot: str,
    magic: int,
    server_offset: timedelta,
    known: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """One row per CLOSED position the bot opened.

    ``deals`` are MT5 ``history_deals_get`` rows (``time`` is broker server
    time as epoch seconds; ``server_offset`` = server clock minus UTC).
    A position counts as closed once the volume closed out reaches the
    volume opened. Positions of the same setup share ``setup_key`` (the
    ``s=xxxxxxxx`` tag in the order comment, or the ticket the bot
    registered at placement), so the calendar can count trades, not
    layers.
    """
    ins: dict[int, dict[str, Any]] = {}
    outs: dict[int, list[dict[str, Any]]] = {}
    for d in deals:
        pos = d.get("position_id")
        if pos is None:
            continue
        entry = d.get("entry")
        if entry == DEAL_ENTRY_IN:
            if d.get("magic") == magic:
                ins[int(pos)] = d
        elif entry in _OUT_ENTRIES:
            outs.setdefault(int(pos), []).append(d)

    rows: list[dict[str, Any]] = []
    for pos, din in ins.items():
        closing = outs.get(pos)
        if not closing:
            continue
        vol_in = float(din.get("volume", 0.0) or 0.0)
        vol_out = sum(float(x.get("volume", 0.0) or 0.0) for x in closing)
        if vol_out + 1e-9 < vol_in:
            continue  # partly closed, still open
        last = max(closing, key=lambda x: float(x.get("time", 0)))
        close_px = (
            sum(float(x.get("price", 0.0)) * float(x.get("volume", 0.0)) for x in closing)
            / vol_out if vol_out > 0 else float(last.get("price", 0.0))
        )
        tag = (known or {}).get(pos)
        if tag is None:
            m = _SETUP_TAG.search(din.get("comment") or "")
            tag = m.group(1) if m else f"pos{pos}"
        open_utc = datetime.fromtimestamp(float(din.get("time", 0)), timezone.utc) - server_offset
        close_utc = datetime.fromtimestamp(float(last.get("time", 0)), timezone.utc) - server_offset
        profit = _deal_pnl(din) + sum(_deal_pnl(x) for x in closing)
        rows.append({
            "position_id": pos,
            "bot": bot,
            "setup_key": tag,
            "direction": "BUY" if din.get("type") == DEAL_TYPE_BUY else "SELL",
            "volume": round(vol_in, 2),
            "open_time": open_utc.isoformat(),
            "close_time": close_utc.isoformat(),
            "open_price": round(float(din.get("price", 0.0)), 2),
            "close_price": round(close_px, 2),
            "profit": round(profit, 2),
            "comment": din.get("comment") or None,
            "trading_day": calendar_day(close_utc).isoformat(),
        })
    rows.sort(key=lambda r: r["close_time"])
    return rows


@dataclass(frozen=True)
class ZoneView:
    direction: str
    top: float
    bottom: float
    entry: float
    sl: float | None
    tp1: float | None
    distance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "top": round(self.top, 2),
            "bottom": round(self.bottom, 2),
            "entry": round(self.entry, 2),
            "sl": None if self.sl is None else round(self.sl, 2),
            "tp1": None if self.tp1 is None else round(self.tp1, 2),
            "distance": round(self.distance, 2),
        }


def pick_closest_zones(
    zones: Iterable[ZoneView],
) -> tuple[ZoneView | None, ZoneView | None]:
    """Closest live BUY and closest live SELL (by distance to entry)."""
    best: dict[str, ZoneView] = {}
    for z in zones:
        cur = best.get(z.direction)
        if cur is None or z.distance < cur.distance:
            best[z.direction] = z
    return best.get("BUY"), best.get("SELL")


def _missing_table(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "does not exist" in text or "pgrst205" in text or "could not find the table" in text
        or "schema cache" in text or "404" in text
    )


class DashboardBridge:
    """Writes status + closed trades for the dashboard. Never raises."""

    RETRY_SECONDS = 300.0

    def __init__(self, supabase: Any, bot: str) -> None:
        self._sb = supabase
        self.bot = bot
        self._off_until = 0.0
        self._warned = False
        self.synced_positions: set[int] = set()
        # closed_trades.account (SQL 017): None = not tried yet, False = the
        # column isn't there, True = working.
        self._account_col: bool | None = None

    @property
    def available(self) -> bool:
        return time.monotonic() >= self._off_until

    def _failed(self, what: str, exc: Exception) -> None:
        if _missing_table(exc):
            if not self._warned:
                logger.warning(
                    f"dashboard: {what} skipped, the dashboard tables aren't in "
                    "Supabase yet (run migrations/016_dashboard.sql). Trading is "
                    "not affected; retrying every 5 minutes."
                )
                self._warned = True
            self._off_until = time.monotonic() + self.RETRY_SECONDS
        else:
            logger.warning(f"dashboard: {what} failed ({exc}); trading not affected")

    def write_status(self, data: dict[str, Any], now: datetime) -> bool:
        if not self.available:
            return False
        try:
            self._sb.upsert_rows(
                "bot_status",
                [{"bot": self.bot, "updated_at": now.isoformat(), "data": data}],
                on_conflict="bot",
            )
            self._warned = False
            return True
        except Exception as exc:  # noqa: BLE001
            self._failed("status write", exc)
            return False

    def sync_trades(self, rows: list[dict[str, Any]], account: int | None = None) -> int:
        """Upsert closed positions not sent yet. Returns how many were sent.

        ``account`` (the MT5 login) tags each row so the calendar follows
        the account the bot is logged into. The first time, rows saved
        before the tag existed are given this account (the demo)."""
        if not self.available:
            return 0
        if account is not None and self._account_col is None:
            try:
                self._sb.claim_untagged_trades(self.bot, account)
                self._account_col = True
            except Exception as exc:  # noqa: BLE001
                self._account_col = False
                logger.warning(
                    "dashboard: closed_trades has no account column yet (run "
                    f"migrations/017_account_login.sql); calendar not split by account ({exc})"
                )
        new = [r for r in rows if r["position_id"] not in self.synced_positions]
        if not new:
            return 0
        if account is not None and self._account_col:
            new = [{**r, "account": account} for r in new]
        try:
            for i in range(0, len(new), 200):
                self._sb.upsert_rows("closed_trades", new[i:i + 200], on_conflict="position_id")
        except Exception as exc:  # noqa: BLE001
            self._failed("trade sync", exc)
            return 0
        self.synced_positions.update(r["position_id"] for r in new)
        if len(new) > 1:
            logger.info(f"dashboard: synced {len(new)} closed positions for the calendar")
        return len(new)
