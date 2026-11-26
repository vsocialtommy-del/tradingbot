"""Trade photo journal runner (Tommy, 26 Nov 2026).

Every closed trade gets a picture (``trade_chart.render``) 30 minutes after
its last position closed, so the picture also shows what price did next.
The picture goes to the Supabase storage bucket ``trade-charts`` and a row
in ``trade_charts`` holds its path plus the trade's numbers (``summary``)
for analysis later. The dashboard shows it when you tap the trade.

Never affects trading: candles are read from MT5 in the bot's loop (cheap),
everything else (Supabase lookups, drawing, upload) runs on one background
thread, one trade at a time. Anything that fails is logged and retried a
few times, then left.

Needs ``migrations/018_trade_charts.sql``; until it's run the journal
switches itself off with one warning.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger

from bot.journal.trade_chart import ChartTrade, Layer, classify_exit, render, summary

BUCKET = "trade-charts"
AFTER_CLOSE = timedelta(minutes=30)
MAX_TRIES = 3
# How much chart before the entry, at most (the zone may be older).
LOOKBACK = {"M1": timedelta(hours=3), "M5": timedelta(hours=12)}
LEAD_BARS = 25            # candles shown before the zone formed


def _ts(v: Any) -> datetime:
    t = pd.Timestamp(v)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC").to_pydatetime()


def layers_from_rows(rows: list[dict[str, Any]]) -> list[Layer]:
    """closed_trades rows of one setup -> named layers, oldest first. The
    last of three is the runner."""
    rows = sorted(rows, key=lambda r: (str(r["open_time"]), int(r["position_id"])))
    n = len(rows)
    out: list[Layer] = []
    for i, r in enumerate(rows):
        name = f"L{i + 1}" + (" runner" if n >= 3 and i == n - 1 else "")
        out.append(Layer(name, float(r["volume"]), _ts(r["open_time"]), float(r["open_price"]),
                         _ts(r["close_time"]), float(r["close_price"]), float(r["profit"])))
    return out


def h1_trend_at(m5: pd.DataFrame, when: datetime, period: int = 50
                ) -> tuple[str | None, float | None, float | None, pd.Series]:
    """H1 trend at ``when`` from M5 candles (UTC index), the bot's rule:
    last CLOSED H1 close vs the 50 EMA of H1 closes. Also returns the EMA
    as a series indexed by when each H1 candle closed (for drawing)."""
    if m5 is None or len(m5) == 0:
        return None, None, None, pd.Series(dtype=float)
    h1 = m5["close"].astype(float).resample("1h").last().dropna()
    ema = h1.ewm(span=period, adjust=False).mean()
    ema_closed = pd.Series(ema.to_numpy(), index=ema.index + pd.Timedelta(hours=1))
    done = h1[h1.index + pd.Timedelta(hours=1) <= pd.Timestamp(when)]
    if len(done) < period:
        return None, None, None, ema_closed
    e = float(ema.loc[done.index[-1]])
    c = float(done.iloc[-1])
    return ("UPTREND" if c > e else "DOWNTREND"), round(c, 2), round(e, 2), ema_closed


class TradeJournal:
    def __init__(
        self, supabase: Any, *, bot: str, bot_label: str, symbol: str,
        since: datetime, default_tf: str, ema_period: int = 50,
    ) -> None:
        self._sb = supabase
        self.bot = bot
        self.bot_label = bot_label
        self.symbol = symbol
        self.since = since
        self.default_tf = default_tf
        self.ema_period = ema_period
        self._pending: dict[str, list[dict[str, Any]]] = {}
        self._done: set[str] | None = None
        self._account: int | None = None
        self._tries: dict[str, int] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="journal")
        self._job: tuple[str, Future] | None = None
        self._off_until = 0.0
        self._warned = False

    # ---- called from the bot loop ----------------------------------------

    def note_rows(self, rows: list[dict[str, Any]], account: int | None) -> None:
        """closed_trades rows from the dashboard sync (every position the bot
        closed in the window)."""
        if account != self._account:
            self._account = account
            self._done = None
            self._pending.clear()
        for r in rows:
            if _ts(r["open_time"]) < self.since:
                continue
            self._pending.setdefault(str(r["setup_key"]), [])
            lst = self._pending[str(r["setup_key"])]
            if all(x["position_id"] != r["position_id"] for x in lst):
                lst.append(r)

    def due(self, now: datetime) -> list[str]:
        """Setups whose last position closed 30+ min ago, not drawn yet."""
        out = []
        for key, rows in self._pending.items():
            if self._done is not None and key in self._done:
                continue
            if self._tries.get(key, 0) >= MAX_TRIES:
                continue
            first = min(_ts(r["open_time"]) for r in rows)
            last = max(_ts(r["close_time"]) for r in rows)
            if first < self.since or now - last < AFTER_CLOSE:
                continue
            out.append((last, key))
        return [k for _, k in sorted(out)]

    def step(self, now: datetime, get_ohlc: Callable[[str, int], pd.DataFrame],
             offset: timedelta | None) -> None:
        """Collect a finished job, then start the next one. ``get_ohlc(tf,
        count)`` reads MT5 (main thread only); ``offset`` = broker clock
        minus UTC."""
        if offset is None or time.monotonic() < self._off_until:
            return
        if self._job is not None:
            key, fut = self._job
            if not fut.done():
                return
            self._job = None
            self._finish(key, fut)
        if self._done is None:
            try:
                self._done = set(self._sb.trade_chart_keys(self.bot, self._account or 0))
            except Exception as exc:  # noqa: BLE001
                self._failed("reading trade_charts", exc)
                return
        keys = self.due(now)
        if not keys:
            return
        key = keys[0]
        rows = self._pending[key]
        first = min(_ts(r["open_time"]) for r in rows)
        try:
            m1 = self._utc(get_ohlc("M1", self._count(now, first, LOOKBACK["M1"], 1)), offset)
            m5 = self._utc(get_ohlc("M5", self._count(now, first, timedelta(days=5), 5)), offset)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"journal: reading candles for {key} failed ({exc}); will retry")
            self._tries[key] = self._tries.get(key, 0) + 1
            return
        self._job = (key, self._pool.submit(self._make, key, list(rows), m1, m5, self._account))

    # ---- background thread ------------------------------------------------

    def _make(self, key: str, rows: list[dict[str, Any]], m1: pd.DataFrame,
              m5: pd.DataFrame, account: int | None) -> str:
        trade = self.build_trade(key, rows, account, m5)
        candles = m1 if trade.tf == "M1" else m5
        first, last = trade.opened, trade.closed
        start = first - LOOKBACK.get(trade.tf, LOOKBACK["M5"])
        if trade.zone_formed is not None:
            bar = timedelta(minutes=1 if trade.tf == "M1" else 5)
            start = max(start, trade.zone_formed - LEAD_BARS * bar)
        window = candles[(candles.index >= pd.Timestamp(start)) &
                         (candles.index <= pd.Timestamp(last + AFTER_CLOSE))]
        if len(window) < 5:
            raise ValueError(f"only {len(window)} candles for the chart")
        _, _, _, ema = h1_trend_at(m5, first, self.ema_period)
        png = render(trade, window, ema)
        info = summary(trade, window)
        day = pd.Timestamp(first).tz_convert("Europe/London").strftime("%Y-%m-%d")
        path = f"{self.bot}/{account or 'acct'}/{day}/{key}.png"
        self._sb.upload_file(BUCKET, path, png, "image/png")
        self._sb.upsert_rows("trade_charts", [{
            "bot": self.bot, "account": account or 0, "setup_key": key, "path": path,
            "opened_at": first.isoformat(), "closed_at": last.isoformat(), "data": info,
        }], on_conflict="bot,account,setup_key")
        return path

    def build_trade(self, key: str, rows: list[dict[str, Any]], account: int | None,
                    m5: pd.DataFrame) -> ChartTrade:
        layers = layers_from_rows(rows)
        setup = zone = None
        notes: list[str] = []
        if len(key) == 8:
            try:
                setup = self._sb.find_setup_by_key(key, layers[0].open_time)
                if setup and setup.get("zone_id"):
                    zone = self._sb.get_zone_row(setup["zone_id"])
            except Exception as exc:  # noqa: BLE001
                notes.append("zone details not found")
                logger.debug(f"journal: setup lookup for {key} failed ({exc})")
        ev = (zone or {}).get("last_evaluation_result") or {}
        tf = str(ev.get("timeframe") or self.default_tf)
        kind = "FVG" if str(ev.get("kind") or "").upper() == "FVG" else "W/M"
        state, h1c, h1e, _ = h1_trend_at(m5, layers[0].open_time, self.ema_period)
        f = (lambda k: float(setup[k]) if setup and setup.get(k) is not None else None)
        t = ChartTrade(
            bot_label=self.bot_label, account=account, setup_key=key,
            direction=str(rows[0]["direction"]), tf=tf, kind=kind, layers=layers,
            zone_top=float(zone["top"]) if zone else None,
            zone_bottom=float(zone["bottom"]) if zone else None,
            zone_formed=_ts(zone["formed_at"]) if zone and zone.get("formed_at") else None,
            entry=f("planned_layer1_price"), sl=f("planned_sl_price"), tp1=f("planned_tp1_price"),
            h1_state=state, h1_close=h1c, h1_ema=h1e, notes=notes,
        )
        if setup is None and not notes:
            t.notes.append("zone details not found")
        for ly in t.layers:
            ly.how = classify_exit(t, ly)
        return t

    # ---- helpers ------------------------------------------------------------

    def _finish(self, key: str, fut: Future) -> None:
        try:
            path = fut.result()
        except Exception as exc:  # noqa: BLE001
            if _missing(exc):
                self._failed("saving a trade picture", exc)
                return
            self._tries[key] = self._tries.get(key, 0) + 1
            logger.warning(f"journal: picture for trade {key} failed ({exc}); "
                           f"try {self._tries[key]} of {MAX_TRIES}")
            return
        if self._done is not None:
            self._done.add(key)
        self._pending.pop(key, None)
        logger.info(f"journal: trade picture saved for {key} ({path})")

    def _failed(self, what: str, exc: Exception) -> None:
        if _missing(exc):
            if not self._warned:
                logger.warning(f"journal: {what} skipped, run migrations/018_trade_charts.sql in "
                               "Supabase to turn on trade pictures. Trading not affected.")
                self._warned = True
            self._off_until = time.monotonic() + 600
        else:
            logger.warning(f"journal: {what} failed ({exc}); trading not affected")
            self._off_until = time.monotonic() + 120

    @staticmethod
    def _count(now: datetime, first: datetime, back: timedelta, minutes: int) -> int:
        span = (now - (first - back)).total_seconds() / 60 / minutes
        return int(min(max(span + 30, 100), 60000))

    @staticmethod
    def _utc(df: pd.DataFrame, offset: timedelta) -> pd.DataFrame:
        """MT5 candles are stamped in broker time; shift to real UTC."""
        out = df.copy()
        out.index = out.index - offset
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _missing(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("trade_charts" in s and ("does not exist" in s or "not find" in s or "42p01" in s)) \
        or ("bucket not found" in s)
