"""MT5 terminal connection wrapper.

Thin object-oriented wrapper around the ``MetaTrader5`` Python library.
All other modules talk to the broker through this layer so the
integration stays swappable and testable.

Design notes
------------
* **Floats at the API boundary.** MT5 speaks float64 throughout — prices,
  volumes, balances. We keep that here. Conversion to ``Decimal`` happens
  at the persistence boundary (``supabase_logger``), where pydantic
  coerces automatically.
* **Magic number tagging.** Every order this bot places is stamped with
  a magic number so position queries can filter "ours" from any manual
  trades the operator places in the same MT5 terminal.
* **Filling mode is broker-specific.** We probe ``symbol_info.filling_mode``
  (a bitmask) and pick a supported mode at runtime. Pending orders are
  always sent with ``ORDER_FILLING_RETURN`` since most brokers reject
  IOC/FOK on pendings.
* **Retry policy.** ``order_send`` is retried with exponential backoff on
  transient retcodes (REQUOTE, PRICE_OFF, PRICE_CHANGED) and on null
  results (connection blip). Hard rejections (NO_MONEY, INVALID_FILL,
  REJECT) raise immediately — retrying won't help.
* **Timezones.** MT5 returns Unix seconds for bar/tick times. We convert
  to timezone-aware UTC ``datetime`` at the boundary so callers never
  see naive timestamps.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
# Lazy MetaTrader5 import
#
# The ``MetaTrader5`` package is **Windows-only** (or Wine on Linux for the
# Hetzner VPS). Importing this module on Linux/macOS — e.g. for the
# backtest framework, which only needs the ``MT5Connector`` class for
# type annotations — must not fail.
#
# The :class:`_LazyMT5` proxy intercepts attribute access and triggers the
# real ``import MetaTrader5`` only when something actually uses it. The
# error message points users at the backtest framework if they hit this
# path on a Linux host.
# --------------------------------------------------------------------------- #


class _LazyMT5:
    """Lazy proxy for ``MetaTrader5``. ``mt5.foo()`` triggers real import."""

    _module: Any = None

    def __getattr__(self, name: str) -> Any:
        if _LazyMT5._module is None:
            try:
                import MetaTrader5 as _real_mt5  # noqa: PLC0415
            except ImportError as e:
                raise ImportError(
                    "MetaTrader5 is required for live MT5 connections but "
                    "is not installed. The library is Windows-only "
                    "(use a Windows VPS, or Linux+Wine for production). "
                    "If you only need the backtest framework, use "
                    "``bot.backtest`` — it does not depend on MetaTrader5."
                ) from e
            _LazyMT5._module = _real_mt5
        return getattr(_LazyMT5._module, name)


mt5: Any = _LazyMT5()

Direction = Literal["BUY", "SELL"]
Timeframe = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


# These dicts dereference ``mt5.TIMEFRAME_*`` / ``mt5.TRADE_RETCODE_*``
# constants. Building them at module import time would force the lazy
# proxy to load MetaTrader5 — which is exactly what we're trying to
# avoid. Defer to first-use via the getter functions below.

_TIMEFRAME_MAP: dict[str, int] | None = None
_RETRYABLE_RETCODES: set[int] | None = None


def _get_timeframe_map() -> dict[str, int]:
    global _TIMEFRAME_MAP
    if _TIMEFRAME_MAP is None:
        _TIMEFRAME_MAP = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
    return _TIMEFRAME_MAP


def _get_retryable_retcodes() -> set[int]:
    global _RETRYABLE_RETCODES
    if _RETRYABLE_RETCODES is None:
        _RETRYABLE_RETCODES = {
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
        }
    return _RETRYABLE_RETCODES

DEFAULT_DEVIATION_POINTS = 20  # max acceptable slippage on market orders
DEFAULT_MAGIC = 234567         # bot's order tag — distinguishes from manual trades
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SEC = 1.0
DEFAULT_CONNECT_ATTEMPTS = 3


class MT5Error(Exception):
    """Raised for any MT5 API failure that callers should treat as fatal."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class MT5Connector:
    """Connection + order management wrapper around MetaTrader5."""

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str | None = None,
        magic: int = DEFAULT_MAGIC,
        deviation_points: int = DEFAULT_DEVIATION_POINTS,
    ) -> None:
        self._login = int(login)
        self._password = password
        self._server = server
        self._path = path
        self._magic = magic
        self._deviation = deviation_points
        self._connected = False

    @classmethod
    def from_env(cls) -> MT5Connector:
        """Build from ``MT5_LOGIN``, ``MT5_PASSWORD``, ``MT5_SERVER``, ``MT5_PATH``."""
        return cls(
            login=int(os.environ["MT5_LOGIN"]),
            password=os.environ["MT5_PASSWORD"],
            server=os.environ["MT5_SERVER"],
            path=os.environ.get("MT5_PATH") or None,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self, max_attempts: int = DEFAULT_CONNECT_ATTEMPTS) -> None:
        """Initialise the MT5 terminal connection and verify login."""
        kwargs: dict[str, Any] = {
            "login": self._login,
            "password": self._password,
            "server": self._server,
        }
        if self._path:
            kwargs["path"] = self._path

        last_err: tuple[int, str] | None = None
        for attempt in range(max_attempts):
            if mt5.initialize(**kwargs):
                info = mt5.account_info()
                if info is None:
                    last_err = mt5.last_error()
                    mt5.shutdown()
                    logger.warning(
                        f"connect attempt {attempt + 1}: account_info() returned None ({last_err})"
                    )
                elif info.login != self._login:
                    mt5.shutdown()
                    raise MT5Error(
                        f"connected as login={info.login}, expected {self._login} "
                        f"(check MT5_LOGIN / cached terminal credentials)"
                    )
                else:
                    self._connected = True
                    logger.info(
                        f"MT5 connected: login={info.login}, server={info.server}, "
                        f"balance={info.balance} {info.currency}"
                    )
                    return
            else:
                last_err = mt5.last_error()
                logger.warning(
                    f"connect attempt {attempt + 1} failed: {last_err}"
                )

            time.sleep(DEFAULT_RETRY_BACKOFF_SEC * (2**attempt))

        raise MT5Error(
            f"connect failed after {max_attempts} attempts: {last_err}",
            code=last_err[0] if last_err else None,
        )

    def disconnect(self) -> None:
        """Shut down the terminal connection. Idempotent."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def __enter__(self) -> MT5Connector:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def get_balance(self) -> float:
        """Return the current account balance."""
        info = mt5.account_info()
        if info is None:
            raise MT5Error(f"account_info() returned None: {mt5.last_error()}")
        return float(info.balance)

    def get_account_info(self) -> dict[str, Any]:
        """Return the full ``AccountInfo`` namedtuple as a dict."""
        info = mt5.account_info()
        if info is None:
            raise MT5Error(f"account_info() returned None: {mt5.last_error()}")
        return info._asdict()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------
    def get_current_price(self, symbol: str) -> dict[str, Any]:
        """Return the latest tick as ``{'bid', 'ask', 'time'}`` (UTC)."""
        self._ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(
                f"symbol_info_tick({symbol}) returned None: {mt5.last_error()}"
            )
        return {
            "bid": float(tick.bid),
            "ask": float(tick.ask),
            "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
            "time_msc": int(tick.time_msc),
        }

    def get_ohlc(
        self, symbol: str, timeframe: Timeframe, count: int
    ) -> pd.DataFrame:
        """Return the last ``count`` bars as a UTC-indexed DataFrame.

        Columns: ``open, high, low, close, volume`` (where ``volume`` is
        MT5's ``tick_volume`` — number of ticks during the bar; CFDs
        rarely have real exchange volume).
        """
        if timeframe not in _get_timeframe_map():
            raise MT5Error(
                f"unknown timeframe {timeframe!r}; supported: "
                f"{sorted(_get_timeframe_map())}"
            )
        self._ensure_symbol(symbol)
        rates = mt5.copy_rates_from_pos(
            symbol, _get_timeframe_map()[timeframe], 0, count
        )
        if rates is None or len(rates) == 0:
            raise MT5Error(
                f"copy_rates_from_pos({symbol},{timeframe}) returned no data: "
                f"{mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").rename(columns={"tick_volume": "volume"})
        return df[["open", "high", "low", "close", "volume"]]

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def place_market_order(
        self,
        symbol: str,
        direction: Direction,
        lot_size: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> int:
        """Place a market order (Layer 1 entry). Returns the broker ticket."""
        self._ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise MT5Error(f"no tick for {symbol}: {mt5.last_error()}")

        order_type = (
            mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        )
        price = tick.ask if direction == "BUY" else tick.bid

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(price),
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": comment or "bot:market",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling_mode(symbol),
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)

        result = self._send_order_with_retry(request)
        # For market deals, order ID is in `order`; deal ID in `deal`.
        return int(result.order or result.deal)

    def place_limit_order(
        self,
        symbol: str,
        direction: Direction,
        lot_size: float,
        price: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> int:
        """Place a pending limit order (Layers 2/3). Returns the order ticket."""
        self._ensure_symbol(symbol)
        order_type = (
            mt5.ORDER_TYPE_BUY_LIMIT
            if direction == "BUY"
            else mt5.ORDER_TYPE_SELL_LIMIT
        )

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(lot_size),
            "type": order_type,
            "price": float(price),
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": comment or "bot:limit",
            "type_time": mt5.ORDER_TIME_GTC,
            # Pending orders are filled when price reaches them — most
            # brokers only accept RETURN here.
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        if sl is not None:
            request["sl"] = float(sl)
        if tp is not None:
            request["tp"] = float(tp)

        result = self._send_order_with_retry(request)
        return int(result.order)

    def modify_order(
        self,
        ticket: int,
        sl: float | None = None,
        tp: float | None = None,
    ) -> None:
        """Modify SL/TP on an open position.

        Used by ``tp1_manager`` for the SL→break-even move (spec 6.1).
        Raises ``MT5Error`` if the ticket is not an open position. To
        modify a pending order's price, cancel and re-place — most
        brokers require a separate code path.
        """
        position = self._get_position(ticket)
        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "magic": self._magic,
            # MT5 requires both fields; pass current values for the side
            # we're not changing so we don't accidentally clear it.
            "sl": float(sl) if sl is not None else float(position.sl),
            "tp": float(tp) if tp is not None else float(position.tp),
        }
        self._send_order_with_retry(request)

    def close_position(
        self,
        ticket: int,
        partial_lots: float | None = None,
    ) -> None:
        """Close an open position fully or partially.

        ``partial_lots=None`` closes the whole position; otherwise closes
        that many lots (used at TP1 to take 50% — spec 6.1). MT5
        validates the volume against the broker's minimum lot step.
        """
        position = self._get_position(ticket)
        volume = (
            float(partial_lots) if partial_lots is not None else float(position.volume)
        )
        if volume <= 0 or volume > position.volume:
            raise MT5Error(
                f"invalid close volume {volume} for position {ticket} "
                f"(open volume={position.volume})"
            )

        # Closing reverses direction.
        is_long = position.type == mt5.POSITION_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY

        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            raise MT5Error(f"no tick for {position.symbol}: {mt5.last_error()}")
        price = tick.bid if is_long else tick.ask

        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": float(price),
            "deviation": self._deviation,
            "magic": self._magic,
            "comment": "bot:close" if partial_lots is None else "bot:close_partial",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._pick_filling_mode(position.symbol),
        }
        self._send_order_with_retry(request)

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """List positions opened by this bot (filtered by magic number)."""
        positions = (
            mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        )
        if positions is None:
            return []
        return [p._asdict() for p in positions if p.magic == self._magic]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise MT5Error(f"unknown symbol: {symbol} ({mt5.last_error()})")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise MT5Error(
                f"symbol_select({symbol}) failed: {mt5.last_error()} — "
                f"is the symbol enabled in the broker's instrument list?"
            )

    def _pick_filling_mode(self, symbol: str) -> int:
        """Choose the first filling mode the broker advertises support for.

        ``symbol_info.filling_mode`` is a bitmask (FOK=1, IOC=2). If
        neither bit is set we fall back to RETURN, which is the
        most permissive but slowest mode.
        """
        info = mt5.symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC  # benign default
        if info.filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        if info.filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _get_position(self, ticket: int) -> Any:
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            raise MT5Error(f"position {ticket} not found: {mt5.last_error()}")
        return positions[0]

    def _send_order_with_retry(
        self,
        request: dict[str, Any],
        max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    ) -> Any:
        last_result: Any = None
        for attempt in range(max_attempts):
            result = mt5.order_send(request)

            if result is None:
                # Connection blip or terminal hiccup — worth retrying.
                err = mt5.last_error()
                logger.warning(
                    f"order_send returned None (attempt {attempt + 1}/{max_attempts}): {err}"
                )
                last_result = None
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"order ok: ticket={result.order} deal={result.deal} "
                    f"vol={result.volume} price={result.price}"
                )
                return result
            elif result.retcode in _get_retryable_retcodes():
                logger.warning(
                    f"order requote (attempt {attempt + 1}/{max_attempts}): "
                    f"retcode={result.retcode} — refreshing price"
                )
                last_result = result
                # Refresh price for market deals; pendings keep their original price.
                if request.get("action") == mt5.TRADE_ACTION_DEAL:
                    tick = mt5.symbol_info_tick(request["symbol"])
                    if tick is not None:
                        request["price"] = (
                            tick.ask
                            if request["type"] == mt5.ORDER_TYPE_BUY
                            else tick.bid
                        )
            else:
                # Hard rejection — no point retrying.
                raise MT5Error(
                    f"order_send rejected: retcode={result.retcode} "
                    f"comment={result.comment!r}",
                    code=result.retcode,
                )

            time.sleep(DEFAULT_RETRY_BACKOFF_SEC * (2**attempt))

        if last_result is None:
            raise MT5Error("order_send: connection lost across all retries")
        raise MT5Error(
            f"order_send: retries exhausted (last retcode={last_result.retcode} "
            f"comment={last_result.comment!r})",
            code=last_result.retcode,
        )
