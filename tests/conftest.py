"""pytest session setup.

The :mod:`bot.execution.mt5_connector` module imports the
``MetaTrader5`` Python library at module load. That library is
**Windows-only** and cannot be installed on Linux test runners
(including the Hetzner VPS, where production runs MT5 + Python under
Wine — but pytest runs on a different process).

To let unit tests collect and run on any host, we inject a stub
``MetaTrader5`` module into :data:`sys.modules` before any test file
imports the connector. The stub provides the module-level constants
the connector reads at import time (``TIMEFRAME_*``, ``ORDER_TYPE_*``,
``TRADE_RETCODE_*``, etc.); methods are unused because tests always
work against a :class:`unittest.mock.MagicMock(spec=MT5Connector)`,
not the real MetaTrader5 API.
"""

from __future__ import annotations

import sys
import types

if "MetaTrader5" not in sys.modules:
    stub = types.ModuleType("MetaTrader5")

    # Timeframe constants (used as dict keys / values).
    stub.TIMEFRAME_M1 = 1
    stub.TIMEFRAME_M5 = 5
    stub.TIMEFRAME_M15 = 15
    stub.TIMEFRAME_M30 = 30
    stub.TIMEFRAME_H1 = 16385
    stub.TIMEFRAME_H4 = 16388
    stub.TIMEFRAME_D1 = 16408

    # Order types.
    stub.ORDER_TYPE_BUY = 0
    stub.ORDER_TYPE_SELL = 1
    stub.ORDER_TYPE_BUY_LIMIT = 2
    stub.ORDER_TYPE_SELL_LIMIT = 3

    # Trade actions.
    stub.TRADE_ACTION_DEAL = 1
    stub.TRADE_ACTION_PENDING = 5
    stub.TRADE_ACTION_SLTP = 6
    stub.TRADE_ACTION_MODIFY = 7

    # Order time / filling modes.
    stub.ORDER_TIME_GTC = 0
    stub.ORDER_FILLING_FOK = 0
    stub.ORDER_FILLING_IOC = 1
    stub.ORDER_FILLING_RETURN = 2

    # Position types.
    stub.POSITION_TYPE_BUY = 0
    stub.POSITION_TYPE_SELL = 1

    # Trade return codes — only the few connector code references.
    stub.TRADE_RETCODE_DONE = 10009
    stub.TRADE_RETCODE_REQUOTE = 10004
    stub.TRADE_RETCODE_REJECT = 10006
    stub.TRADE_RETCODE_CANCEL = 10007
    stub.TRADE_RETCODE_PRICE_OFF = 10021
    stub.TRADE_RETCODE_PRICE_CHANGED = 10020
    stub.TRADE_RETCODE_INVALID_FILL = 10030
    stub.TRADE_RETCODE_NO_MONEY = 10019

    # Functions — unused in tests (we always mock MT5Connector itself)
    # but referenced at import time. Stub them to avoid AttributeError.
    def _noop(*args, **kwargs):
        return None

    stub.initialize = _noop
    stub.shutdown = _noop
    stub.last_error = lambda: (0, "ok")
    stub.account_info = _noop
    stub.symbol_info = _noop
    stub.symbol_info_tick = _noop
    stub.symbol_select = _noop
    stub.copy_rates_from_pos = _noop
    stub.order_send = _noop
    stub.positions_get = _noop

    sys.modules["MetaTrader5"] = stub


# ---------------------------------------------------------------------------
# Stub the ``supabase`` package the same way. The bot imports
# ``Client`` and ``create_client`` at module load — neither is needed
# for unit tests that mock :class:`SupabaseLogger`.
# ---------------------------------------------------------------------------
if "supabase" not in sys.modules:
    supabase_stub = types.ModuleType("supabase")

    class _StubClient:
        """Placeholder — never instantiated in tests, but referenced
        in type hints inside the connector."""

    def _create_client(*args, **kwargs):
        return _StubClient()

    supabase_stub.Client = _StubClient
    supabase_stub.create_client = _create_client
    sys.modules["supabase"] = supabase_stub
