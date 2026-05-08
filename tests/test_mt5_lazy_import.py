"""Regression tests for the lazy ``MetaTrader5`` import.

The ``MetaTrader5`` library is Windows-only. Two guarantees we lock down
here:

1. **Importing ``bot.backtest`` (and its transitive deps) does NOT
   trigger ``import MetaTrader5``** — backtest must run on Linux/macOS
   (Colab, CI, dev laptops) without the live trading library
   installed.

2. **Importing ``bot.execution.mt5_connector`` itself ALSO does not
   trigger MetaTrader5** — only actually using a function or constant
   on the connector does. This means type-only imports (``from
   ...mt5_connector import MT5Connector`` for parameter annotations)
   work everywhere.

3. **Calling a real method without the library raises a clear
   ImportError** that points users at the backtest framework.

Tests run each scenario in a subprocess so we don't pollute the parent
process's ``sys.modules`` cache (``MetaTrader5`` may already be loaded
in the parent, which would mask the bug).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    """Run a Python script with MetaTrader5 hidden via sys.modules."""
    cleaned = textwrap.dedent(script).strip()
    prelude = (
        "import sys\n"
        # Hiding the module — any attempt to ``import MetaTrader5`` will
        # raise ModuleNotFoundError as if it weren't installed.
        "sys.modules['MetaTrader5'] = None\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + cleaned + "\n"],
        capture_output=True, text=True, timeout=30,
    )


# --------------------------------------------------------------------------- #
# Backtest decoupling
# --------------------------------------------------------------------------- #


class TestBacktestDecoupling:
    def test_bot_backtest_imports_without_metatrader5(self) -> None:
        """The headline guarantee: bot.backtest is usable on Linux/Colab."""
        result = _run_in_subprocess(
            """
            from bot.backtest import (
                BacktestConfig,
                BacktestEngine,
                BacktestResult,
                load_dukascopy_csv,
                generate_html_report,
                generate_equity_curve,
            )
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_running_engine_without_metatrader5(self) -> None:
        """End-to-end smoke: can we actually RUN a backtest with MT5 hidden?"""
        result = _run_in_subprocess(
            """
            import numpy as np
            import pandas as pd
            from bot.backtest import BacktestConfig, BacktestEngine

            n = 150
            times = pd.date_range(
                "2026-01-01T00:00:00Z", periods=n, freq="5min", tz="UTC",
            )
            df = pd.DataFrame(
                {
                    "open":  [1900.0] * n,
                    "high":  [1900.5] * n,
                    "low":   [1899.5] * n,
                    "close": [1900.0] * n,
                    "volume":[100] * n,
                },
                index=times,
            )
            cfg = BacktestConfig(
                min_history_bars=100, progress_log_every_bars=0,
            )
            result = BacktestEngine(cfg).run(df)
            print(f"BARS={result.bars_processed}")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "BARS=150" in result.stdout


# --------------------------------------------------------------------------- #
# mt5_connector decoupling
# --------------------------------------------------------------------------- #


class TestMt5ConnectorImport:
    def test_module_imports_without_metatrader5(self) -> None:
        """The module itself imports — only DEREFERENCING the proxy fails."""
        result = _run_in_subprocess(
            """
            from bot.execution.mt5_connector import (
                MT5Connector, MT5Error, Direction, Timeframe,
            )
            # Class definitions and type aliases are accessible.
            assert MT5Error
            assert MT5Connector
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_construction_does_not_trigger_metatrader5(self) -> None:
        """``MT5Connector(...)`` doesn't touch the library — only ``connect()``."""
        result = _run_in_subprocess(
            """
            from bot.execution.mt5_connector import MT5Connector
            connector = MT5Connector(
                login=12345, password="x", server="srv", path=None,
            )
            assert connector is not None
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_calling_real_mt5_method_raises_clear_importerror(self) -> None:
        """Actual MT5 use without the library raises a helpful ImportError."""
        result = _run_in_subprocess(
            """
            from bot.execution.mt5_connector import MT5Connector
            c = MT5Connector(login=1, password="x", server="srv")
            try:
                c.connect()  # triggers mt5.initialize() under the hood
            except ImportError as e:
                msg = str(e)
                assert "MetaTrader5 is required" in msg
                assert "bot.backtest" in msg, (
                    "error message should point users at the backtest framework"
                )
                print("OK_IMPORTERROR")
            else:
                print("UNEXPECTED_NO_RAISE")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK_IMPORTERROR" in result.stdout

    def test_timeframe_constants_lazy(self) -> None:
        """Importing ``mt5_connector`` does NOT touch ``mt5.TIMEFRAME_*``.

        If ``_TIMEFRAME_MAP`` were eagerly built at module load, the
        proxy's ``__getattr__`` would have triggered ``import
        MetaTrader5`` already and the test in this subprocess would
        have failed at module import time.
        """
        result = _run_in_subprocess(
            """
            import bot.execution.mt5_connector as mt5_mod
            # Sentinels must still be ``None`` post-import.
            assert mt5_mod._TIMEFRAME_MAP is None, (
                "_TIMEFRAME_MAP was eagerly built — would have required MetaTrader5"
            )
            assert mt5_mod._RETRYABLE_RETCODES is None, (
                "_RETRYABLE_RETCODES was eagerly built — same problem"
            )
            print("OK")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
