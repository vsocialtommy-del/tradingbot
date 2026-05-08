"""End-to-end "this code base must work in Colab" guarantees.

Two flavours of test live here:

1. **Holistic** — hide every Colab-incompatible dep simultaneously and
   assert that ``bot.backtest`` imports clean and an engine actually
   runs end-to-end. This catches future regressions where someone
   adds a fresh import that pulls in a new heavy dep.

2. **Per-dep** — focused tests for each lazy-loaded library (currently
   ``supabase``; ``MetaTrader5`` has its own file
   ``test_mt5_lazy_import.py``). Asserts the specific module imports
   clean, calling the live API surfaces a clear ImportError pointing
   users at ``bot.backtest`` for the offline path.

Tests run each scenario in a **subprocess** because in our dev/CI env
the deps ARE installed; in-process ``sys.modules`` shadowing leaks
across tests.

If you add a new heavy dep to the live bot's path:

* If the backtest LEGITIMATELY needs it → install it in the Colab
  notebook's ``%pip install`` cell.
* If the backtest does NOT need it → add it to ``HIDDEN_LIBS`` below
  AND make the import lazy at its source (see
  :mod:`bot.execution.mt5_connector._LazyMT5` for the pattern).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# Libraries the backtest must NOT need. Add new entries here when
# making a new dep lazy.
HIDDEN_LIBS: tuple[str, ...] = (
    "MetaTrader5",  # Windows-only (PR #24)
    "supabase",     # backtest doesn't write to DB (this PR)
)


def _run_in_subprocess(
    script: str, *, hide: tuple[str, ...] = HIDDEN_LIBS,
) -> subprocess.CompletedProcess[str]:
    """Run a Python script with the named libraries hidden."""
    cleaned = textwrap.dedent(script).strip()
    hide_lines = "\n".join(
        f"sys.modules[{lib!r}] = None" for lib in hide
    )
    prelude = (
        "import sys\n"
        f"{hide_lines}\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + cleaned + "\n"],
        capture_output=True, text=True, timeout=30,
    )


# --------------------------------------------------------------------------- #
# Holistic — backtest works with EVERY incompatible dep hidden
# --------------------------------------------------------------------------- #


class TestColabBacktestBundle:
    """Full backtest path with the entire HIDDEN_LIBS set unavailable."""

    def test_imports_succeed_with_all_libs_hidden(self) -> None:
        result = _run_in_subprocess(
            """
            from bot.backtest import (
                BacktestConfig,
                BacktestEngine,
                BacktestResult,
                load_dukascopy_csv,
                generate_html_report,
                generate_equity_curve,
                generate_drawdown_chart,
                generate_trade_scatter,
                generate_r_multiple_histogram,
                generate_hourly_heatmap,
                generate_skip_reasons_pie,
            )
            print("OK")
            """
        )
        assert result.returncode == 0, (
            f"backtest import broke with hidden libs:\n"
            f"  HIDDEN: {HIDDEN_LIBS}\n"
            f"  STDERR: {result.stderr}"
        )
        assert "OK" in result.stdout

    def test_engine_runs_end_to_end_with_all_libs_hidden(self) -> None:
        """Smoke: actually run a backtest with everything hidden.

        If a future change makes the engine itself touch a hidden lib
        (e.g. by adding a Supabase write inside _try_open_setup), this
        test fails noisily.
        """
        result = _run_in_subprocess(
            """
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
            print(f"OK BARS={result.bars_processed}")
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK BARS=150" in result.stdout

    def test_html_report_renders_with_all_libs_hidden(self) -> None:
        """The reporter is the second-most-likely place a new dep
        sneaks in (Plotly / template imports)."""
        result = _run_in_subprocess(
            """
            import pandas as pd
            import tempfile, os
            from bot.backtest import (
                BacktestConfig, BacktestEngine, generate_html_report,
            )
            n = 150
            times = pd.date_range(
                "2026-01-01T00:00:00Z", periods=n, freq="5min", tz="UTC",
            )
            df = pd.DataFrame(
                {"open":[1900.0]*n, "high":[1900.5]*n, "low":[1899.5]*n,
                 "close":[1900.0]*n, "volume":[100]*n},
                index=times,
            )
            cfg = BacktestConfig(min_history_bars=100, progress_log_every_bars=0)
            result = BacktestEngine(cfg).run(df)
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                path = f.name
            try:
                generate_html_report(result, path, df=df)
                size = os.path.getsize(path)
                print(f"OK SIZE={size}")
            finally:
                os.unlink(path)
            """
        )
        assert result.returncode == 0, result.stderr
        assert "OK SIZE=" in result.stdout


# --------------------------------------------------------------------------- #
# Supabase-specific (mirrors test_mt5_lazy_import.py for MetaTrader5)
# --------------------------------------------------------------------------- #


class TestSupabaseLazyImport:
    def test_supabase_logger_module_imports_without_supabase(self) -> None:
        """``bot.logging.supabase_logger`` itself imports clean.

        Type aliases (``Setup``, ``Trade``, ``NewsEvent``, etc.) and
        the ``SupabaseLogger`` class should all be accessible — only
        instantiating SupabaseLogger triggers the supabase-py import.
        """
        result = _run_in_subprocess(
            """
            from bot.logging.supabase_logger import (
                Setup, Trade, NewsEvent, SupabaseLogger,
                ZoneInput, SetupInput, TradeInput,
            )
            assert SupabaseLogger
            assert Setup
            print("OK")
            """,
            hide=("supabase",),
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_constructing_supabase_logger_raises_clear_importerror(self) -> None:
        """``SupabaseLogger(url, key)`` without the package raises a clear error."""
        result = _run_in_subprocess(
            """
            from bot.logging.supabase_logger import SupabaseLogger
            try:
                SupabaseLogger("http://x", "key")
            except ImportError as e:
                msg = str(e)
                assert "supabase-py is required" in msg, msg
                assert "bot.backtest" in msg, (
                    "error should point users at the backtest framework"
                )
                print("OK")
            else:
                print("UNEXPECTED_NO_RAISE")
            """,
            hide=("supabase",),
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout


# --------------------------------------------------------------------------- #
# Sanity: HIDDEN_LIBS doesn't accidentally hide a backtest dep
# --------------------------------------------------------------------------- #


class TestHiddenLibsListIsCorrect:
    """Per-lib: hide ONLY that one library and import bot.backtest.

    If bot.backtest reaches the hidden lib, the ``import …`` statement
    triggers ``ModuleNotFoundError`` (because ``sys.modules[lib] = None``
    is a real sentinel — not the same as the lib being absent), the
    subprocess exits non-zero, and we get a precise per-lib failure
    message rather than the holistic "everything broke" signal.
    """

    @pytest.mark.parametrize("lib", HIDDEN_LIBS)
    def test_each_hidden_lib_is_not_imported_by_backtest(
        self, lib: str,
    ) -> None:
        result = _run_in_subprocess(
            "import bot.backtest\nprint('OK')",
            hide=(lib,),
        )
        assert result.returncode == 0, (
            f"bot.backtest reaches {lib!r} despite the lazy guard:\n"
            f"{result.stderr}"
        )
        assert "OK" in result.stdout
