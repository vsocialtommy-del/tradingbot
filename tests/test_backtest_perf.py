"""Perf regression tests for the backtest engine.

Runs a representative-sized synthetic backtest with all loguru output
muted and asserts a wall-clock budget. The budget is generous (20s
for 500 bars) so this is a regression alarm, not a benchmark — it
catches an *order-of-magnitude* slowdown like the one fixed in this
PR (4.9s → 2.3s on the dev box; was unbounded in Colab before the
swing-reuse fix), not a 10% drift.

Two flavours of guarantee:

1. **Wall-clock budget** for a realistic backtest size.
2. **Single ``detect_swings`` call per pipeline iteration** — locks in
   the swing-reuse optimisation. If a future change re-introduces a
   per-detector ``detect_swings`` call, this test fails immediately
   and points at the offending stage.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from bot.backtest import BacktestConfig, BacktestEngine


# Generous budget so flaky CI hosts don't fail; we're only catching
# order-of-magnitude regressions. Locally typical run is ~2-3s.
PERF_BUDGET_SECONDS_500_BARS = 20.0


def _make_synthetic(n_bars: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 2300.0
    drift = np.linspace(0, 25.0, n_bars)
    swing = 15.0 * np.sin(2 * np.pi * np.arange(n_bars) / 50)
    fast = 5.0 * np.sin(2 * np.pi * np.arange(n_bars) / 13)
    noise = np.cumsum(rng.normal(scale=2.0, size=n_bars)) * 0.05
    closes = base + drift + swing + fast + noise
    opens = np.concatenate([[closes[0]], closes[:-1]])
    opens += rng.normal(scale=0.3, size=n_bars)
    bar_range = np.abs(rng.normal(scale=2.0, size=n_bars)) + 1.0
    highs = np.maximum(opens, closes) + bar_range / 2
    lows = np.minimum(opens, closes) - bar_range / 2
    times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=n_bars, freq="5min", tz="UTC",
    )
    return pd.DataFrame(
        {
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": [100] * n_bars,
        },
        index=times,
    )


@pytest.fixture(autouse=True)
def _quiet_loguru():
    """Mute loguru for the duration of the test — the engine emits
    progress logs and DEBUG strategy logs which dominate runtime if
    captured by pytest's stderr handler. We want to measure engine
    work, not log throughput."""
    logger.remove()
    yield
    logger.remove()


# --------------------------------------------------------------------------- #
# Wall-clock budget
# --------------------------------------------------------------------------- #


class TestPerfBudget:
    def test_500_bar_backtest_under_budget(self) -> None:
        df = _make_synthetic(500)
        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )
        t0 = time.perf_counter()
        BacktestEngine(cfg).run(df)
        elapsed = time.perf_counter() - t0
        assert elapsed < PERF_BUDGET_SECONDS_500_BARS, (
            f"500-bar backtest took {elapsed:.2f}s, budget is "
            f"{PERF_BUDGET_SECONDS_500_BARS}s. The strategy pipeline "
            f"is the usual culprit — re-profile with cProfile and look "
            f"at detect_swings call counts."
        )


# --------------------------------------------------------------------------- #
# Lock the swing-reuse optimisation
# --------------------------------------------------------------------------- #


class TestSwingReuse:
    def test_pipeline_calls_detect_swings_once_per_iteration(self) -> None:
        """Pipeline must compute swings ONCE per call.

        Previously detect_w_patterns and detect_m_patterns each made
        their own detect_swings call, on top of the one already done
        by analyze_structure — three full O(n) scans per pipeline
        iteration. The fix: pipeline pre-computes swings via
        analyze_structure and passes them to both detectors.
        """
        df = _make_synthetic(150)
        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )

        # Patch detect_swings at every import site so we count total
        # call invocations across the whole strategy stack.
        from bot.strategy import structure as structure_mod
        from bot.strategy import pattern_detection as pattern_mod

        with patch.object(
            structure_mod, "detect_swings",
            wraps=structure_mod.detect_swings,
        ) as struct_spy, patch.object(
            pattern_mod, "detect_swings",
            wraps=pattern_mod.detect_swings,
        ) as pattern_spy:
            BacktestEngine(cfg).run(df)

            # 50 detection bars (150 - 100 min_history). Each pipeline
            # iteration must call detect_swings AT MOST ONCE — through
            # analyze_structure (in the structure module). The pattern
            # module's detect_swings reference must NEVER fire because
            # pipeline pre-computes and passes swings.
            #
            # ``struct_spy`` covers the call from analyze_structure;
            # ``pattern_spy`` covers any (now-eliminated) call from
            # detect_w_patterns / detect_m_patterns.
            n_bars_processed = 50
            assert struct_spy.call_count == n_bars_processed, (
                f"expected {n_bars_processed} detect_swings calls via "
                f"analyze_structure, got {struct_spy.call_count}"
            )
            assert pattern_spy.call_count == 0, (
                f"detect_w_patterns / detect_m_patterns called "
                f"detect_swings {pattern_spy.call_count} times; "
                f"pipeline should pass swings instead. The "
                f"swing-reuse optimisation has regressed."
            )