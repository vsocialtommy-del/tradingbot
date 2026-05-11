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

    def test_per_bar_runtime_scales_linearly_not_quadratically(self) -> None:
        """Catch O(n²) regressions like the one fixed by the
        ``pipeline_window_bars`` slice.

        With a fixed pipeline window, doubling bar count should at
        most double total runtime. The old (no-window) implementation
        was O(n²) so 2000 bars ran ~4× slower than 500. We assert the
        ratio stays under 6× (generous so flaky CI hosts pass).
        """
        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )

        def time_run(n_bars: int) -> float:
            df = _make_synthetic(n_bars)
            t0 = time.perf_counter()
            BacktestEngine(cfg).run(df)
            return time.perf_counter() - t0

        # Warm-up — JIT-y caches, import overhead, etc.
        time_run(500)

        small = time_run(500)
        big = time_run(2000)
        # 2000 / 500 = 4× more bars. With linear scaling the ratio is
        # ~4×; with O(n²) it'd be ~16×. Budget of 6× catches the
        # quadratic regression while leaving CI headroom.
        ratio = big / small if small > 0 else float("inf")
        assert ratio < 6.0, (
            f"backtest runtime scaling regressed: 500 bars = {small:.2f}s, "
            f"2000 bars = {big:.2f}s (ratio {ratio:.1f}×). Expected linear "
            f"scaling (~4×); a much larger ratio means the pipeline is "
            f"doing O(n²) work — most likely the engine stopped passing "
            f"a fixed-size window to ``run_strategy_pipeline``."
        )


# --------------------------------------------------------------------------- #
# Lock the swing-reuse optimisation
# --------------------------------------------------------------------------- #


class TestAnalyzeStructureCalledOncePerIteration:
    """analyze_structure should fire exactly once per pipeline call.

    Replaces PR #26's ``TestSwingReuse`` which locked in a property
    of the old W/M pattern detector (where detect_swings was called
    3× per iteration). The S&D pipeline (PR #31) doesn't call
    detect_swings from pattern_detection at all — structure analysis
    is the single source of swings — so the equivalent property is:
    analyze_structure is invoked exactly once per detection bar.
    """

    def test_analyze_structure_called_once_per_detection_bar(self) -> None:
        df = _make_synthetic(150)
        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )
        # Patch the pipeline's bound reference (the pipeline imports
        # analyze_structure into its namespace at module load).
        import bot.strategy.pipeline as pipeline_mod

        with patch.object(
            pipeline_mod, "analyze_structure",
            wraps=pipeline_mod.analyze_structure,
        ) as spy:
            BacktestEngine(cfg).run(df)
            # 50 detection bars (150 - 100 min_history). Pipeline runs
            # once per detection bar; each call invokes analyze_structure
            # exactly once.
            assert spy.call_count == 50, (
                f"expected 50 analyze_structure calls (one per detection "
                f"bar); got {spy.call_count}. If higher, the pipeline is "
                f"doing redundant structure analysis."
            )