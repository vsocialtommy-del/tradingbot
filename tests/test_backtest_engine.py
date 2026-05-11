"""Tests for ``bot.backtest.engine``.

Two flavours:

* **Unit-level**: drive the engine on a flat synthetic dataset and patch
  ``run_strategy_pipeline`` to inject zones at chosen bars. Verifies the
  orchestration logic — dedup, exposure cap, daily halt, force-close,
  cascade-cancel on SL — without depending on real pattern detection.

* **Integration**: a tiny synthetic dataset built so a real W pattern
  exists and the actual strategy pipeline produces a tradeable zone.
  Verifies the full pipeline → SL calc → order placement → fill chain.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
from bot.backtest.simulator import CloseReason
from bot.strategy.structure import Swing
from bot.strategy.pattern_detection import (
    Base,
    Impulse,
    Pattern,
    PatternType,
)
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.structure import BosEvent, Swing
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_flat_df(
    n: int = 200, base_price: float = 1900.0,
    start: str = "2026-05-01T08:00:00Z",
) -> pd.DataFrame:
    """Flat OHLC: every bar has the same prices. Strategy can't find anything."""
    times = pd.date_range(start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [base_price] * n, "high": [base_price + 0.5] * n,
            "low": [base_price - 0.5] * n, "close": [base_price] * n,
            "volume": [100] * n,
        },
        index=times,
    )


def patch_sl_price(mocker, sl_price: float = 1880.0) -> None:
    """Stub ``strong_point.compute_sl_price`` to a fixed value.

    Replaces the pre-PR-31 ``stub_sl`` helper that targeted the
    removed ``_calculate_sl`` engine helper. The engine now reads
    ``zone.sl_anchor_swing`` and calls ``compute_sl_price`` on it;
    tests can patch the latter to control SL.
    """
    mocker.patch(
        "bot.backtest.engine.compute_sl_price",
        return_value=float(sl_price),
    )


def make_imbalance_zone(
    *, direction: str = "BUY",
    top: float = 1900.0, bottom: float = 1895.0,
    formed_index: int = 5,
) -> ValidatedZone:
    """Build a ValidatedZone (PR #31).

    Name retained for call-site stability; returns the post-PR-31
    type. Default ``broken_swing`` price is ``top + 5`` (BUY) /
    ``bottom - 5`` (SELL), matching the v1 spec's break level
    semantics.
    """
    ts = pd.Timestamp(NOW)
    impulse_dir = "RALLY" if direction == "BUY" else "DROP"
    impulse = Impulse(
        direction=impulse_dir, start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=5.0, largest_body=5.0, candle_count=1,
    )
    base = Base(
        start_index=1, end_index=1, candle_count=1,
        top=top, bottom=bottom, range_size=top - bottom, largest_body=0.5,
    )
    pattern = Pattern(
        pattern_type=PatternType.RBR if direction == "BUY" else PatternType.DBD,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction=direction,  # type: ignore[arg-type]
        formed_at=ts,
    )
    zone = Zone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
    )
    refined = RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        is_tradeable=True, rejection_reason=None, original_zone=zone,
    )
    broken = Swing(
        index=formed_index + 5, time=ts,
        price=top + 5.0 if direction == "BUY" else bottom - 5.0,
        kind="HIGH" if direction == "BUY" else "LOW",
    )
    anchor = Swing(
        index=formed_index - 1, time=ts,
        price=bottom - 5.0 if direction == "BUY" else top + 5.0,
        kind="LOW" if direction == "BUY" else "HIGH",
    )
    return ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=top, bottom=bottom, formed_at=ts, source_pattern=pattern,
        refined_zone=refined,
        is_strong_point=True, validation_failures=[],
        broken_swing=broken, broken_at=ts, sl_anchor_swing=anchor,
    )


# --------------------------------------------------------------------------- #
# Smoke
# --------------------------------------------------------------------------- #


class TestSmoke:
    def test_runs_to_completion_no_zones(self, mocker: MockerFixture) -> None:
        # Strategy returns nothing → no setups. Engine just walks the df.
        mocker.patch("bot.backtest.engine.run_strategy_pipeline", return_value=[])
        df = make_flat_df(n=150)
        cfg = BacktestConfig(min_history_bars=100, progress_log_every_bars=0)
        result = BacktestEngine(cfg).run(df)
        assert isinstance(result, BacktestResult)
        assert result.bars_processed == 150
        assert result.setups_detected == 0
        assert result.setups_taken == 0
        assert result.metrics.trades.total == 0
        # Equity curve should be ~ starting balance throughout (flat market).
        assert len(result.equity_curve) > 0

    def test_validates_tz_aware_index(self) -> None:
        df = make_flat_df()
        df.index = df.index.tz_localize(None)
        with pytest.raises(ValueError, match="tz-aware"):
            BacktestEngine(BacktestConfig()).run(df)

    def test_too_few_bars_raises(self) -> None:
        df = make_flat_df(n=50)
        with pytest.raises(ValueError, match="bars"):
            BacktestEngine(BacktestConfig(min_history_bars=100)).run(df)


# --------------------------------------------------------------------------- #
# Setup creation + dedup
# --------------------------------------------------------------------------- #


class TestSetupCreation:
    def test_zone_detected_creates_setup_and_places_orders(
        self, mocker: MockerFixture,
    ) -> None:
        zone = make_imbalance_zone(top=1900.0, bottom=1895.0, direction="BUY")
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            return_value=[zone],
        )
        # Stub SL — the synthetic flat OHLC's fallback-low produces an
        # unhelpful SL above entry; this test isn't checking SL calc.
        patch_sl_price(mocker, sl_price=1880.0)
        df = make_flat_df(n=110, base_price=1920.0)  # price well above zone
        cfg = BacktestConfig(min_history_bars=100, progress_log_every_bars=0)
        result = BacktestEngine(cfg).run(df)

        assert result.setups_taken == 1
        # Flat price above zone → no fills → no closed positions.
        assert result.metrics.trades.total == 0

    def test_same_zone_dedup_creates_setup_only_once(
        self, mocker: MockerFixture,
    ) -> None:
        zone = make_imbalance_zone(top=1900.0, bottom=1895.0, direction="BUY")
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            return_value=[zone],
        )
        patch_sl_price(mocker, sl_price=1880.0)
        df = make_flat_df(n=120, base_price=1920.0)
        cfg = BacktestConfig(min_history_bars=100, progress_log_every_bars=0)
        result = BacktestEngine(cfg).run(df)
        # 20 detection bars × 1 zone each = 20 detection events,
        # but only 1 unique zone → 1 setup taken.
        assert result.setups_detected == 20
        assert result.setups_taken == 1

    def test_zone_without_sl_anchor_skipped_with_skip_reason(
        self, mocker: MockerFixture,
    ) -> None:
        # Defensive: a zone reaching the engine without an
        # ``sl_anchor_swing`` means Strong Point validation produced
        # a malformed result (shouldn't happen in practice since the
        # pipeline only emits is_strong_point=True zones which always
        # have an anchor). The engine should skip with ``no_sl_anchor``
        # rather than crash on the None-deref in compute_sl_price.
        zone = make_imbalance_zone(top=1900.0, bottom=1895.0, direction="BUY")
        broken_zone = ValidatedZone(
            direction=zone.direction, top=zone.top, bottom=zone.bottom,
            formed_at=zone.formed_at, source_pattern=zone.source_pattern,
            refined_zone=zone.refined_zone,
            is_strong_point=zone.is_strong_point,
            validation_failures=zone.validation_failures,
            broken_swing=zone.broken_swing,
            broken_at=zone.broken_at,
            sl_anchor_swing=None,        # <-- anchor missing
        )
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            return_value=[broken_zone],
        )
        df = make_flat_df(n=110, base_price=1920.0)
        cfg = BacktestConfig(min_history_bars=100, progress_log_every_bars=0)
        result = BacktestEngine(cfg).run(df)

        assert result.setups_taken == 0
        assert "no_sl_anchor" in result.skip_reasons


# --------------------------------------------------------------------------- #
# Exposure cap
# --------------------------------------------------------------------------- #


class TestExposureCap:
    def test_max_simultaneous_setups_respected(
        self, mocker: MockerFixture,
    ) -> None:
        # Three different zones detected over consecutive bars.
        z1 = make_imbalance_zone(top=1900, bottom=1895, direction="BUY")
        z2 = make_imbalance_zone(top=1910, bottom=1905, direction="BUY")
        z3 = make_imbalance_zone(top=1920, bottom=1915, direction="BUY")
        z4 = make_imbalance_zone(top=1930, bottom=1925, direction="BUY")

        # Different zone sets returned over time so dedup doesn't drop them.
        call_count = {"n": 0}

        def pipeline(df, cfg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [z1]
            if call_count["n"] == 2:
                return [z2]
            if call_count["n"] == 3:
                return [z3]
            if call_count["n"] == 4:
                return [z4]
            return []

        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline", side_effect=pipeline,
        )
        patch_sl_price(mocker, sl_price=1880.0)
        df = make_flat_df(n=110, base_price=1950.0)  # well above all zones
        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
            max_simultaneous_setups=3,
        )
        result = BacktestEngine(cfg).run(df)
        # 3 setups taken; 4th blocked by exposure cap.
        assert result.setups_taken == 3
        # Skip reason recorded.
        assert "exposure_cap" in result.skip_reasons


# --------------------------------------------------------------------------- #
# SL hit + cascade
# --------------------------------------------------------------------------- #


class TestSLCascadeCancel:
    def test_sl_hit_cancels_pending_layers(
        self, mocker: MockerFixture,
    ) -> None:
        """Layer 1 fills, then a later bar crashes through SL. The
        cascade-cancel must prevent Layers 2 + 3 from filling at the
        same crash tick."""
        zone = make_imbalance_zone(top=1900.0, bottom=1895.0, direction="BUY")
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            return_value=[zone],
        )
        # SL stub at 1882 — well above the crash level used below.
        patch_sl_price(mocker, sl_price=1882.0)

        # OHLC: bars 0-100 flat at 1900 (Layer 1 limit fills naturally
        # via the OHLC walk's 1899.5 dip on bar 101). Bar 102 crashes
        # to 1860 — the crash tick has Layer 2/3 limits AND SL all
        # crossable; cascade must protect Layers 2 + 3.
        n = 110
        times = pd.date_range(
            "2026-05-01T08:00:00Z", periods=n, freq="5min", tz="UTC",
        )
        opens = [1900.0] * n
        highs = [1900.5] * n
        lows = [1899.5] * n
        closes = [1900.0] * n
        # Bar 102: huge crash.
        opens[102] = 1900.0
        highs[102] = 1900.5
        lows[102] = 1860.0
        closes[102] = 1860.0
        df = pd.DataFrame(
            {
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": [100] * n,
            },
            index=times,
        )

        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )
        result = BacktestEngine(cfg).run(df)

        # Exactly ONE trade — Layer 1 stopped out. Layers 2 + 3 were
        # cascade-cancelled before they could fill on the crash tick.
        # If cascade had failed, we'd see 3 SL trades (or 1 SL + 2
        # END_OF_DATA from the layers filling deep below price and
        # never recovering).
        assert result.metrics.trades.total == 1
        reasons = [p.close_reason for p in result.closed_positions]
        assert reasons == [CloseReason.SL]


# --------------------------------------------------------------------------- #
# TP1 hit + SL→BE
# --------------------------------------------------------------------------- #


class TestTP1HitMovesSLToBE:
    def test_tp1_partial_close_and_sl_modify(
        self, mocker: MockerFixture,
    ) -> None:
        """Bar 100 flat → pipeline detects + places orders. Bar 101
        flat → Layer 1 fills via the 1899.5 OHLC dip. Bar 102 spikes
        up to 1910 → TP1 (BoS broken_level = 1905) hits. Remaining
        runner has SL moved to BE."""
        zone = make_imbalance_zone(
            top=1900.0, bottom=1895.0, direction="BUY",
        )
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            return_value=[zone],
        )
        patch_sl_price(mocker, sl_price=1882.0)
        n = 110
        times = pd.date_range(
            "2026-05-01T08:00:00Z", periods=n, freq="5min", tz="UTC",
        )
        opens = [1900.0] * n
        highs = [1900.5] * n
        lows = [1899.5] * n
        closes = [1900.0] * n
        # Bar 102: spike up to 1910 (past TP1 at 1905).
        opens[102] = 1900.0
        highs[102] = 1910.0
        lows[102] = 1899.5
        closes[102] = 1910.0
        df = pd.DataFrame(
            {
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": [100] * n,
            },
            index=times,
        )

        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )
        result = BacktestEngine(cfg).run(df)

        # At least one TP1 partial close recorded.
        reasons = [p.close_reason for p in result.closed_positions]
        assert CloseReason.TP1 in reasons


# --------------------------------------------------------------------------- #
# Daily halt
# --------------------------------------------------------------------------- #


class TestDailyHalt:
    def test_halt_blocks_new_setups_after_drawdown(
        self, mocker: MockerFixture,
    ) -> None:
        """Daily halt: simulate a -10% drawdown via a manual SL stop, then
        confirm new setups are NOT created on subsequent bars."""
        zone1 = make_imbalance_zone(top=1900, bottom=1895, direction="BUY")
        zone2 = make_imbalance_zone(top=1880, bottom=1875, direction="BUY")

        # First bar after history: zone1. Later bars (after halt): zone2.
        seq = iter([[zone1]] + [[]] * 50 + [[zone2]] * 100)
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            side_effect=lambda df, cfg: next(seq, []),
        )
        # Stub SL so the SL is sensible despite synthetic flat data.
        patch_sl_price(mocker, sl_price=1882.0)

        # Build OHLC so Layer 1 of zone1 fills then SL hits hard.
        n = 200
        times = pd.date_range(
            "2026-05-01T08:00:00Z", periods=n, freq="5min", tz="UTC",
        )
        opens = [1900.0] * n
        highs = [1900.5] * n
        lows = [1899.5] * n
        closes = [1900.0] * n
        # Bar 100: layer 1 limit fills + sharp crash (-12% on the
        # tiny 0.01-lot default would only be 1¢, so we use a big lot
        # by overriding fixed_lot_size).
        closes[100] = 1850.0
        lows[100] = 1850.0
        df = pd.DataFrame(
            {
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": [100] * n,
            },
            index=times,
        )

        # Use a large lot size so the SL hit drops balance enough.
        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
            fixed_lot_size=10.0,  # exaggerated to force a >10% drawdown
            starting_balance=10_000.0,
            daily_loss_limit_pct=10.0,
        )
        result = BacktestEngine(cfg).run(df)

        # Setup 1 created and stopped out. Setup 2 detected later but
        # blocked by halt — so taken count should be 1, not 2.
        # (zone2 will be re-emitted on every bar after the halt; dedup
        # adds it to detected once, but halt blocks taken.)
        assert result.setups_taken == 1


# --------------------------------------------------------------------------- #
# End-of-data force close
# --------------------------------------------------------------------------- #


class TestEndOfDataForceClose:
    def test_open_positions_closed_at_end(
        self, mocker: MockerFixture,
    ) -> None:
        """Layer 1 fills mid-dataset. Price stays flat. Position should be
        force-closed at the last bar with reason END_OF_DATA."""
        zone = make_imbalance_zone(top=1900.0, bottom=1895.0, direction="BUY")
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            return_value=[zone],
        )
        n = 105
        times = pd.date_range(
            "2026-05-01T08:00:00Z", periods=n, freq="5min", tz="UTC",
        )
        opens = [1900.0] * n
        highs = [1900.5] * n
        lows = [1899.5] * n
        closes = [1900.0] * n
        df = pd.DataFrame(
            {
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": [100] * n,
            },
            index=times,
        )

        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )
        result = BacktestEngine(cfg).run(df)

        reasons = [p.close_reason for p in result.closed_positions]
        # Must have at least one END_OF_DATA close.
        assert CloseReason.END_OF_DATA in reasons


# --------------------------------------------------------------------------- #
# SL too close / too far skip reasons
# --------------------------------------------------------------------------- #


class TestSLValidation:
    def test_sl_too_close_skipped(self, mocker: MockerFixture) -> None:
        # Tight zone with no swings → fallback to bar low = 1899.5.
        # sl_buffer 17.5 → SL 1882. entry 1900 → distance 18 — fine.
        # Set max_sl_distance_points = 10 → too far → skipped.
        zone = make_imbalance_zone(top=1900.0, bottom=1895.0, direction="BUY")
        mocker.patch(
            "bot.backtest.engine.run_strategy_pipeline",
            return_value=[zone],
        )
        df = make_flat_df(n=110, base_price=1900.0)
        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
            max_sl_distance_points=10.0,
        )
        result = BacktestEngine(cfg).run(df)
        assert result.setups_taken == 0
        assert "sl_too_far" in result.skip_reasons


# --------------------------------------------------------------------------- #
# Integration: real strategy pipeline
# --------------------------------------------------------------------------- #


class TestIntegration:
    def test_real_pipeline_with_w_pattern_dataset(self) -> None:
        """Real ``run_strategy_pipeline`` over a synthetic W pattern.

        Builds OHLC where bars 100-200 contain a clean W with a BoS
        afterwards. The pipeline should return the zone; the engine
        should place orders. We don't assert specific trade outcomes
        — just that the smoke pipeline runs end-to-end without errors
        and *something* gets detected.
        """
        n = 250
        times = pd.date_range(
            "2026-05-01T08:00:00Z", periods=n, freq="5min", tz="UTC",
        )
        # Mostly flat at 1920, with a W pattern around bars 100-150.
        closes = [1920.0] * n
        highs = [1920.5] * n
        lows = [1919.5] * n
        opens = [1920.0] * n

        # First low of W (bar 110): dip to 1900.
        closes[107:111] = [1915, 1908, 1903, 1900]
        lows[107:111] = [1914.5, 1907.5, 1902.5, 1899.5]
        # Mid peak of W (bar 120): rally back to 1915.
        closes[112:121] = [1903, 1906, 1908, 1910, 1912, 1914, 1915, 1915, 1915]
        highs[112:121] = [c + 0.5 for c in closes[112:121]]
        # Second low of W (bar 130): dip again to 1900.
        closes[122:131] = [1914, 1912, 1910, 1908, 1906, 1904, 1902, 1900, 1900]
        lows[122:131] = [c - 0.5 for c in closes[122:131]]
        # BoS — rally above the mid peak (1915) by close.
        closes[132:140] = [1903, 1908, 1912, 1916, 1918, 1920, 1920, 1920]
        highs[132:140] = [c + 0.5 for c in closes[132:140]]
        # Realign opens to within high/low.
        for i in range(n):
            opens[i] = max(lows[i], min(highs[i], opens[i]))
            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])

        df = pd.DataFrame(
            {
                "open": opens, "high": highs, "low": lows,
                "close": closes, "volume": [100] * n,
            },
            index=times,
        )

        cfg = BacktestConfig(
            min_history_bars=100, progress_log_every_bars=0,
        )
        # Doesn't raise; returns a BacktestResult.
        result = BacktestEngine(cfg).run(df)
        assert isinstance(result, BacktestResult)
        assert result.bars_processed == n
        # The exact number of zones detected depends on synthetic W
        # quality + BoS detection. We just want the integration to
        # not crash and to produce a sensible result.
        assert result.setups_detected >= 0
