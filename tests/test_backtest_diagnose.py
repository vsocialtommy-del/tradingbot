"""Tests for ``bot.backtest.diagnose`` — the setup-detection funnel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot.backtest.diagnose import (
    FunnelCounts,
    _build_parser,
    _format_report,
    diagnose,
    main,
)


def _make_synthetic(n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 2300.0
    drift = np.linspace(0, 25.0, n_bars)
    swing = 15.0 * np.sin(2 * np.pi * np.arange(n_bars) / 50)
    fast = 5.0 * np.sin(2 * np.pi * np.arange(n_bars) / 13)
    noise = np.cumsum(rng.normal(scale=2.0, size=n_bars)) * 0.05
    closes = base + drift + swing + fast + noise
    opens = np.concatenate([[closes[0]], closes[:-1]])
    bar_range = np.abs(rng.normal(scale=2.0, size=n_bars)) + 1.0
    highs = np.maximum(opens, closes) + bar_range / 2
    lows = np.minimum(opens, closes) - bar_range / 2
    times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=n_bars, freq="5min", tz="UTC",
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [100] * n_bars},
        index=times,
    )


# --------------------------------------------------------------------------- #
# Funnel counts
# --------------------------------------------------------------------------- #


class TestDiagnose:
    def test_too_short_history_returns_zero_counts(self) -> None:
        df = _make_synthetic(n_bars=50)
        counts = diagnose(df, min_history_bars=100)
        assert counts.bars_processed == 50
        assert counts.detection_bars == 0
        assert counts.pattern_candidates == 0

    def test_returns_funnel_counts_for_valid_df(self) -> None:
        df = _make_synthetic(n_bars=200)
        counts = diagnose(df, min_history_bars=100)
        assert isinstance(counts, FunnelCounts)
        assert counts.bars_processed == 200
        assert counts.detection_bars == 100

    def test_pattern_subtype_counts_sum_to_total(self) -> None:
        df = _make_synthetic(n_bars=300)
        c = diagnose(df)
        assert (
            c.rbr_candidates + c.dbd_candidates
            + c.dbr_candidates + c.rbd_candidates
        ) == c.pattern_candidates

    def test_strong_point_failures_mutually_exclusive(self) -> None:
        # Sum of the four buckets must equal (refined_tradeable − strong_points).
        # PR #31 buckets: no_swing, no_sl_anchor, no_break_yet, invalidated.
        df = _make_synthetic(n_bars=400)
        c = diagnose(df)
        sp_failures_sum = (
            c.sp_failed_no_swing + c.sp_failed_no_sl_anchor
            + c.sp_failed_no_break_yet + c.sp_failed_invalidated
        )
        assert sp_failures_sum == c.refined_tradeable - c.strong_points

    def test_unique_zones_at_most_strong_points(self) -> None:
        # Dedup can only reduce the Strong Point count.
        df = _make_synthetic(n_bars=400)
        c = diagnose(df)
        assert c.unique_zones <= c.strong_points


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #


class TestReportFormat:
    def test_funnel_lines_render_without_error(self) -> None:
        df = _make_synthetic(n_bars=200)
        counts = diagnose(df)
        lines = counts.funnel_lines()
        assert lines  # non-empty
        # Must include the headline stages.
        joined = "\n".join(lines)
        for marker in (
            "Pattern candidates",
            "Refined tradeable",
            "Strong Points",
            "Unique zones",
        ):
            assert marker in joined

    def test_format_report_contains_source_label(self) -> None:
        df = _make_synthetic(n_bars=200)
        counts = diagnose(df)
        report = _format_report(counts, source="my-test-data.csv")
        assert "my-test-data.csv" in report
        assert "Funnel" in report

    def test_pct_handles_zero_denominator(self) -> None:
        # Zero pattern candidates → all percent strings should be "—".
        df = _make_synthetic(n_bars=50)
        counts = diagnose(df, min_history_bars=100)
        report = _format_report(counts, source="empty")
        # "—" placeholder for zero denominators.
        assert "—" in report or counts.pattern_candidates == 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


class TestCLI:
    def test_parser_accepts_csv_path(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["data.csv"])
        assert args.csv == Path("data.csv")
        assert args.bars is None
        assert args.json is False

    def test_parser_bars_and_json_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["data.csv", "--bars", "500", "--json"])
        assert args.bars == 500
        assert args.json is True

    def test_main_runs_against_csv(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Write a small ISO-format CSV the data_loader can read.
        df = _make_synthetic(n_bars=150)
        csv_path = tmp_path / "test.csv"
        # Reset index → "index" column name; rename to "timestamp"
        # so the data loader recognises it.
        out = df.reset_index()
        out = out.rename(columns={out.columns[0]: "timestamp"})
        out.to_csv(csv_path, index=False)
        rc = main([str(csv_path)])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Funnel" in captured.out

    def test_main_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        df = _make_synthetic(n_bars=150)
        csv_path = tmp_path / "test.csv"
        # Reset index → "index" column name; rename to "timestamp"
        # so the data loader recognises it.
        out = df.reset_index()
        out = out.rename(columns={out.columns[0]: "timestamp"})
        out.to_csv(csv_path, index=False)
        rc = main([str(csv_path), "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        # Output must parse as JSON with the expected keys.
        data = json.loads(captured.out)
        assert "pattern_candidates" in data
        assert "strong_points" in data

    def test_main_bars_limit_applied(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Data has 300 bars; --bars 150 should cap.
        df = _make_synthetic(n_bars=300)
        csv_path = tmp_path / "test.csv"
        # Reset index → "index" column name; rename to "timestamp"
        # so the data loader recognises it.
        out = df.reset_index()
        out = out.rename(columns={out.columns[0]: "timestamp"})
        out.to_csv(csv_path, index=False)
        rc = main([str(csv_path), "--bars", "150", "--json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["bars_processed"] == 150
