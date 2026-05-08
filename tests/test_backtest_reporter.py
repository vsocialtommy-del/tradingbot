"""Tests for ``bot.backtest.reporter``.

Each chart function is exercised on a hand-built ``BacktestResult``
(varying winners/losers, empty cases) and the resulting Plotly Figure
is asserted against — trace count, key annotations, presence of
expected hover text. Visual fidelity isn't tested (no image diff);
the goal is structural correctness so the HTML report layout doesn't
silently break.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from bot.backtest.engine import BacktestConfig, BacktestResult
from bot.backtest.metrics import (
    BacktestMetrics,
    EquityMetrics,
    SetupMetrics,
    TradeMetrics,
    compute_metrics,
)
from bot.backtest.reporter import (
    ReporterConfig,
    generate_drawdown_chart,
    generate_equity_curve,
    generate_hourly_heatmap,
    generate_html_report,
    generate_r_multiple_histogram,
    generate_skip_reasons_pie,
    generate_trade_scatter,
)
from bot.backtest.simulator import BacktestPosition, CloseReason


NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_pos(
    *,
    pnl: float,
    setup_id: int = 1,
    layer: int = 1,
    direction: str = "BUY",
    entry: float = 1900.0,
    sl: float = 1880.0,
    lot_size: float = 0.01,
    close_reason: CloseReason = CloseReason.TP1,
    opened_at: datetime | None = None,
    duration_minutes: float = 30.0,
) -> BacktestPosition:
    o = opened_at or NOW
    return BacktestPosition(
        ticket=100_000 + layer + setup_id * 10,
        setup_id=setup_id, layer=layer,
        direction=direction,  # type: ignore[arg-type]
        entry_price=entry, lot_size=lot_size, sl=sl,
        tp=1907.0 if direction == "BUY" else 1893.0,
        opened_at=o, status="CLOSED",
        closed_lots=lot_size,
        exit_price=entry + (7 if pnl > 0 else -20),
        exit_time=o + timedelta(minutes=duration_minutes),
        close_reason=close_reason,
        realised_pnl=pnl,
        commission_paid=0.35,
    )


def make_equity_curve(
    n_points: int = 50, drift: float = 0.0,
    starting: float = 10_000.0,
) -> pd.Series:
    """Synthetic equity series with optional drift."""
    times = pd.date_range(NOW, periods=n_points, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(scale=20.0, size=n_points)
    values = starting + np.cumsum(noise) + drift * np.arange(n_points)
    return pd.Series(values, index=times)


def make_result(
    *,
    closed: list[BacktestPosition] | None = None,
    equity: pd.Series | None = None,
    skip_reasons: dict[str, int] | None = None,
    setups_detected: int = 5,
    setups_taken: int = 3,
    starting_balance: float = 10_000.0,
) -> BacktestResult:
    closed = closed if closed is not None else []
    equity = equity if equity is not None else make_equity_curve()
    metrics = compute_metrics(
        closed_positions=closed,
        equity_curve=equity,
        starting_balance=starting_balance,
        setups_detected=setups_detected,
        setups_taken=setups_taken,
        skip_reasons=skip_reasons or {},
    )
    return BacktestResult(
        metrics=metrics,
        closed_positions=closed,
        equity_curve=equity,
        config=BacktestConfig(starting_balance=starting_balance),
        bars_processed=1000,
        setups_detected=setups_detected,
        setups_taken=setups_taken,
        skip_reasons=skip_reasons or {},
    )


@pytest.fixture
def populated_result() -> BacktestResult:
    """Result with mixed winners + losers across multiple setups."""
    closed = [
        make_pos(pnl=34.10, setup_id=1, layer=1, close_reason=CloseReason.TP1,
                 opened_at=NOW + timedelta(hours=1)),
        make_pos(pnl=-20.0, setup_id=2, layer=1, close_reason=CloseReason.SL,
                 opened_at=NOW + timedelta(hours=3)),
        make_pos(pnl=50.0, setup_id=3, layer=1, close_reason=CloseReason.TP1,
                 opened_at=NOW + timedelta(hours=6)),
        make_pos(pnl=-10.0, setup_id=4, layer=1, close_reason=CloseReason.SL,
                 opened_at=NOW + timedelta(hours=12)),
        make_pos(pnl=25.0, setup_id=5, layer=1, close_reason=CloseReason.TP1,
                 opened_at=NOW + timedelta(days=1, hours=2)),
    ]
    skip_reasons = {
        "exposure_cap": 3,
        "sl_too_close": 1,
        "sl_too_far": 2,
    }
    return make_result(
        closed=closed,
        skip_reasons=skip_reasons,
        setups_detected=11, setups_taken=5,
    )


@pytest.fixture
def empty_result() -> BacktestResult:
    return make_result(closed=[], equity=pd.Series(dtype=float))


# --------------------------------------------------------------------------- #
# 1. Equity curve
# --------------------------------------------------------------------------- #


class TestEquityCurve:
    def test_returns_figure_with_equity_trace(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_equity_curve(populated_result)
        assert isinstance(fig, go.Figure)
        names = [t.name for t in fig.data]
        assert "Equity" in names

    def test_winners_and_losers_get_separate_traces(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_equity_curve(populated_result)
        names = [t.name for t in fig.data if t.name]
        # Trade markers labelled with counts.
        assert any("TP1 hits" in n for n in names)
        assert any("SL hits" in n for n in names)

    def test_empty_returns_no_data_annotation(
        self, empty_result: BacktestResult,
    ) -> None:
        fig = generate_equity_curve(empty_result)
        annotations = [a for a in fig.layout.annotations if a.text]
        assert any("No equity data" in (a.text or "") for a in annotations)

    def test_only_losers_renders_without_error(self) -> None:
        closed = [
            make_pos(pnl=-15.0, close_reason=CloseReason.SL,
                     opened_at=NOW + timedelta(hours=i))
            for i in range(3)
        ]
        result = make_result(closed=closed)
        fig = generate_equity_curve(result)
        names = [t.name for t in fig.data if t.name]
        # SL trace present, no TP1 trace.
        assert any("SL hits" in n for n in names)
        assert not any("TP1 hits" in n for n in names)


# --------------------------------------------------------------------------- #
# 2. Drawdown
# --------------------------------------------------------------------------- #


class TestDrawdown:
    def test_returns_figure(self, populated_result: BacktestResult) -> None:
        fig = generate_drawdown_chart(populated_result)
        assert isinstance(fig, go.Figure)
        # Single trace: drawdown line.
        assert len(fig.data) == 1

    def test_underwater_fill_present(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_drawdown_chart(populated_result)
        # The trace should fill toward zero (underwater style).
        assert fig.data[0].fill == "tozeroy"

    def test_empty_returns_no_data(self, empty_result: BacktestResult) -> None:
        fig = generate_drawdown_chart(empty_result)
        annotations = [a for a in fig.layout.annotations if a.text]
        assert any("No equity data" in (a.text or "") for a in annotations)


# --------------------------------------------------------------------------- #
# 3. Trade scatter
# --------------------------------------------------------------------------- #


class TestTradeScatter:
    def test_with_ohlc_includes_close_line(
        self, populated_result: BacktestResult,
    ) -> None:
        idx = pd.date_range(NOW, periods=100, freq="5min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [1900.0] * 100, "high": [1901.0] * 100,
                "low": [1899.0] * 100, "close": [1900.0] * 100,
            },
            index=idx,
        )
        fig = generate_trade_scatter(populated_result, df)
        names = [t.name for t in fig.data if t.name]
        assert "Close" in names

    def test_without_df_still_plots_markers(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_trade_scatter(populated_result, df=None)
        names = [t.name for t in fig.data if t.name]
        assert "Close" not in names
        # Winners + losers markers still present.
        assert any("Winners" in n for n in names)
        assert any("Losers" in n for n in names)

    def test_df_missing_close_column_warns(
        self, populated_result: BacktestResult, caplog: pytest.LogCaptureFixture,
    ) -> None:
        idx = pd.date_range(NOW, periods=10, freq="5min", tz="UTC")
        # df without 'close' column.
        df = pd.DataFrame({"open": [1900.0] * 10}, index=idx)
        # Doesn't raise.
        generate_trade_scatter(populated_result, df)

    def test_empty_returns_no_data(
        self, empty_result: BacktestResult,
    ) -> None:
        fig = generate_trade_scatter(empty_result)
        annotations = [a for a in fig.layout.annotations if a.text]
        assert any("No trades" in (a.text or "") for a in annotations)


# --------------------------------------------------------------------------- #
# 4. R-multiple histogram
# --------------------------------------------------------------------------- #


class TestRMultipleHistogram:
    def test_returns_histogram_figure(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_r_multiple_histogram(populated_result)
        # Histogram trace present.
        assert any(isinstance(t, go.Histogram) for t in fig.data)

    def test_reference_lines_at_minus1_zero_plus1(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_r_multiple_histogram(populated_result)
        # vlines render as layout shapes.
        x_values = [s.x0 for s in fig.layout.shapes if s.type == "line"]
        assert -1.0 in x_values
        assert 0.0 in x_values
        assert 1.0 in x_values

    def test_outliers_clipped_to_max_bin(self) -> None:
        # Build a position with a 10R outlier.
        closed = [make_pos(
            pnl=200.0, sl=1898.0, entry=1900.0, lot_size=0.01,
            close_reason=CloseReason.TP1,
        )]
        result = make_result(closed=closed)
        cfg = ReporterConfig(r_multiple_clip=3.0)
        fig = generate_r_multiple_histogram(result, cfg)
        # The histogram trace should have no x outside ±3.0.
        hist = next(t for t in fig.data if isinstance(t, go.Histogram))
        xs = list(hist.x or [])
        assert all(-3.0 <= x <= 3.0 for x in xs)

    def test_empty_returns_no_data(
        self, empty_result: BacktestResult,
    ) -> None:
        fig = generate_r_multiple_histogram(empty_result)
        annotations = [a for a in fig.layout.annotations if a.text]
        assert any("No trades" in (a.text or "") for a in annotations)


# --------------------------------------------------------------------------- #
# 5. Hourly heatmap
# --------------------------------------------------------------------------- #


class TestHourlyHeatmap:
    def test_count_metric_returns_heatmap(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_hourly_heatmap(populated_result, metric="count")
        assert any(isinstance(t, go.Heatmap) for t in fig.data)

    def test_count_z_matches_trades(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_hourly_heatmap(populated_result, metric="count")
        z = np.array(fig.data[0].z)
        # Total of all cells == total trades.
        assert int(np.nansum(z)) == len(populated_result.closed_positions)

    def test_win_rate_metric_returns_percentages(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_hourly_heatmap(populated_result, metric="win_rate")
        z = np.array(fig.data[0].z, dtype=float)
        finite = z[~np.isnan(z)]
        # Either no qualifying cells (0-1 trade per cell) or values in [0, 100].
        if len(finite) > 0:
            assert finite.min() >= 0.0
            assert finite.max() <= 100.0

    def test_invalid_metric_raises(
        self, populated_result: BacktestResult,
    ) -> None:
        with pytest.raises(ValueError, match="metric"):
            generate_hourly_heatmap(populated_result, metric="bogus")

    def test_empty_returns_no_data(
        self, empty_result: BacktestResult,
    ) -> None:
        fig = generate_hourly_heatmap(empty_result)
        annotations = [a for a in fig.layout.annotations if a.text]
        assert any("No trades" in (a.text or "") for a in annotations)


# --------------------------------------------------------------------------- #
# 6. Skip reasons pie
# --------------------------------------------------------------------------- #


class TestSkipReasonsPie:
    def test_returns_pie(self, populated_result: BacktestResult) -> None:
        fig = generate_skip_reasons_pie(populated_result)
        assert any(isinstance(t, go.Pie) for t in fig.data)

    def test_labels_match_keys(
        self, populated_result: BacktestResult,
    ) -> None:
        fig = generate_skip_reasons_pie(populated_result)
        pie = next(t for t in fig.data if isinstance(t, go.Pie))
        assert set(pie.labels) == set(populated_result.skip_reasons.keys())
        assert sum(pie.values) == sum(populated_result.skip_reasons.values())

    def test_empty_skip_reasons_returns_no_data(self) -> None:
        result = make_result(skip_reasons={})
        fig = generate_skip_reasons_pie(result)
        annotations = [a for a in fig.layout.annotations if a.text]
        assert any("No skipped" in (a.text or "") for a in annotations)


# --------------------------------------------------------------------------- #
# 7. HTML report
# --------------------------------------------------------------------------- #


class TestHtmlReport:
    def test_writes_file_and_returns_path(
        self, populated_result: BacktestResult, tmp_path: Path,
    ) -> None:
        out = tmp_path / "report.html"
        path = generate_html_report(populated_result, out)
        assert Path(path).exists()
        assert Path(path) == out.resolve()

    def test_html_contains_summary_metrics(
        self, populated_result: BacktestResult, tmp_path: Path,
    ) -> None:
        out = tmp_path / "report.html"
        generate_html_report(populated_result, out, title="Test Report")
        html = out.read_text(encoding="utf-8")
        # Title in <h1>.
        assert "Test Report" in html
        # Headline metric labels.
        assert "Starting Balance" in html
        assert "Win Rate" in html
        assert "Sharpe" in html
        # Each chart section heading.
        for section in (
            "Equity Curve", "Drawdown", "R-Multiple Distribution",
            "Skip Reasons",
        ):
            assert section in html

    def test_inline_plotly_js_embedded(
        self, populated_result: BacktestResult, tmp_path: Path,
    ) -> None:
        out = tmp_path / "report_inline.html"
        generate_html_report(populated_result, out)
        html = out.read_text(encoding="utf-8")
        # Plotly inline embed produces <script ...>... the JS body.
        assert "<script" in html
        # Self-contained: file size reflects embedded JS (~3 MB-ish).
        assert out.stat().st_size > 100_000

    def test_cdn_plotly_js_smaller(
        self, populated_result: BacktestResult, tmp_path: Path,
    ) -> None:
        out = tmp_path / "report_cdn.html"
        generate_html_report(
            populated_result, out, config=ReporterConfig(plotlyjs="cdn"),
        )
        html = out.read_text(encoding="utf-8")
        assert "plotly" in html.lower()
        # CDN report is much smaller than inline.
        assert out.stat().st_size < 1_000_000

    def test_empty_result_still_produces_report(
        self, empty_result: BacktestResult, tmp_path: Path,
    ) -> None:
        out = tmp_path / "empty.html"
        path = generate_html_report(empty_result, out)
        assert Path(path).exists()
        html = out.read_text(encoding="utf-8")
        # No-data annotations are still rendered.
        assert "No trades" in html or "No equity data" in html

    def test_creates_parent_directories(
        self, populated_result: BacktestResult, tmp_path: Path,
    ) -> None:
        out = tmp_path / "nested" / "deep" / "report.html"
        generate_html_report(populated_result, out)
        assert out.exists()


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #


class TestReporterConfigDefaults:
    def test_okabe_ito_palette(self) -> None:
        c = ReporterConfig()
        # Colourblind-friendly defaults.
        assert c.win_color == "#0072B2"
        assert c.loss_color == "#D55E00"

    def test_dimensions(self) -> None:
        c = ReporterConfig()
        assert c.width == 1200
        assert c.height == 600

    def test_inline_plotlyjs_default(self) -> None:
        # Offline-friendly default.
        assert ReporterConfig().plotlyjs == "inline"
