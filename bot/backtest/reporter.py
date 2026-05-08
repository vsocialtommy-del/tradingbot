"""Plotly-based visual reports from :class:`BacktestResult`.

Pure functions over the result + an optional :class:`ReporterConfig`.
No state, no I/O outside the explicit ``output_path`` of
:func:`generate_html_report`.

Charts produced
---------------

* :func:`generate_equity_curve`
* :func:`generate_drawdown_chart`
* :func:`generate_trade_scatter`        (needs OHLC ``df``)
* :func:`generate_r_multiple_histogram`
* :func:`generate_hourly_heatmap`
* :func:`generate_skip_reasons_pie`
* :func:`generate_html_report`          (combines all of the above)

Design decisions called out in the PR
-------------------------------------

1. **Functions, not a Reporter class.** No state to share between
   chart calls; passing a :class:`ReporterConfig` is cleaner than
   wrapping every call in instance method dispatch.

2. **Plotly only.** Works in Colab inline, in any browser, and exports
   to standalone HTML. No matplotlib / seaborn, no JS dependencies
   beyond what Plotly bundles.

3. **Self-contained HTML reports.** ``include_plotlyjs="inline"``
   embeds Plotly's JS directly so the file is ~3 MB but works offline,
   in email attachments, on a phone, on a USB stick. The CDN
   alternative would be ~50 KB but breaks the moment the user is
   offline or behind a strict firewall — the wrong default for a
   trading-results report you might forward to your accountant.

4. **Colorblind-friendly palette by default** (Okabe-Ito):

   * wins / TP1: blue ``#0072B2``
   * losses / SL: orange ``#D55E00``
   * neutral / equity: black/grey ``#444``
   * background drawdown fill: ``#FF6B6B`` at low alpha

   Override via :attr:`ReporterConfig.win_color` etc. for legacy
   green/red if requested.

5. **Empty / sparse results render gracefully.** Zero trades → still
   returns a valid Figure with a "No trades to display" annotation.
   This keeps the HTML report layout stable across runs (so
   pre/post-tuning comparisons line up) and avoids exceptions in
   notebook auto-render.

6. **Trade scatter takes the OHLC ``df``** as a separate parameter (the
   :class:`BacktestResult` doesn't carry it — keeps the result
   payload small). When ``df=None``, falls back to a markers-only
   chart.

7. **R-multiple histogram clipped to ±5R**: outliers are still counted
   but binned at the ±5R edges. Long-tail distortion otherwise hides
   the meaningful ±2R range where most trades cluster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from bot.backtest.engine import BacktestResult
from bot.backtest.simulator import BacktestPosition, CloseReason


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReporterConfig:
    """Styling + layout knobs."""

    # Colours (Okabe-Ito colourblind-friendly defaults).
    win_color: str = "#0072B2"
    loss_color: str = "#D55E00"
    neutral_color: str = "#444444"
    drawdown_fill_color: str = "rgba(213, 94, 0, 0.25)"
    grid_color: str = "rgba(0, 0, 0, 0.08)"

    # Sizes.
    width: int = 1200
    height: int = 600
    heatmap_height: int = 500
    pie_height: int = 500

    # Theme.
    template: str = "plotly_white"

    # R-multiple histogram bins.
    r_multiple_clip: float = 5.0
    r_multiple_bin_size: float = 0.5

    # HTML report.
    plotlyjs: str = "inline"
    """``inline`` embeds Plotly's JS in the HTML (offline-friendly).
    Set to ``"cdn"`` for ~50 KB files at the cost of needing internet."""


_DEFAULT_CONFIG = ReporterConfig()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _empty_figure(message: str, cfg: ReporterConfig) -> go.Figure:
    """A blank figure with a centred annotation. Used for zero-data charts."""
    fig = go.Figure()
    fig.update_layout(
        template=cfg.template,
        width=cfg.width, height=cfg.height,
        annotations=[dict(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=cfg.neutral_color),
        )],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _is_winner(p: BacktestPosition) -> bool:
    return p.realised_pnl > 0.01  # match metrics.BE_TOLERANCE


def _is_loser(p: BacktestPosition) -> bool:
    return p.realised_pnl < -0.01


def _r_multiple_for(p: BacktestPosition, clip: float) -> float:
    risk = abs(p.entry_price - p.sl) * p.lot_size * 100.0  # contract size
    if risk <= 0:
        return 0.0
    r = p.realised_pnl / risk
    return float(max(-clip, min(clip, r)))


def _format_pnl(v: float) -> str:
    return f"${v:+.2f}"


# --------------------------------------------------------------------------- #
# 1. Equity curve
# --------------------------------------------------------------------------- #


def generate_equity_curve(
    result: BacktestResult, config: ReporterConfig | None = None,
) -> go.Figure:
    """Account balance over time, with TP1 / SL trade markers."""
    cfg = config or _DEFAULT_CONFIG
    if result.equity_curve.empty:
        return _empty_figure("No equity data", cfg)

    eq = result.equity_curve
    fig = go.Figure()

    # Equity line.
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        mode="lines",
        name="Equity",
        line=dict(color=cfg.neutral_color, width=2),
        hovertemplate="%{x}<br>$%{y:,.2f}<extra></extra>",
    ))

    # Trade markers.
    tp1 = [
        p for p in result.closed_positions
        if p.close_reason == CloseReason.TP1
    ]
    sl = [
        p for p in result.closed_positions
        if p.close_reason == CloseReason.SL
    ]

    if tp1:
        fig.add_trace(go.Scatter(
            x=[p.exit_time for p in tp1],
            y=[_equity_at(eq, p.exit_time) for p in tp1],
            mode="markers",
            name=f"TP1 hits ({len(tp1)})",
            marker=dict(color=cfg.win_color, size=8, symbol="triangle-up"),
            hovertemplate=(
                "TP1<br>%{x}<br>"
                "Setup %{customdata[0]} L%{customdata[1]}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
            customdata=[
                (p.setup_id, p.layer, _format_pnl(p.realised_pnl))
                for p in tp1
            ],
        ))
    if sl:
        fig.add_trace(go.Scatter(
            x=[p.exit_time for p in sl],
            y=[_equity_at(eq, p.exit_time) for p in sl],
            mode="markers",
            name=f"SL hits ({len(sl)})",
            marker=dict(color=cfg.loss_color, size=8, symbol="triangle-down"),
            hovertemplate=(
                "SL<br>%{x}<br>"
                "Setup %{customdata[0]} L%{customdata[1]}<br>"
                "%{customdata[2]}<extra></extra>"
            ),
            customdata=[
                (p.setup_id, p.layer, _format_pnl(p.realised_pnl))
                for p in sl
            ],
        ))

    # Mark max-drawdown trough.
    peaks = eq.cummax()
    underwater = eq - peaks
    if not underwater.empty:
        trough_idx = underwater.idxmin()
        if underwater.loc[trough_idx] < 0:
            fig.add_annotation(
                x=trough_idx, y=eq.loc[trough_idx],
                text=f"Max DD: {_format_pnl(underwater.loc[trough_idx])}",
                showarrow=True, arrowhead=2,
                font=dict(color=cfg.loss_color),
            )

    fig.update_layout(
        title=f"Equity Curve  ·  Net P&L {_format_pnl(eq.iloc[-1] - result.config.starting_balance)}",
        xaxis_title="Time", yaxis_title="Account balance (USD)",
        template=cfg.template,
        width=cfg.width, height=cfg.height,
        hovermode="x unified",
    )
    return fig


def _equity_at(eq: pd.Series, t: datetime | None) -> float:
    """Look up the equity at-or-before time ``t``. Used for marker Y-coords."""
    if t is None:
        return float(eq.iloc[-1])
    # ``asof`` returns the last value at-or-before t; NaN if none.
    val = eq.asof(t)
    if pd.isna(val):
        return float(eq.iloc[0])
    return float(val)


# --------------------------------------------------------------------------- #
# 2. Drawdown
# --------------------------------------------------------------------------- #


def generate_drawdown_chart(
    result: BacktestResult, config: ReporterConfig | None = None,
) -> go.Figure:
    """Underwater chart: drawdown % from running peak."""
    cfg = config or _DEFAULT_CONFIG
    if result.equity_curve.empty:
        return _empty_figure("No equity data", cfg)

    eq = result.equity_curve
    peaks = eq.cummax()
    dd_pct = (eq - peaks) / peaks * 100.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd_pct.index, y=dd_pct.values,
        mode="lines",
        name="Drawdown",
        line=dict(color=cfg.loss_color, width=1),
        fill="tozeroy",
        fillcolor=cfg.drawdown_fill_color,
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))

    # Annotate trough.
    if not dd_pct.empty:
        trough_idx = dd_pct.idxmin()
        trough_val = float(dd_pct.loc[trough_idx])
        if trough_val < 0:
            fig.add_annotation(
                x=trough_idx, y=trough_val,
                text=f"{trough_val:.2f}%",
                showarrow=True, arrowhead=2,
                font=dict(color=cfg.loss_color),
            )

    fig.update_layout(
        title=f"Drawdown  ·  Max {result.metrics.equity.max_drawdown_pct:.2f}%",
        xaxis_title="Time", yaxis_title="Drawdown (%)",
        template=cfg.template,
        width=cfg.width, height=cfg.height,
    )
    return fig


# --------------------------------------------------------------------------- #
# 3. Trade scatter on price chart
# --------------------------------------------------------------------------- #


def generate_trade_scatter(
    result: BacktestResult,
    df: pd.DataFrame | None = None,
    config: ReporterConfig | None = None,
) -> go.Figure:
    """Price chart with entry/exit markers for every closed position.

    When ``df`` is None, the chart still plots the trade markers (with
    no underlying price line). Useful for spot-checking trade timing
    in absence of the OHLC data.
    """
    cfg = config or _DEFAULT_CONFIG
    if not result.closed_positions:
        return _empty_figure("No trades to display", cfg)

    fig = go.Figure()

    # Underlying price (close series) when available.
    if df is not None and "close" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            mode="lines",
            name="Close",
            line=dict(color=cfg.neutral_color, width=1),
            hoverinfo="skip",
        ))
    elif df is not None:
        logger.warning(
            "reporter: trade_scatter: 'close' column missing from df; "
            "plotting markers only"
        )

    winners = [p for p in result.closed_positions if _is_winner(p)]
    losers = [p for p in result.closed_positions if _is_loser(p)]

    if winners:
        fig.add_trace(go.Scatter(
            x=[p.opened_at for p in winners],
            y=[p.entry_price for p in winners],
            mode="markers",
            name=f"Winners ({len(winners)})",
            marker=dict(color=cfg.win_color, size=8, symbol="triangle-up"),
            hovertemplate=(
                "Entry %{x}<br>$%{y:.2f}<br>"
                "Setup %{customdata[0]} L%{customdata[1]}<br>"
                "P&L %{customdata[2]}<extra></extra>"
            ),
            customdata=[
                (p.setup_id, p.layer, _format_pnl(p.realised_pnl))
                for p in winners
            ],
        ))
    if losers:
        fig.add_trace(go.Scatter(
            x=[p.opened_at for p in losers],
            y=[p.entry_price for p in losers],
            mode="markers",
            name=f"Losers ({len(losers)})",
            marker=dict(color=cfg.loss_color, size=8, symbol="triangle-down"),
            hovertemplate=(
                "Entry %{x}<br>$%{y:.2f}<br>"
                "Setup %{customdata[0]} L%{customdata[1]}<br>"
                "P&L %{customdata[2]}<extra></extra>"
            ),
            customdata=[
                (p.setup_id, p.layer, _format_pnl(p.realised_pnl))
                for p in losers
            ],
        ))

    fig.update_layout(
        title=f"Trades on Price  ·  {len(winners)}W / {len(losers)}L",
        xaxis_title="Time", yaxis_title="Price (USD)",
        template=cfg.template,
        width=cfg.width, height=cfg.height,
        hovermode="closest",
    )
    return fig


# --------------------------------------------------------------------------- #
# 4. R-multiple histogram
# --------------------------------------------------------------------------- #


def generate_r_multiple_histogram(
    result: BacktestResult, config: ReporterConfig | None = None,
) -> go.Figure:
    """Distribution of trade outcomes in R terms.

    R-multiples are clipped to ``±r_multiple_clip`` (default ±5) so a
    handful of outliers don't compress the histogram.
    """
    cfg = config or _DEFAULT_CONFIG
    if not result.closed_positions:
        return _empty_figure("No trades to display", cfg)

    rs = [_r_multiple_for(p, cfg.r_multiple_clip) for p in result.closed_positions]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rs,
        xbins=dict(
            start=-cfg.r_multiple_clip,
            end=cfg.r_multiple_clip + cfg.r_multiple_bin_size,
            size=cfg.r_multiple_bin_size,
        ),
        marker=dict(color=cfg.win_color, line=dict(width=1, color="white")),
        name="Trades",
        hovertemplate="R=%{x:.1f}<br>Count=%{y}<extra></extra>",
    ))

    # Reference lines.
    for r, label, colour in (
        (-1.0, "-1R", cfg.loss_color),
        (0.0, "0R", cfg.neutral_color),
        (1.0, "+1R", cfg.win_color),
    ):
        fig.add_vline(
            x=r, line_dash="dash", line_color=colour, line_width=1,
            annotation_text=label, annotation_position="top",
        )

    avg_r = float(np.mean(rs)) if rs else 0.0
    fig.update_layout(
        title=(
            f"R-Multiple Distribution  ·  "
            f"avg R={avg_r:.2f}, n={len(rs)}"
        ),
        xaxis_title="R-multiple (clipped at ±{:.0f}R)".format(cfg.r_multiple_clip),
        yaxis_title="Count",
        template=cfg.template,
        width=cfg.width, height=cfg.height,
        bargap=0.05,
    )
    return fig


# --------------------------------------------------------------------------- #
# 5. Hourly / day-of-week heatmap
# --------------------------------------------------------------------------- #


def generate_hourly_heatmap(
    result: BacktestResult,
    config: ReporterConfig | None = None,
    *,
    metric: str = "count",
) -> go.Figure:
    """Day-of-week × hour-of-day heatmap.

    ``metric``:
      * ``"count"``     — number of trades opened in the cell (default)
      * ``"win_rate"``  — % winners; cells with <2 trades shown as NaN
                          (sample too small to be meaningful)
    """
    cfg = config or _DEFAULT_CONFIG
    if not result.closed_positions:
        return _empty_figure("No trades to display", cfg)
    if metric not in ("count", "win_rate"):
        raise ValueError(f"metric must be 'count' or 'win_rate', got {metric!r}")

    # Build a 7×24 matrix.
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    counts = np.zeros((7, 24), dtype=int)
    wins = np.zeros((7, 24), dtype=int)
    for p in result.closed_positions:
        dow = p.opened_at.weekday()
        hour = p.opened_at.hour
        counts[dow, hour] += 1
        if _is_winner(p):
            wins[dow, hour] += 1

    if metric == "count":
        z = counts.astype(float)
        title = f"Trade Count by Day × Hour  ·  total {counts.sum()}"
        colorscale = "Blues"
        zfmt = "%{z:.0f}"
    else:
        z = np.full_like(counts, np.nan, dtype=float)
        mask = counts >= 2
        z[mask] = wins[mask] / counts[mask] * 100.0
        title = "Win Rate (%) by Day × Hour  ·  cells with <2 trades blank"
        colorscale = "RdYlGn"
        zfmt = "%{z:.1f}%"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"{h:02d}" for h in range(24)],
        y=dow_labels,
        colorscale=colorscale,
        hovertemplate=f"{{y}} {{x}}:00<br>{zfmt}<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Hour (UTC)", yaxis_title="Day",
        template=cfg.template,
        width=cfg.width, height=cfg.heatmap_height,
    )
    return fig


# --------------------------------------------------------------------------- #
# 6. Skip reasons pie
# --------------------------------------------------------------------------- #


def generate_skip_reasons_pie(
    result: BacktestResult, config: ReporterConfig | None = None,
) -> go.Figure:
    """Why detected zones didn't become setups."""
    cfg = config or _DEFAULT_CONFIG
    reasons = result.skip_reasons
    if not reasons:
        return _empty_figure("No skipped setups", cfg)

    labels = list(reasons.keys())
    values = list(reasons.values())
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(line=dict(color="white", width=2)),
        hovertemplate="%{label}<br>%{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=f"Skip Reasons  ·  total {sum(values)}",
        template=cfg.template,
        width=cfg.width, height=cfg.pie_height,
    )
    return fig


# --------------------------------------------------------------------------- #
# 7. HTML report
# --------------------------------------------------------------------------- #


def generate_html_report(
    result: BacktestResult,
    output_path: str | Path,
    *,
    df: pd.DataFrame | None = None,
    config: ReporterConfig | None = None,
    title: str = "Backtest Report",
) -> str:
    """Combine all charts + a metrics table into one self-contained HTML file.

    Returns the absolute path to the written file.
    """
    cfg = config or _DEFAULT_CONFIG
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    figures = [
        ("Equity Curve", generate_equity_curve(result, cfg)),
        ("Drawdown", generate_drawdown_chart(result, cfg)),
        ("Trades on Price", generate_trade_scatter(result, df, cfg)),
        ("R-Multiple Distribution", generate_r_multiple_histogram(result, cfg)),
        ("Hourly Heatmap (Trade Count)", generate_hourly_heatmap(result, cfg)),
        ("Skip Reasons", generate_skip_reasons_pie(result, cfg)),
    ]

    # First chart includes the Plotly JS; subsequent embed only the div.
    chart_html_parts: list[str] = []
    for i, (label, fig) in enumerate(figures):
        include_js = cfg.plotlyjs if i == 0 else False
        chart_html_parts.append(
            f'<section class="chart"><h2>{_html_escape(label)}</h2>'
            + fig.to_html(
                full_html=False,
                include_plotlyjs=include_js,  # type: ignore[arg-type]
                config={"displaylogo": False},
            )
            + "</section>"
        )

    summary_html = _summary_table_html(result, cfg)

    html = _PAGE_TEMPLATE.format(
        title=_html_escape(title),
        summary=summary_html,
        charts="\n".join(chart_html_parts),
    )

    out.write_text(html, encoding="utf-8")
    logger.info(
        f"reporter: wrote {out} ({out.stat().st_size / 1024:.0f} KB)"
    )
    return str(out)


def _summary_table_html(
    result: BacktestResult, cfg: ReporterConfig,
) -> str:
    """Plain HTML metrics summary at the top of the report."""
    m = result.metrics
    rows = [
        ("Starting Balance", f"${m.equity.starting_balance:,.2f}"),
        ("Ending Balance", f"${m.equity.ending_balance:,.2f}"),
        ("Net P&L", _format_pnl(m.trades.net_pnl)),
        ("Total Return", f"{m.equity.total_return_pct:+.2f}%"),
        ("Max Drawdown", f"{m.equity.max_drawdown_pct:.2f}% (${m.equity.max_drawdown_dollars:,.2f})"),
        ("Sharpe (annualised)", f"{m.equity.sharpe_ratio:.2f}"),
        ("", ""),
        ("Total Trades", f"{m.trades.total}"),
        ("Win Rate", f"{m.trades.win_rate * 100:.1f}%"),
        ("Profit Factor", f"{m.trades.profit_factor:.2f}" if m.trades.profit_factor != float("inf") else "∞"),
        ("Expectancy / trade", _format_pnl(m.trades.expectancy)),
        ("Avg R-multiple", f"{m.trades.avg_r_multiple:+.2f}R"),
        ("Avg Winner", _format_pnl(m.trades.avg_winner)),
        ("Avg Loser", _format_pnl(m.trades.avg_loser)),
        ("Largest Winner", _format_pnl(m.trades.largest_winner)),
        ("Largest Loser", _format_pnl(m.trades.largest_loser)),
        ("", ""),
        ("Setups Detected", f"{m.setups.detected}"),
        ("Setups Taken", f"{m.setups.taken}"),
        ("TP1 Hit Rate", f"{m.setups.tp1_hit_rate * 100:.1f}%"),
        ("SL Stop Rate", f"{m.setups.sl_stop_rate * 100:.1f}%"),
    ]
    body = "\n".join(
        '<tr><td class="k">{k}</td><td class="v">{v}</td></tr>'.format(
            k=_html_escape(k), v=_html_escape(v),
        ) if k or v else '<tr class="spacer"><td colspan="2"></td></tr>'
        for k, v in rows
    )
    return f'<table class="summary">{body}</table>'


_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, sans-serif; max-width: 1300px; margin: 24px auto;
          padding: 0 16px; color: #222; }}
  h1 {{ font-weight: 600; margin-bottom: 8px; }}
  h2 {{ font-weight: 500; margin: 28px 0 8px; color: #333; }}
  table.summary {{ border-collapse: collapse; margin: 16px 0 32px;
          font-size: 14px; }}
  table.summary td {{ padding: 4px 16px 4px 0; border: none; }}
  table.summary td.k {{ color: #666; }}
  table.summary td.v {{ font-variant-numeric: tabular-nums;
          font-weight: 500; }}
  table.summary tr.spacer td {{ padding: 8px 0; }}
  section.chart {{ margin: 24px 0; }}
  .meta {{ color: #888; font-size: 12px; margin-bottom: 24px; }}
</style>
</head><body>
<h1>{title}</h1>
{summary}
{charts}
</body></html>
"""


def _html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
