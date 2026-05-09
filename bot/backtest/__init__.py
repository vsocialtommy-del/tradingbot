"""Backtest package — historical replay over the live strategy pipeline.

Public API::

    from bot.backtest import (
        BacktestConfig, BacktestEngine, BacktestResult,
        BacktestMetrics, compute_metrics,
        load_dukascopy_csv, validate_ohlc,
    )

The bot orchestrator (``bot.main``) does **not** import from this
package — it's a one-way dependency.
"""

from bot.backtest.data_loader import (
    load_dukascopy_csv,
    validate_ohlc,
)
from bot.backtest.diagnose import (
    FunnelCounts,
    diagnose,
)
from bot.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
)
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
from bot.backtest.simulator import (
    BacktestBroker,
    BrokerConfig,
    CloseReason,
    LayerFilled,
    OrderType,
    SLHit,
    Tick,
    TPHit,
    generate_ticks_from_bar,
)

__all__ = [
    # data_loader
    "load_dukascopy_csv",
    "validate_ohlc",
    # diagnose
    "FunnelCounts",
    "diagnose",
    # engine
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    # metrics
    "BacktestMetrics",
    "EquityMetrics",
    "SetupMetrics",
    "TradeMetrics",
    "compute_metrics",
    # reporter
    "ReporterConfig",
    "generate_drawdown_chart",
    "generate_equity_curve",
    "generate_hourly_heatmap",
    "generate_html_report",
    "generate_r_multiple_histogram",
    "generate_skip_reasons_pie",
    "generate_trade_scatter",
    # simulator
    "BacktestBroker",
    "BrokerConfig",
    "CloseReason",
    "LayerFilled",
    "OrderType",
    "SLHit",
    "Tick",
    "TPHit",
    "generate_ticks_from_bar",
]
