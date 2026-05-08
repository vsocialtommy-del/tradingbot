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
