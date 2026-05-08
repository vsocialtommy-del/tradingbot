"""Backtest engine — bar-by-bar replay over historical OHLC.

The engine reuses the live strategy pipeline (Phase B) **exactly** so a
zone detected here is the same zone the live bot would detect. The
broker, exposure cap, daily halt, layer-trigger handling, TP1 partial
+ SL→BE move, and cascade-cancellation logic are re-implemented here
in plain Python so the backtest stays decoupled from ``bot.execution.*``
(which is tightly bound to MT5 and Supabase).

Per-bar flow
------------

1. Generate 4 ticks from the bar (OHLC walk; see :mod:`simulator`).
2. For each tick:

   a. ``broker.process_tick`` checks SL → TP → pending fills.
   b. The engine reacts to each emitted event:
      * ``SLHit``      → cascade-cancel pending orders for that setup;
                         mark the setup CLOSED if no positions remain.
      * ``TPHit``      → cascade-cancel pending orders;
                         move SL to BE on the remaining runner;
                         mark the setup TP1_HIT.
      * ``LayerFilled``→ mark the setup ACTIVE.
   c. Equity at the tick is recorded for the curve.

3. On bar close, gated by **daily halt** + **exposure cap**:
   ``run_strategy_pipeline`` is run on the history slice. New zones are
   deduped against existing setups (by direction + top + bottom). For
   each fresh zone, the engine computes SL (via the live algorithm —
   see :func:`_calculate_sl`), validates the SL distance, and places
   three layered limit orders + waits for the broker's tick stream to
   fill them.

End-of-data
-----------

Any positions still open when the historical data ends are force-closed
at the last bar's close price with reason ``END_OF_DATA``. Pending
orders are cancelled. This is what gives the metrics a clean
"settled" picture — leaving open positions out would understate
drawdown and skew win rate.

Design decisions
----------------

* **Same-zone dedup** uses ``(direction, round(top, 2), round(bottom, 2))``.
  Different formed_at on the same band ⇒ same zone.
* **Re-entry after SL** is **not** attempted. Live ``track_imbalance``
  sets ``is_tapped=True`` once price entered the zone, and
  ``run_strategy_pipeline`` filters tapped zones at exit. So once the
  setup's first layer fills (= zone tapped), the pipeline stops
  returning that zone, and we never re-enter naturally.
* **No news filter** — the spec calls this out as a known limitation:
  no historical Finnhub data, so blackout windows aren't simulated.
* **Fixed lot size** for v1; ``BacktestConfig.fixed_lot_size`` is
  passed unchanged to every layer (v1.1 will integrate
  ``position_sizing``).
* **SL calc** reuses the live ``SLManager.calculate_initial_sl`` via a
  loud-failure stub for the unused mt5/supabase deps. If the live
  method ever starts touching its deps, the stub raises a clear
  message. This is preferable to forking the algorithm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import pandas as pd
from loguru import logger

from bot.backtest.metrics import BacktestMetrics, compute_metrics
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
from bot.exits.sl_manager import SLCalculation, SLManager, SLManagerConfig
from bot.risk.daily_halt import current_trading_date
from bot.strategy.imbalance import ImbalanceZone
from bot.strategy.pipeline import (
    StrategyPipelineConfig,
    run_strategy_pipeline,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BacktestConfig:
    """All knobs for a backtest run.

    Notes on units
    --------------
    ``spread_points`` and ``slippage_points`` use the **broker convention**
    (1 point = $0.01 for XAUUSD). Default 23 = $0.23 matches Vantage Gold
    raw spread; 1.5 = $0.015 is a typical slippage allowance.

    ``sl_buffer_points``, ``layer_*_offset_points``,
    ``tp1_fixed_distance_points`` use **price units** (= dollars for
    XAUUSD). 17.5 means $17.50. Inherited from live ``bot_config``
    where the same naming-but-different-meaning convention is in
    place. Standardising would require a coordinated migration.
    """

    # Capital
    starting_balance: float = 10_000.0

    # Broker simulation
    spread_points: float = 23.0
    slippage_points: float = 1.5
    commission_per_lot: float = 3.50
    sl_tp_conflict_sl_wins: bool = True

    # Position sizing (v1: fixed)
    fixed_lot_size: float = 0.01

    # Risk gates
    max_simultaneous_setups: int = 3
    daily_loss_limit_pct: float = 10.0

    # SL / TP / layers (price units)
    sl_buffer_points: float = 17.5
    recent_swing_lookback: int = 20
    swing_strength: int = 3
    min_sl_distance_points: float = 5.0
    max_sl_distance_points: float = 200.0
    layer_2_offset_points: float = 2.5
    layer_3_offset_points: float = 5.0
    tp1_method: Literal["BOS_LEVEL", "FIXED_DISTANCE"] = "BOS_LEVEL"
    tp1_fixed_distance_points: float = 70.0

    # Strategy pipeline tunables
    strategy: StrategyPipelineConfig = field(
        default_factory=StrategyPipelineConfig,
    )

    # Engine cadence
    min_history_bars: int = 100
    """Skip the first N bars before running the pipeline. Pipeline
    needs swing detection + lookback windows to be populated."""

    progress_log_every_bars: int = 1000


# --------------------------------------------------------------------------- #
# Internal state
# --------------------------------------------------------------------------- #


@dataclass
class _BacktestSetup:
    setup_id: int
    direction: Literal["BUY", "SELL"]
    zone_top: float
    zone_bottom: float
    sl_price: float
    tp1_price: float
    layer_prices: dict[int, float]
    pending_tickets: dict[int, int] = field(default_factory=dict)
    open_tickets: dict[int, int] = field(default_factory=dict)
    status: Literal["PENDING", "ACTIVE", "TP1_HIT", "CLOSED"] = "PENDING"
    created_at: datetime | None = None
    closed_at: datetime | None = None

    def zone_key(self) -> tuple[str, float, float]:
        return (self.direction, round(self.zone_top, 2), round(self.zone_bottom, 2))


@dataclass
class _DailyHaltState:
    current_date: object | None = None  # date | None
    starting_balance: float = 0.0
    halted: bool = False


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    closed_positions: list  # list[BacktestPosition]
    equity_curve: pd.Series
    config: BacktestConfig
    bars_processed: int
    setups_detected: int
    setups_taken: int
    skip_reasons: dict[str, int]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.broker = BacktestBroker(
            starting_balance=self.config.starting_balance,
            config=BrokerConfig(
                spread_points=self.config.spread_points,
                slippage_points=self.config.slippage_points,
                commission_per_lot=self.config.commission_per_lot,
                sl_tp_conflict_sl_wins=self.config.sl_tp_conflict_sl_wins,
            ),
        )
        self._setups: list[_BacktestSetup] = []
        self._setup_id_counter: int = 0
        self._zone_keys_seen: set[tuple[str, float, float]] = set()
        self._setups_detected: int = 0
        self._skip_reasons: dict[str, int] = {}
        self._daily_state = _DailyHaltState()
        self._equity_records: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Replay ``df`` (M5 OHLC, tz-aware UTC index) and return the result."""
        if df.index.tz is None:
            raise ValueError("df.index must be tz-aware UTC")
        n = len(df)
        if n <= self.config.min_history_bars:
            raise ValueError(
                f"need > {self.config.min_history_bars} bars; got {n}"
            )

        for i in range(self.config.min_history_bars, n):
            bar_time = df.index[i].to_pydatetime()
            bar = df.iloc[i]
            ticks = generate_ticks_from_bar(
                bar_time=bar_time,
                open_=float(bar["open"]),
                high=float(bar["high"]),
                low=float(bar["low"]),
                close=float(bar["close"]),
                spread_points=self.config.spread_points,
            )
            for tick in ticks:
                self._step_tick(tick)

            # On bar close: maybe detect new setups.
            self._maybe_run_strategy(df.iloc[: i + 1], bar_time, ticks[-1])

            if (
                self.config.progress_log_every_bars > 0
                and (i + 1) % self.config.progress_log_every_bars == 0
            ):
                logger.info(
                    f"backtest: {i + 1}/{n} bars "
                    f"(balance=${self.broker.balance:.2f} "
                    f"setups_taken={len(self._setups)})"
                )

        # End-of-data: force-close everything at the last tick.
        self._force_close_open(df)

        return self._build_result(df)

    # ------------------------------------------------------------------ #
    # Per-tick
    # ------------------------------------------------------------------ #

    def _step_tick(self, tick: Tick) -> None:
        events = self.broker.process_tick(tick)
        for event in events:
            if isinstance(event, SLHit):
                self._on_sl(event, tick)
            elif isinstance(event, TPHit):
                self._on_tp(event, tick)
            elif isinstance(event, LayerFilled):
                self._on_layer(event)
        self._equity_records.append((tick.time, self.broker.equity))

    def _on_sl(self, event: SLHit, tick: Tick) -> None:
        """Update setup state. Pending cancellation is broker-side (cascade)."""
        setup = self._find_setup(event.setup_id)
        if setup is None:
            return
        setup.pending_tickets.clear()  # broker already cancelled
        setup.open_tickets.pop(event.layer, None)
        if not setup.open_tickets:
            setup.status = "CLOSED"
            setup.closed_at = tick.time

    def _on_tp(self, event: TPHit, tick: Tick) -> None:
        """Mark TP1_HIT, move SL to BE on remaining runners."""
        setup = self._find_setup(event.setup_id)
        if setup is None:
            return
        setup.pending_tickets.clear()  # broker already cancelled
        setup.status = "TP1_HIT"
        for layer, ticket in list(setup.open_tickets.items()):
            pos = self.broker.positions.get(ticket)
            if pos is None:
                setup.open_tickets.pop(layer, None)
                continue
            self.broker.modify_position(ticket, sl=pos.entry_price)

    def _on_layer(self, event: LayerFilled) -> None:
        setup = self._find_setup(event.setup_id)
        if setup is None:
            return
        # Pending → open.
        setup.pending_tickets.pop(event.layer, None)
        setup.open_tickets[event.layer] = event.ticket
        if setup.status == "PENDING":
            setup.status = "ACTIVE"

    def _find_setup(self, setup_id: int) -> _BacktestSetup | None:
        for s in self._setups:
            if s.setup_id == setup_id:
                return s
        return None

    # ------------------------------------------------------------------ #
    # On bar close: detect new setups
    # ------------------------------------------------------------------ #

    def _maybe_run_strategy(
        self,
        history: pd.DataFrame,
        bar_time: datetime,
        last_tick: Tick,
    ) -> None:
        # Daily halt rolls over at 17:00 ET (handled by current_trading_date).
        self._update_daily_halt(bar_time)
        if self._daily_state.halted:
            return

        try:
            zones = run_strategy_pipeline(history, self.config.strategy)
        except Exception:
            logger.exception("backtest: strategy pipeline raised")
            return
        if not zones:
            return

        # We always run the pipeline (instead of early-returning at the
        # cap) so cap-blocked zones are recorded in skip_reasons —
        # operators want telemetry on what they're missing.
        active_count = sum(
            1 for s in self._setups
            if s.status in ("PENDING", "ACTIVE", "TP1_HIT")
        )

        for zone in zones:
            self._setups_detected += 1
            zone_key = (
                zone.direction, round(zone.top, 2), round(zone.bottom, 2),
            )
            if zone_key in self._zone_keys_seen:
                continue
            self._zone_keys_seen.add(zone_key)

            if active_count >= self.config.max_simultaneous_setups:
                self._record_skip("exposure_cap")
                continue

            if not self._try_open_setup(zone, history, bar_time, last_tick):
                continue
            active_count += 1

    def _try_open_setup(
        self,
        zone: ImbalanceZone,
        history: pd.DataFrame,
        bar_time: datetime,
        last_tick: Tick,
    ) -> bool:
        # 1. SL.
        try:
            sl_calc = _calculate_sl(zone, history, self._sl_config())
        except Exception:
            self._record_skip("sl_calc_failed")
            return False

        entry = zone.top if zone.direction == "BUY" else zone.bottom
        sl_distance = abs(entry - sl_calc.sl_price)
        if sl_distance < self.config.min_sl_distance_points:
            self._record_skip("sl_too_close")
            return False
        if sl_distance > self.config.max_sl_distance_points:
            self._record_skip("sl_too_far")
            return False

        # 2. TP1.
        tp1 = self._calculate_tp1(zone, entry)
        # Guard: TP must be on the profit side of entry. If BoS broken_level
        # ended up the wrong side (rare, but spec says fall back), use fixed.
        if zone.direction == "BUY" and tp1 <= entry:
            tp1 = entry + self.config.tp1_fixed_distance_points
        elif zone.direction == "SELL" and tp1 >= entry:
            tp1 = entry - self.config.tp1_fixed_distance_points

        # 3. Construct setup.
        self._setup_id_counter += 1
        sid = self._setup_id_counter
        layer_prices = self._compute_layer_prices(zone)
        setup = _BacktestSetup(
            setup_id=sid,
            direction=zone.direction,
            zone_top=zone.top,
            zone_bottom=zone.bottom,
            sl_price=sl_calc.sl_price,
            tp1_price=tp1,
            layer_prices=layer_prices,
            created_at=bar_time,
        )

        # 4. Place 3 limit orders.
        order_type = (
            OrderType.BUY_LIMIT if zone.direction == "BUY"
            else OrderType.SELL_LIMIT
        )
        for layer, price in layer_prices.items():
            order = self.broker.place_pending_order(
                direction=zone.direction,
                order_type=order_type,
                price=price,
                lot_size=self.config.fixed_lot_size,
                sl=sl_calc.sl_price,
                tp=tp1,
                setup_id=sid,
                layer=layer,
                now=last_tick.time,
            )
            setup.pending_tickets[layer] = order.ticket

        self._setups.append(setup)
        return True

    def _compute_layer_prices(
        self, zone: ImbalanceZone,
    ) -> dict[int, float]:
        """Layer 1 at zone edge; 2 + 3 step into the zone by config offsets."""
        if zone.direction == "BUY":
            edge = zone.top
            return {
                1: edge,
                2: edge - self.config.layer_2_offset_points,
                3: edge - self.config.layer_3_offset_points,
            }
        edge = zone.bottom
        return {
            1: edge,
            2: edge + self.config.layer_2_offset_points,
            3: edge + self.config.layer_3_offset_points,
        }

    def _calculate_tp1(self, zone: ImbalanceZone, entry: float) -> float:
        if (
            self.config.tp1_method == "BOS_LEVEL"
            and zone.bos_event is not None
        ):
            return float(zone.bos_event.broken_level)
        # Fixed fallback.
        if zone.direction == "BUY":
            return entry + self.config.tp1_fixed_distance_points
        return entry - self.config.tp1_fixed_distance_points

    def _sl_config(self) -> SLManagerConfig:
        return SLManagerConfig(
            symbol="XAUUSD",
            swing_strength=self.config.swing_strength,
            recent_swing_lookback=self.config.recent_swing_lookback,
            sl_buffer_points=self.config.sl_buffer_points,
            min_sl_distance_points=self.config.min_sl_distance_points,
            max_sl_distance_points=self.config.max_sl_distance_points,
        )

    # ------------------------------------------------------------------ #
    # Daily halt
    # ------------------------------------------------------------------ #

    def _update_daily_halt(self, bar_time: datetime) -> None:
        trading_date = current_trading_date(bar_time)
        if self._daily_state.current_date != trading_date:
            # Day rolled over — capture starting balance.
            self._daily_state.current_date = trading_date
            self._daily_state.starting_balance = self.broker.balance
            self._daily_state.halted = False
            return
        # Check intraday drawdown.
        starting = self._daily_state.starting_balance
        if starting <= 0:
            return
        dd_pct = (
            (self.broker.balance - starting) / starting * 100.0
        )
        if dd_pct < -self.config.daily_loss_limit_pct:
            self._daily_state.halted = True

    # ------------------------------------------------------------------ #
    # End-of-data + result
    # ------------------------------------------------------------------ #

    def _force_close_open(self, df: pd.DataFrame) -> None:
        if not self.broker.positions and not self.broker.pending:
            return
        last_bar_time = df.index[-1].to_pydatetime()
        last_close = float(df["close"].iloc[-1])
        # Synthesize a tick at the bar close mid.
        from bot.backtest.simulator import POINT_TO_DOLLARS
        half_spread = (self.config.spread_points * POINT_TO_DOLLARS) / 2.0
        final_tick = Tick(
            time=last_bar_time,
            bid=last_close - half_spread,
            ask=last_close + half_spread,
        )
        # Cancel pending orders.
        for ticket in list(self.broker.pending.keys()):
            self.broker.cancel_pending(ticket)
        # Close open positions at market.
        for ticket in list(self.broker.positions.keys()):
            self.broker.close_position(
                ticket, tick=final_tick, fraction=1.0,
                reason=CloseReason.END_OF_DATA,
            )

    def _build_result(self, df: pd.DataFrame) -> BacktestResult:
        if self._equity_records:
            times, vals = zip(*self._equity_records)
            equity_curve = pd.Series(vals, index=pd.DatetimeIndex(times))
        else:
            equity_curve = pd.Series(dtype=float)

        metrics = compute_metrics(
            closed_positions=self.broker.closed_positions,
            equity_curve=equity_curve,
            starting_balance=self.config.starting_balance,
            setups_detected=self._setups_detected,
            setups_taken=len(self._setups),
            skip_reasons=self._skip_reasons,
        )
        return BacktestResult(
            metrics=metrics,
            closed_positions=list(self.broker.closed_positions),
            equity_curve=equity_curve,
            config=self.config,
            bars_processed=len(df),
            setups_detected=self._setups_detected,
            setups_taken=len(self._setups),
            skip_reasons=dict(self._skip_reasons),
        )

    def _record_skip(self, reason: str) -> None:
        self._skip_reasons[reason] = self._skip_reasons.get(reason, 0) + 1


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


class _UnusedDep:
    """Loud-failure stub for SLManager constructor params it never reads.

    See module docstring (Design decisions). If
    ``SLManager.calculate_initial_sl`` ever starts touching its
    ``mt5`` / ``supabase`` dependencies, the AttributeError raised here
    will surface clearly in tests rather than producing a silent bug.
    """

    def __getattr__(self, name: str) -> object:
        raise AttributeError(
            f"backtest stub: SLManager.calculate_initial_sl tried to "
            f"access dep attribute {name!r}; the live API may have "
            f"changed and the backtest engine needs updating."
        )


def _calculate_sl(
    zone: ImbalanceZone,
    df: pd.DataFrame,
    cfg: SLManagerConfig,
) -> SLCalculation:
    """Reuse the live SL algorithm via SLManager with stub deps."""
    stub = _UnusedDep()
    mgr = SLManager(mt5=stub, supabase=stub, config=cfg)  # type: ignore[arg-type]
    return mgr.calculate_initial_sl(zone, df)
