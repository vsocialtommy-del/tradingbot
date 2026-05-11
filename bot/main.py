"""Bot orchestrator — main loop entry point (Phase C completion).

Composes every Phase B + C module into a single while-loop that:

* Polls MT5 for the current tick (per iteration ~10 Hz)
* Fires entry triggers + checks TP1 (per iteration)
* Detects externally closed positions (every 30 s)
* Refreshes the live ``bot_config`` flags — kill switch + pause-until —
  every 30 s
* Fetches OHLC + runs the strategy pipeline (only on M5 candle close)
* Reconciles state with MT5 (every 5 min)
* Emits heartbeat logs (every 5 min)

Run with::

    python -m bot.main

Design decisions called out in the PR
-------------------------------------

1. **Class with ``run_iteration(now)`` factored from ``run()``.** The
   while-loop is a paper-thin wrapper that catches exceptions, sleeps,
   and re-enters. All real work is in ``run_iteration``, which takes
   ``now`` as a parameter so tests can drive it without faking time.

2. **Kill-switch + pause-until are the only LIVE configs.** Strategy
   parameters and per-manager configs are read once at startup. A
   parameter change in the dashboard requires a bot restart for v1 —
   simpler than per-iteration config diffing, and matches the operator
   workflow ("pause the bot, change params, restart"). Kill switch
   and the manual pause-until DO need to take effect mid-trade, so
   those are re-read every 30 s.

3. **Strategy pipeline is gated on M5 candle close.** Re-running the
   detector every 100 ms would be wasteful (the inputs only change on
   bar close) and could re-detect the same zone repeatedly mid-bar.
   We track the last bar's timestamp and only re-run when it
   advances. ``OHLCProvider`` has its own 30-s cache too.

4. **News / kill / daily-halt block new setups but don't force-close
   existing positions.** Force-closing winning runners just because a
   blackout window opened would be punitive. Operators close manually
   if they want to (mobile MT5 app, dashboard kill switch on the
   broker side). v1 keeps this behaviour minimal and predictable.

5. **Per-iteration top-level try/except with 5 s backoff.** A single
   transient error (network blip, Supabase hiccup) shouldn't crash the
   whole bot. Per-module code already handles its own errors
   gracefully where practical (news_filter, sl_manager, tp1_manager);
   the loop-level safety net catches anything that slipped through.

6. **Graceful shutdown via signal handlers.** SIGINT / SIGTERM flip a
   ``_stopped`` flag; the loop exits cleanly at the next iteration
   boundary, the MT5 connection is closed, and pending positions
   stay open on the broker side (we do not auto-close on shutdown —
   the operator's MT5 mobile app remains the manual override).

7. **Starting balance captured once at startup.** Daily halt uses the
   17:00 ET broker rollover; ``daily_halt`` module knows about that.
   For v1 we store one ``starting_balance`` at boot and never refresh
   it during the bot's run — the operator restarts the bot at
   rollover. v1.1 should re-read on rollover boundary (TODO in
   docstring of ``_check_daily_halt``).
"""

from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal
from uuid import UUID

import httpx
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from bot.data.ohlc_provider import OHLCProvider
from bot.execution.entry_trigger import EntryTrigger
from bot.execution.mt5_connector import MT5Connector
from bot.execution.order_manager import (
    OrderManagerConfig,
    place_layered_orders,
)
from bot.execution.position_tracker import PositionTracker
from bot.exits.sl_manager import SLManager, SLManagerConfig
from bot.exits.tp1_manager import TP1Manager
from bot.filters.news_filter import NewsFilter, NewsFilterConfig
from bot.logging.supabase_logger import (
    Setup,
    SupabaseLogger,
    ZoneInput,
)
from bot.risk.daily_halt import DailyHaltConfig, check_daily_halt
from bot.risk.exposure_check import check_exposure, count_active_setups
from bot.risk.position_sizing import (
    SizingConfig,
    SizingMode,
    calculate_lot_size,
)
from bot.strategy.strong_point import (
    StrongPointConfig,
    ValidatedZone,
    compute_sl_price,
)
from bot.strategy.pipeline import (
    StrategyPipelineConfig,
    run_strategy_pipeline,
)


# --------------------------------------------------------------------------- #
# Config + state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BotLoopConfig:
    """Cadence + assembly knobs for the orchestrator."""

    symbol: str = "XAUUSD"
    main_loop_sleep_ms: int = 1000
    """Per-iteration sleep. 1000 ms = ~1 Hz.

    M5 trading doesn't need sub-second responsiveness — bars only
    change every 5 minutes. The pre-2026-05 default of 100 ms hammered
    Supabase at ~18 queries/sec (entry_trigger + tp1_manager loops
    each call ``get_active_setups()``), tripping Supabase's HTTP/2
    max-requests-per-connection limit (~10K) every ~14 minutes and
    cycling the connection with a noisy traceback. 1 Hz is plenty for
    Layer 2/3 trigger detection on M5; the worst-case fill delay
    (~1 s) is negligible vs spread + slippage."""

    config_refresh_seconds: int = 30
    detect_closed_seconds: int = 30
    reconcile_seconds: int = 300
    heartbeat_seconds: int = 300

    max_simultaneous_setups: int = 3
    daily_loss_limit_pct: float = 10.0
    ohlc_count: int = 200
    ohlc_timeframe: Literal["M5"] = "M5"
    error_backoff_seconds: float = 5.0


@dataclass
class BotState:
    """Mutable per-run state."""

    starting_balance: float | None = None
    last_config_refresh: datetime | None = None
    last_detect_closed: datetime | None = None
    last_reconcile: datetime | None = None
    last_heartbeat: datetime | None = None
    last_m5_bar_time: pd.Timestamp | None = None
    kill_switch: bool = False
    pause_until: datetime | None = None
    fired_layer_count: int = 0
    tp1_count: int = 0
    placed_setup_count: int = 0
    iteration_count: int = 0


# --------------------------------------------------------------------------- #
# Bot
# --------------------------------------------------------------------------- #


class Bot:
    """The orchestrator. Owns every manager and the runtime state."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        config: BotLoopConfig | None = None,
        *,
        order_manager_config: OrderManagerConfig | None = None,
        sl_manager_config: SLManagerConfig | None = None,
        news_filter_config: NewsFilterConfig | None = None,
        strategy_pipeline_config: StrategyPipelineConfig | None = None,
        sizing_config: SizingConfig | None = None,
        daily_halt_config: DailyHaltConfig | None = None,
    ) -> None:
        self.mt5 = mt5
        self.supabase = supabase
        self.config = config or BotLoopConfig()

        # Per-manager configs — startup-time snapshot from bot_config.
        self.order_manager_config = order_manager_config or OrderManagerConfig()
        self.sl_manager_config = sl_manager_config or SLManagerConfig()
        self.news_filter_config = news_filter_config or NewsFilterConfig()
        self.strategy_pipeline_config = (
            strategy_pipeline_config or StrategyPipelineConfig()
        )
        self.sizing_config = sizing_config or SizingConfig()
        self.daily_halt_config = daily_halt_config or DailyHaltConfig()

        # Managers.
        self.position_tracker = PositionTracker(mt5, supabase)
        self.ohlc_provider = OHLCProvider(mt5)
        self.tp1_manager = TP1Manager(mt5, supabase, self.position_tracker)
        self.sl_manager = SLManager(
            mt5, supabase, config=self.sl_manager_config,
        )
        self.entry_trigger = EntryTrigger(
            mt5, supabase, self.position_tracker,
        )
        self.news_filter = NewsFilter(
            supabase, config=self.news_filter_config,
        )

        self.state = BotState()
        self._stopped = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Connect to MT5, capture starting balance, reconcile."""
        self.mt5.connect()
        try:
            self.state.starting_balance = self.mt5.get_balance()
            logger.info(
                f"bot startup: balance=${self.state.starting_balance:.2f}"
            )
        except Exception:
            logger.exception(
                "bot startup: get_balance failed; daily halt will be inactive "
                "until first successful balance read"
            )
        try:
            recon = self.position_tracker.reconcile_with_mt5()
            logger.info(
                f"bot startup reconcile: matched={recon.matched_count} "
                f"ghost={len(recon.ghost_tickets)} "
                f"lost={len(recon.lost_trade_ids)}"
            )
        except Exception:
            logger.exception("bot startup: reconcile failed; continuing")

        now = datetime.now(timezone.utc)
        self.state.last_reconcile = now
        self._refresh_runtime_config(now)

    def shutdown(self) -> None:
        """Best-effort disconnect. Idempotent. Never raises."""
        try:
            self.mt5.disconnect()
        except Exception:
            logger.exception("shutdown: MT5 disconnect failed (non-fatal)")

    def stop(self) -> None:
        """Flag the loop to exit at the next iteration boundary."""
        logger.info("stop signal received")
        self._stopped = True

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Drive the main loop until ``stop()`` is called.

        ``initialize()`` is inside the try/finally so the MT5 disconnect
        in ``shutdown()`` always runs, even if startup itself failed
        partway (e.g. balance read OK but reconcile blew up).
        """
        sleep_sec = self.config.main_loop_sleep_ms / 1000.0
        try:
            self.initialize()
            while not self._stopped:
                now = datetime.now(timezone.utc)
                try:
                    self.run_iteration(now)
                except Exception:
                    logger.exception(
                        f"main loop iteration failed; "
                        f"backoff {self.config.error_backoff_seconds}s"
                    )
                    time.sleep(self.config.error_backoff_seconds)
                    continue
                time.sleep(sleep_sec)
        finally:
            self.shutdown()

    def run_iteration(self, now: datetime) -> None:
        """Single loop iteration. Testable independently of timing."""
        self.state.iteration_count += 1

        # 1. Live config refresh (kill switch + pause-until).
        if self._should_refresh_config(now):
            self._refresh_runtime_config(now)

        # 2. Current price.
        try:
            tick = self.mt5.get_current_price(self.config.symbol)
            bid = float(tick["bid"])
            ask = float(tick["ask"])
        except Exception:
            logger.exception("main loop: get_current_price failed")
            return

        # 3. Per-tick: entry triggers + TP1 checks.
        try:
            fired = self.entry_trigger.check_live(bid, ask)
            self.state.fired_layer_count += len(fired)
        except Exception:
            logger.exception("main loop: entry_trigger.check_live failed")

        for setup in self._safe_get_active_setups():
            if setup.status != "ACTIVE":
                continue  # PENDING needs Layer 1 to fill first; TP1_HIT is past
            try:
                tp1_result = self.tp1_manager.check(setup, bid, ask)
                if tp1_result.triggered:
                    self.state.tp1_count += 1
                    logger.info(
                        f"TP1 fired: setup={setup.id} closed_lots="
                        f"{tp1_result.closed_lots} new_sl="
                        f"{tp1_result.new_sl_price}"
                    )
                    if tp1_result.sl_modify_pending:
                        logger.error(
                            f"TP1 SL→BE pending retry on setup {setup.id} "
                            f"(modify_order failed)"
                        )
            except Exception:
                logger.exception(
                    f"main loop: tp1_manager.check failed for setup {setup.id}"
                )

        # 4. Detect externally-closed positions (cadenced).
        if self._should_detect_closed(now):
            for setup in self._safe_get_active_setups():
                try:
                    self.position_tracker.detect_closed_positions(setup)
                except Exception:
                    logger.exception(
                        f"main loop: detect_closed_positions failed for "
                        f"{setup.id}"
                    )
            self.state.last_detect_closed = now

        # 5. New-setup detection (gated by halt + news + kill + M5 close).
        if self._is_paused(now):
            pass  # skip new setups; existing managed above
        else:
            daily_halt_result = self._check_daily_halt(now)
            news_check = self.news_filter.check(now)

            if daily_halt_result.is_halted:
                logger.debug(
                    f"new setups blocked: daily halt "
                    f"(drawdown {daily_halt_result.current_drawdown_pct:.2f}%, "
                    f"threshold {daily_halt_result.threshold_pct:.2f}%)"
                )
            elif news_check.is_blocked:
                logger.debug(
                    f"new setups blocked: news → {news_check.block_reason}"
                )
            elif self._has_new_m5_close(now):
                self._maybe_run_strategy(now)

        # 6. Reconcile (cadenced).
        if self._should_reconcile(now):
            try:
                self.position_tracker.reconcile_with_mt5()
            except Exception:
                logger.exception("main loop: reconcile_with_mt5 failed")
            self.state.last_reconcile = now

        # 7. Heartbeat.
        if self._should_heartbeat(now):
            self._emit_heartbeat(now)
            self.state.last_heartbeat = now

    # ------------------------------------------------------------------ #
    # Cadence helpers
    # ------------------------------------------------------------------ #

    def _should_refresh_config(self, now: datetime) -> bool:
        return _elapsed(self.state.last_config_refresh, now) >= (
            self.config.config_refresh_seconds
        )

    def _should_detect_closed(self, now: datetime) -> bool:
        return _elapsed(self.state.last_detect_closed, now) >= (
            self.config.detect_closed_seconds
        )

    def _should_reconcile(self, now: datetime) -> bool:
        return _elapsed(self.state.last_reconcile, now) >= (
            self.config.reconcile_seconds
        )

    def _should_heartbeat(self, now: datetime) -> bool:
        return _elapsed(self.state.last_heartbeat, now) >= (
            self.config.heartbeat_seconds
        )

    # ------------------------------------------------------------------ #
    # Pause / halt / runtime config
    # ------------------------------------------------------------------ #

    def _refresh_runtime_config(self, now: datetime) -> None:
        """Re-read kill_switch + pause_until from bot_config. Best-effort."""
        try:
            self.state.kill_switch = bool(
                self.supabase.check_bot_config("kill_switch")
            )
        except Exception:
            logger.exception(
                "config refresh: kill_switch read failed (using last value)"
            )

        try:
            raw = self.supabase.check_bot_config("pause_until")
            self.state.pause_until = _parse_pause_until(raw)
        except Exception:
            logger.exception(
                "config refresh: pause_until read failed (using last value)"
            )

        self.state.last_config_refresh = now

    def _is_paused(self, now: datetime) -> bool:
        """True iff kill_switch is on or ``now`` is before pause_until."""
        if self.state.kill_switch:
            return True
        if self.state.pause_until is not None:
            return now < self.state.pause_until
        return False

    def _check_daily_halt(self, now: datetime):
        """Daily halt guard.

        TODO v1.1: re-read ``starting_balance`` on the 17:00 ET rollover
        boundary so the same bot run survives multiple trading days.
        """
        starting = self.state.starting_balance
        if starting is None:
            try:
                self.state.starting_balance = self.mt5.get_balance()
                starting = self.state.starting_balance
            except Exception:
                logger.exception(
                    "daily halt: get_balance failed; treating as not halted"
                )
                return _NotHalted()

        try:
            current = self.mt5.get_balance()
        except Exception:
            logger.exception(
                "daily halt: get_balance failed; treating as not halted"
            )
            return _NotHalted()

        return check_daily_halt(
            starting_balance=starting,
            current_balance=current,
            config=self.daily_halt_config,
            now=now,
        )

    # ------------------------------------------------------------------ #
    # Strategy / new setups
    # ------------------------------------------------------------------ #

    def _has_new_m5_close(self, now: datetime) -> bool:
        """True iff the most recent M5 bar's timestamp has advanced.

        Side effect: fetches OHLC and stores it on the bot for re-use
        within this iteration. We use the cache via OHLCProvider so
        repeated reads are cheap; the gating is mainly to avoid
        re-running the strategy pipeline on every tick.
        """
        try:
            df = self.ohlc_provider.get(
                self.config.symbol,
                self.config.ohlc_timeframe,
                self.config.ohlc_count,
            )
        except Exception:
            logger.exception("main loop: OHLC fetch failed")
            return False

        if len(df) == 0:
            return False
        latest = df.index[-1]
        if (
            self.state.last_m5_bar_time is None
            or latest > self.state.last_m5_bar_time
        ):
            self.state.last_m5_bar_time = latest
            self._latest_ohlc = df  # type: ignore[attr-defined]
            return True
        return False

    def _maybe_run_strategy(self, now: datetime) -> None:
        """Run the strategy pipeline + try to place each detected zone."""
        df: pd.DataFrame | None = getattr(self, "_latest_ohlc", None)
        if df is None:
            return

        try:
            zones = run_strategy_pipeline(df, self.strategy_pipeline_config)
        except Exception:
            logger.exception("main loop: strategy pipeline raised")
            return

        if not zones:
            return

        # Exposure cap.
        try:
            active_setups = self.position_tracker.get_active_setups()
        except Exception:
            logger.exception("main loop: get_active_setups failed")
            return
        active_count = count_active_setups(active_setups)

        for zone in zones:
            exp = check_exposure(
                active_count=active_count,
                max_simultaneous=self.config.max_simultaneous_setups,
                with_candidate=True,
            )
            if not exp.can_open_new:
                logger.info(
                    f"new setups blocked by exposure cap "
                    f"({exp.current_count}/{exp.max_allowed})"
                )
                break
            if self._try_place_setup(zone, df):
                active_count += 1
                self.state.placed_setup_count += 1

    def _try_place_setup(
        self, zone: ValidatedZone, ohlc_df: pd.DataFrame,
    ) -> bool:
        """Compute SL + lot size, validate, log zone, place orders.

        Returns True iff ``order_manager.place_layered_orders`` reported
        ``PLACED`` (not FAILED, not SKIPPED-via-gap).
        """
        # 1. SL — pinned to the strategy-layer sl_anchor_swing.
        if zone.sl_anchor_swing is None:
            logger.warning(
                "new setup skipped: zone has no sl_anchor_swing — "
                "Strong Point validation should have set this"
            )
            return False
        sp_cfg = StrongPointConfig(
            sl_buffer_points=self.sl_manager_config.sl_buffer_points,
        )
        sl_price = compute_sl_price(zone, sp_cfg)

        entry_price = zone.top if zone.direction == "BUY" else zone.bottom
        sl_validation = self.sl_manager.validate_sl_distance(
            entry_price=entry_price,
            sl_price=sl_price,
            direction=zone.direction,
        )
        if not sl_validation.is_valid:
            logger.warning(
                f"new setup skipped: SL validation failed — "
                f"{sl_validation.error}"
            )
            return False

        # 2. Lot size.
        try:
            balance = self.mt5.get_balance()
        except Exception:
            logger.exception("new setup: get_balance failed; skipping zone")
            return False
        lot_result = calculate_lot_size(
            balance=balance,
            entry_price=entry_price,
            sl_price=sl_price,
            config=self.sizing_config,
        )
        if lot_result.lot_size <= 0:
            logger.warning(
                f"new setup skipped: zero lot size ({lot_result.reason})"
            )
            return False

        # 3. Persist the zone (so the setup can FK to it).
        try:
            zone_row = self.supabase.log_zone(_zone_to_input(zone))
            zone_id = UUID(str(zone_row["id"]))
        except Exception:
            logger.exception("new setup: log_zone failed; skipping zone")
            return False

        # 4. Place orders.
        try:
            result = place_layered_orders(
                zone, zone_id,
                lot_size=lot_result.lot_size,
                sl_price=sl_price,
                mt5=self.mt5, supabase=self.supabase,
                config=self.order_manager_config,
            )
        except Exception:
            logger.exception("new setup: place_layered_orders raised")
            return False

        if result.status == "PLACED":
            # New setup written to Supabase via order_manager — that
            # path bypasses position_tracker, so the active-setups
            # cache won't see the new row until its TTL expires.
            # Invalidate now so entry_trigger picks it up on the next
            # tick rather than waiting up to 5 s.
            self.position_tracker.invalidate_active_setups_cache()
            logger.info(
                f"new setup placed: id={result.setup_id} "
                f"direction={zone.direction} "
                f"layer1_ticket={result.layer_1_ticket} "
                f"sl={result.sl_price} tp1={result.tp1_price}"
            )
            return True
        logger.warning(
            f"new setup not placed: status={result.status} "
            f"errors={result.error_messages}"
        )
        return False

    # ------------------------------------------------------------------ #
    # Heartbeat / observability
    # ------------------------------------------------------------------ #

    def _emit_heartbeat(self, now: datetime) -> None:
        try:
            balance = self.mt5.get_balance()
        except Exception:
            balance = float("nan")
        active = self._safe_get_active_setups()
        msg = (
            f"heartbeat: balance={balance:.2f} active_setups={len(active)} "
            f"layers_fired={self.state.fired_layer_count} "
            f"tp1_hits={self.state.tp1_count} "
            f"placed={self.state.placed_setup_count} "
            f"iters={self.state.iteration_count} "
            f"paused={self._is_paused(now)}"
        )
        logger.info(msg)
        try:
            self.supabase.log_event(
                "INFO",
                "heartbeat",
                context={
                    "balance": balance,
                    "active_setups": len(active),
                    "layers_fired": self.state.fired_layer_count,
                    "tp1_hits": self.state.tp1_count,
                    "placed_setups": self.state.placed_setup_count,
                    "iterations": self.state.iteration_count,
                    "paused": self._is_paused(now),
                },
            )
        except Exception:
            logger.exception("heartbeat: bot_logs write failed (non-fatal)")

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _safe_get_active_setups(self) -> list[Setup]:
        """Read the active-setups list; return ``[]`` on any failure.

        Transient httpx errors (e.g. Supabase cycling the HTTP/2
        connection at its 10K-request limit) are logged briefly at
        WARN level — the next iteration will reconnect. Unexpected
        exceptions still get the full traceback so genuine bugs
        surface.
        """
        try:
            return self.position_tracker.get_active_setups()
        except httpx.RequestError as e:
            # Transport-layer (connection cycled, timeout, network
            # blip). Recoverable. Don't dump the full traceback every
            # ~15 min as the HTTP/2 connection rotates.
            logger.warning(
                "get_active_setups: transient transport error "
                "({}: {}); returning empty, next iteration will reconnect",
                type(e).__name__, e,
            )
            return []
        except Exception:
            logger.exception("get_active_setups failed; returning empty")
            return []


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _elapsed(last: datetime | None, now: datetime) -> float:
    """Seconds since ``last``; ``inf`` when ``last`` is None."""
    if last is None:
        return float("inf")
    return (now - last).total_seconds()


def _parse_pause_until(raw: object) -> datetime | None:
    """Parse the ``pause_until`` JSONB value into a UTC datetime or None."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _zone_to_input(zone: ValidatedZone) -> ZoneInput:
    """Project an in-memory zone into the ``zones`` table insert payload.

    v1 only handles Strong Point setups → ``zone_type=STRONG_POINT``.
    ``pattern_type`` persists the real S&D code (RBR / DBD / DBR /
    RBD) so demo-trading analytics can compare continuation vs
    reversal patterns. The CHECK constraint accepts these codes
    as of migration 006.
    """
    return ZoneInput(
        symbol="XAUUSD",
        direction=zone.direction,
        zone_type="STRONG_POINT",
        pattern_type=zone.source_pattern.pattern_type.value,
        top=Decimal(str(zone.top)),
        bottom=Decimal(str(zone.bottom)),
        approach_count=0,            # Imbalance-only field; 0 in v1
        qualified_imbalance_at=None,  # Imbalance-only field; None in v1
        formed_at=zone.formed_at.to_pydatetime(),
    )


@dataclass(frozen=True)
class _NotHalted:
    """Tiny stand-in for DailyHaltResult when balance can't be read.

    Avoids the daily-halt module's stricter input requirements when we
    only need ``is_halted=False`` to fall through. Mirrors just enough
    of the public surface (``is_halted`` + the two fields the debug
    logger formats) for the call sites here.
    """

    is_halted: bool = False
    current_drawdown_pct: float = 0.0
    threshold_pct: float = 0.0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """``python -m bot.main`` entry point.

    Loads ``.env`` from the current working directory (or any parent)
    BEFORE constructing the Bot, so ``MT5Connector.from_env()`` and
    ``SupabaseLogger.from_env()`` see the env vars. dotenv discovery
    walks upward to the filesystem root so running from a sub-
    directory still works.

    ``load_dotenv()`` is a no-op when no ``.env`` is found — env
    vars from the shell still take precedence (override=False by
    default), which is the right behaviour for CI / Docker / VPS
    where vars come from the orchestration layer.
    """
    load_dotenv()
    bot = Bot(
        mt5=MT5Connector.from_env(),
        supabase=SupabaseLogger.from_env(),
    )

    def _handle_signal(signum: int, frame: object) -> None:
        logger.info(f"signal {signum} received; requesting graceful stop")
        bot.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bot.run()


if __name__ == "__main__":
    main()
