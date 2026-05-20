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

import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Literal
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
from bot.exits.tp_manager import TPManager
from bot.exits.trailing_stop_manager import TrailingStopManager
from bot.exits.zone_exit_manager import ZoneExitManager
from bot.signals.next_signal_writer import NextSignalWriter
from bot.filters.news_filter import NewsFilter, NewsFilterConfig
from bot.logging.supabase_logger import (
    Setup,
    SupabaseLogger,
    Zone,
    ZoneInput,
)
from bot.risk.daily_halt import DailyHaltConfig, check_daily_halt
from bot.risk.exposure_check import check_exposure, count_active_setups
from bot.risk.position_sizing import (
    SizingConfig,
    SizingMode,
    calculate_lot_size,
)
from bot.strategy.pattern_detection import (
    Base as _Base,
    Impulse as _Impulse,
    Pattern as _Pattern,
    PatternType as _PT,
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
from bot.strategy.tp_target import find_nearest_local_peak
from bot.strategy.zone_marking import Zone as _ZoneShape
from bot.strategy.zone_refinement import RefinedZone as _RefinedZone
from bot.strategy.zone_visibility import count_bodies_through_zone
from bot.visualization.zone_snapshot import ZoneSnapshotWriter
from bot.strategy.zone_lifecycle import (
    SKIP_NEW_SETUP_STATUSES,
    ZoneRef,
    check_consumption,
    check_flip,
    check_violation,
    flipped_zone_body_broken_since_flip,
    log_transition,
    zone_bounds_overlap,
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
    ohlc_count: int = 1000
    """Number of M5 bars to pull per pipeline run. 1000 ≈ 3.5 days of
    history — wide enough that recent-but-not-current zones (24-48h
    old) are still detectable. Pre-2026-05 default was 200 (~16 h),
    which excluded zones the user trades manually."""
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
        self.tp_manager = TPManager(mt5, supabase, self.position_tracker)
        self.zone_exit_manager = ZoneExitManager(
            mt5, supabase, self.position_tracker,
        )
        self.trailing_stop_manager = TrailingStopManager(
            mt5, supabase, self.position_tracker,
        )
        # PR #51: dashboard signal writer. Updates the ``signals`` table
        # on every M5 close so the Vercel-hosted dashboard can show
        # the closest BUY + SELL zones the bot is currently watching.
        # Failures here NEVER affect trading — the writer is wrapped
        # again at the call site.
        self.next_signal_writer = NextSignalWriter(supabase)
        # PR #49: zone visualization on the MT5 chart via a CSV
        # sidecar file the MQL5 EA reads. No-op if MT5_FILES_DIR is
        # unset (e.g. dev machines without MT5).
        self.zone_snapshot_writer = ZoneSnapshotWriter(
            os.environ.get("MT5_FILES_DIR"),
        )
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

        # Idempotency cache for log_zone. Key is
        # ``ZoneKey = (direction, top_2dp, bottom_2dp, formed_at_iso)``.
        # Value is the persisted zone row's UUID, so callers can FK to
        # it without re-querying. Populated on first detection and
        # rehydrated from DB on Bot.initialize() so a restart doesn't
        # cause duplicate rows. 2-decimal-place rounding matches the
        # ``NUMERIC(10,2)`` precision of ``zones.top`` / ``zones.bottom``
        # — avoids cache misses from float jitter.
        self._persisted_zone_keys: dict[
            tuple[str, float, float, str], UUID
        ] = {}

        # PR #44: pending zones awaiting their first price retest.
        # Populated on M5 close by ``_detect_new_zones``; iterated on
        # every tick by ``_try_place_pending``. Entries pop on
        # successful placement; otherwise survive until the next M5
        # close refreshes the list (whatever the new pipeline output
        # plus current FLIPPED rows produce).
        self._pending_pipeline_zones: list[tuple[ValidatedZone, UUID]] = []
        self._pending_flipped_zones:  list[tuple[ValidatedZone, UUID]] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Connect to MT5, capture starting balance, reconcile."""
        # ~3.5 days of M5 history at the default. Logged so the
        # operator can confirm at restart without grepping config.
        logger.info(
            f"OHLC fetch window: {self.config.ohlc_count} "
            f"{self.config.ohlc_timeframe} bars per pipeline run "
            f"(~{self.config.ohlc_count * 5 / 60 / 24:.1f} days history)"
        )
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

        # Rehydrate the persisted-zone idempotency cache from DB so a
        # restart doesn't insert duplicate rows for zones that already
        # exist. Includes terminal states (FLIPPED) — they still have
        # a row, so a re-detection of the same physical zone (very
        # unlikely but possible) wouldn't re-insert.
        try:
            self._hydrate_persisted_zone_cache()
        except Exception:
            logger.exception(
                "bot startup: zone-cache hydration failed; will run with "
                "an empty cache (may re-insert duplicates until lifecycle "
                "scanner reconciles)"
            )

        now = datetime.now(timezone.utc)
        self.state.last_reconcile = now
        self._refresh_runtime_config(now)

    def _hydrate_persisted_zone_cache(self) -> None:
        """Load every existing zone row → key cache.

        One query, all statuses. Allows the bot to crash + restart
        without re-inserting zones it already persisted.
        """
        existing = self.supabase.get_zones_by_status(
            ["CONFIRMED", "ACTIVE", "CONSUMED", "VIOLATED", "FLIPPED"],
        )
        for z in existing:
            key = _zone_key_from_row(z.direction, z.top, z.bottom, z.formed_at)
            self._persisted_zone_keys[key] = z.id
        logger.info(
            f"bot startup: hydrated zone cache from DB "
            f"({len(self._persisted_zone_keys)} existing zones)"
        )

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

        # 3. Per-tick: entry triggers + per-layer TP checks.
        try:
            fired = self.entry_trigger.check_live(bid, ask)
            self.state.fired_layer_count += len(fired)
        except Exception:
            logger.exception("main loop: entry_trigger.check_live failed")

        for setup in self._safe_get_active_setups():
            if setup.status != "ACTIVE":
                # PENDING: Layer 1 not yet filled. TP1_HIT: legacy state
                # from the pre-PR-41 cascade — no new transitions to it,
                # but old rows still need skipping.
                continue
            try:
                closures = self.tp_manager.check(setup, bid, ask)
            except Exception:
                logger.exception(
                    f"main loop: tp_manager.check failed for setup {setup.id}"
                )
                continue
            for close in closures:
                self.state.tp1_count += 1
                logger.info(
                    f"TP{close.layer_number} fired: setup={close.setup_id} "
                    f"close_price={close.close_price} "
                    f"cascaded_sl={close.cascaded_sl}"
                )
                if close.needs_next_tp_recompute:
                    self._maybe_recompute_next_tp(setup, close.layer_number)

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
                self._detect_new_zones(now)
                # PR #47: zone-exit BE on the same M5 close. Runs AFTER
                # detection so freshly-CONSUMED zones (this bar's
                # lifecycle pass inside _detect_new_zones) are already
                # accounted for. Setups still ACTIVE here are the ones
                # whose body-close-out-of-zone should fire.
                self._run_zone_exit_pass(bid=bid, ask=ask)
                # Trailing-stop pass: locks in progressive profit on
                # L1-only setups. Runs AFTER zone-exit so the
                # comparison is against the (possibly just-BE'd) SL,
                # and only tightens further when the 30 % activation
                # threshold is reached.
                self._run_trailing_stop_pass(bid=bid, ask=ask)
                # PR #51: refresh the dashboard signal snapshot.
                # Strictly operator-facing — wrapped so any failure
                # is logged and the trading loop continues.
                self._run_next_signal_pass(bid=bid, ask=ask)

        # 5a. PR #48: per-tick zone-death pass — cancel WAITING layers
        # on active setups whose zones have been obscured (≥1 candle
        # bodied through since formation). Runs every tick, but the
        # underlying body-cross count only changes on M5 close (the
        # df we use is the post-M5-close snapshot), so per-tick is
        # cheap repeated work that catches state changes promptly.
        if not self._is_paused(now):
            try:
                self._run_zone_death_pass()
            except Exception:
                logger.exception("main loop: _run_zone_death_pass raised")

        # 5b. Per-tick placement check (PR #44). Runs AFTER M5-close
        # detection so the same iteration that detects a zone can
        # also place a setup on it if price is already at the zone.
        # On subsequent ticks (1 Hz) this still runs, catching mid-
        # bar wicks before the next M5 close's lifecycle scan
        # consumes the zone.
        if not self._is_paused(now):
            try:
                self._try_place_pending(bid=bid, ask=ask)
            except Exception:
                logger.exception("main loop: _try_place_pending raised")

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

    def _detect_new_zones(self, now: datetime) -> None:
        """M5-close detection pass: lifecycle scan + pattern detection
        + persistence + pending-list refresh.

        Per PR #44, placement does **not** happen here. Instead the
        candidates are stored on ``self._pending_pipeline_zones`` /
        ``self._pending_flipped_zones`` and the per-tick
        :meth:`_try_place_pending` decides when to fire them. That
        change means a wick into a zone fires placement immediately
        (1 Hz latency) instead of waiting for the next M5 close —
        which used to race against the lifecycle scanner's
        consumption check and lose every time.
        """
        df: pd.DataFrame | None = getattr(self, "_latest_ohlc", None)
        if df is None:
            return

        # Lifecycle pass first: consumption / violation / flip on the
        # just-closed M5 bar, BEFORE detecting new Strong Points. This
        # ensures the dedup pre-flight in _try_place_setup sees the
        # freshly-CONSUMED state from this bar.
        try:
            self._run_zone_lifecycle(df)
        except Exception:
            logger.exception("main loop: zone lifecycle pass raised")
            # Continue — pattern detection isn't impacted by lifecycle errors.

        try:
            zones = run_strategy_pipeline(df, self.strategy_pipeline_config)
        except Exception:
            logger.exception("main loop: strategy pipeline raised")
            return

        # Persist every tradeable zone to Supabase. Each zone gets
        # exactly one row, regardless of how many times it's
        # re-detected in subsequent pipeline runs (idempotency via
        # ``_persisted_zone_keys`` cache). This populates the data
        # trail the lifecycle scanner and dedup pre-flight rely on.
        zone_ids: dict[int, UUID] = {}
        for zone in zones:
            if not zone.refined_zone.is_tradeable:
                continue
            zid = self._persist_zone_if_new(zone)
            if zid is not None:
                zone_ids[id(zone)] = zid

        # SnD Flip side path (PR #38): load FLIPPED zones from DB.
        flipped_candidates = self._load_flipped_candidates(df)

        # Refresh the per-tick placement queues. Each tuple is
        # (zone_view, zone_id). ``_try_place_pending`` pops entries
        # on successful placement; what remains rolls into the next
        # M5 close where this method runs again and rebuilds.
        new_pipeline_pending: list[tuple[ValidatedZone, UUID]] = []
        for zone in zones:
            if not zone.is_strong_point:
                # Body-broken zones persisted but not traded; lifecycle
                # transitions will catch up on subsequent bars.
                continue
            zid = zone_ids.get(id(zone))
            if zid is None:
                logger.warning(
                    "skipping placement: zone wasn't persisted "
                    f"({zone.direction} {zone.bottom:.2f}-{zone.top:.2f})"
                )
                continue
            new_pipeline_pending.append((zone, zid))

        # PR #55: also pull CONFIRMED zones from Supabase that the
        # pipeline didn't re-detect this close. Without this, a zone
        # alive in the DB silently drops out of the placement queue
        # whenever the pipeline run doesn't reproduce it (newer
        # pattern overshadows, lookback aging, validation drift). The
        # signal writer pulls from Supabase and would still show the
        # zone — operator sees "next signal here", bot does nothing
        # because the in-memory queue is empty. Production-observed
        # failure case: BUY 4486.84 zone alive in DB, price drops
        # cleanly through it, no entry, no setup row, no error.
        already_queued = {zid for (_, zid) in new_pipeline_pending}
        extra_confirmed = self._load_confirmed_candidates(
            df, exclude_zone_ids=already_queued,
        )
        new_pipeline_pending.extend(extra_confirmed)

        self._pending_pipeline_zones = new_pipeline_pending
        self._pending_flipped_zones = list(flipped_candidates)
        if self._pending_pipeline_zones or self._pending_flipped_zones:
            logger.debug(
                f"M5 close: pending placement queue refreshed — "
                f"{len(self._pending_pipeline_zones)} pipeline "
                f"({len(extra_confirmed)} from DB), "
                f"{len(self._pending_flipped_zones)} flipped"
            )

        # PR #49: refresh MT5 chart visualization snapshot. No-op if
        # MT5_FILES_DIR isn't set. Wrapped in try/except — viz failure
        # must NEVER affect the trading loop.
        if self.zone_snapshot_writer.enabled:
            try:
                self._refresh_chart_zone_snapshot(df)
            except Exception:
                logger.exception(
                    "zone snapshot write raised; visualization will "
                    "be stale until the next M5 close"
                )

    def _refresh_chart_zone_snapshot(self, df: pd.DataFrame) -> None:
        """Pull renderable zones from Supabase + push to the CSV file.

        Renderable = CONFIRMED / ACTIVE / FLIPPED (CONSUMED / VIOLATED
        are dead and would clutter the chart). Distance, age, and
        visibility filters are applied inside the writer.
        """
        try:
            zones = self.supabase.get_zones_by_status(
                ["CONFIRMED", "ACTIVE", "FLIPPED"],
            )
        except Exception:
            logger.warning(
                "zone snapshot: get_zones_by_status failed; skipping "
                "this refresh"
            )
            return
        if len(df) == 0:
            return
        current_price = float(df.iloc[-1]["close"])
        self.zone_snapshot_writer.write(
            zones, current_price=current_price, df=df,
        )

    def _run_zone_exit_pass(self, *, bid: float, ask: float) -> None:
        """PR #47 + PR #52: per-setup body-close-out-of-zone BE trigger.

        Runs on every M5 close. For each ACTIVE setup, the manager
        checks whether the last N M5 bars (N = ``confirmation_candles``,
        default 2 per PR #52) have all body-closed past L1 in the
        profit direction (BUY: close > zone.top; SELL: close <
        zone.bottom). Two consecutive bars sustained outside the zone
        is much harder to fake than one — filters out the liquidity
        grabs / stop hunts / news spikes the single-bar version used
        to BE on prematurely. On fire:

        * Close the shallowest still-FILLED layer at the current
          bid/ask (``close_reason='ZONE_EXIT'``).
        * Modify SL to entry on every remaining FILLED layer.
        * Cancel every still-WAITING layer
          (``close_reason='ZONE_EXIT_CANCELLED'``).

        Special case: if only one layer is FILLED, BE-only (no close).
        Idempotent — the manager detects an already-BE'd setup from
        live trade state, so re-firing on subsequent closes is a no-op.
        """
        df: pd.DataFrame | None = getattr(self, "_latest_ohlc", None)
        if df is None or len(df) == 0:
            return
        # Pass the last several closes; the manager picks the trailing
        # N it needs based on its config. Slicing the last 10 is plenty
        # (we only ever look at the last ~3 in practice).
        recent_closes = [
            float(c) for c in df["close"].iloc[-10:].tolist()
        ]

        for setup in self._safe_get_active_setups():
            if setup.status != "ACTIVE":
                continue
            try:
                result = self.zone_exit_manager.check(
                    setup, recent_closes=recent_closes,
                    bid=bid, ask=ask,
                )
            except Exception:
                logger.exception(
                    f"main loop: zone_exit_manager.check failed for "
                    f"setup {setup.id}"
                )
                continue
            if result is None:
                continue
            logger.info(
                f"ZONE_EXIT fired: setup={result.setup_id} "
                f"close={result.close_price} "
                f"closed_layer={result.closed_layer} "
                f"be_layers={result.be_layer_count} "
                f"cancelled_waiting={result.cancelled_waiting_count}"
                + (f" error={result.error}" if result.error else "")
            )

    def _run_trailing_stop_pass(self, *, bid: float, ask: float) -> None:
        """Trailing-stop SL on L1-only setups.

        Runs on every M5 close right after the zone-exit pass. For
        each ACTIVE setup where Layer 1 is the ONLY filled layer, the
        manager checks whether current profit has crossed the
        activation threshold (default 30 % of distance to TP1) and,
        if so, tightens the SL to lock in a fraction of the profit
        (default 50 %). Multi-layer setups are skipped here — the
        TP-cascade (PR #41) and zone-exit (PR #47) own those.

        The manager is idempotent and monotonic: re-firing on a
        setup whose SL is already at-or-better than the trailing
        calc produces no broker call. If profit retraces, the SL
        does NOT move back — only ever tightens.
        """
        for setup in self._safe_get_active_setups():
            if setup.status != "ACTIVE":
                continue
            try:
                result = self.trailing_stop_manager.check(
                    setup, bid=bid, ask=ask,
                )
            except Exception:
                logger.exception(
                    f"main loop: trailing_stop_manager.check failed for "
                    f"setup {setup.id}"
                )
                continue
            if result is None:
                continue
            logger.info(
                f"TRAILING_STOP moved: setup={result.setup_id} "
                f"trade={result.trade_id} "
                f"old_sl={result.old_sl} new_sl={result.new_sl} "
                f"price={result.current_price} "
                f"profit={result.current_profit:.2f}"
                + (f" error={result.error}" if result.error else "")
            )

    def _run_next_signal_pass(self, *, bid: float, ask: float) -> None:
        """PR #51: refresh the dashboard signal snapshot.

        Computes the closest pending BUY + SELL zone the bot is
        watching and writes them to ``signals`` so the Vercel
        dashboard can render them. The writer itself wraps every
        Supabase / strategy call in a try/except; this outer
        try/except is a defensive belt — under no circumstances
        may a dashboard refresh affect the trading loop.
        """
        df: pd.DataFrame | None = getattr(self, "_latest_ohlc", None)
        if df is None or len(df) == 0:
            return
        try:
            outcome = self.next_signal_writer.write(df, bid=bid, ask=ask)
        except Exception:
            logger.exception(
                "next-signal writer raised; dashboard will be stale "
                "until the next M5 close"
            )
            return
        logger.debug(
            f"next-signal: BUY={outcome.get('BUY')} "
            f"SELL={outcome.get('SELL')}"
        )

    def _run_zone_death_pass(self) -> None:
        """PR #48: per-tick zone-visibility check for ACTIVE setups.

        Scans every ACTIVE setup. For each, resolves its zone bounds
        + formation timestamp and runs the visibility count. If 1+
        candles have bodied through the zone since formation, the
        zone is dead — cancel every still-WAITING layer with
        ``close_reason='ZONE_DEAD_CANCELLED'``.

        FILLED layers are left untouched here. They have their own
        protection paths:
        * PR #47's zone-exit BE fires on body-close out of zone
          (M5 close), closing the shallowest filled layer and BE'ing
          the rest.
        * Their original SL still protects against runaway losses.
        * TP1/TP2/TP3 still fire if reached.

        The visibility check is read-only on Supabase except for the
        cancellation update_trade_status calls; the body-cross count
        is a pure function on the in-memory df.
        """
        df: pd.DataFrame | None = getattr(self, "_latest_ohlc", None)
        if df is None or len(df) == 0:
            return

        for setup in self._safe_get_active_setups():
            if setup.status != "ACTIVE":
                continue

            # Resolve the zone for this setup. zone_id is the FK on
            # the setup row; the zone carries the canonical
            # top/bottom/formed_at the visibility check needs. If the
            # read fails (transient Supabase issue), skip this setup
            # — next tick re-evaluates.
            try:
                zone_row = self.supabase.get_zone_by_id(setup.zone_id)
            except Exception:
                logger.exception(
                    f"zone-death pass: get_zone_by_id failed for "
                    f"setup={setup.id} zone={setup.zone_id}"
                )
                continue
            if zone_row is None:
                logger.warning(
                    f"zone-death pass: zone {setup.zone_id} not found "
                    f"for setup {setup.id}; skipping"
                )
                continue

            bodies = count_bodies_through_zone(
                df,
                zone_top=float(zone_row.top),
                zone_bottom=float(zone_row.bottom),
                since_time=pd.Timestamp(zone_row.formed_at),
            )
            if bodies < 1:
                continue  # zone still visible — nothing to do

            # Zone is obscured. Cancel all WAITING layers on this setup.
            try:
                trades = self.supabase.get_trades_for_setup(setup.id)
            except Exception:
                logger.exception(
                    f"zone-death pass: get_trades_for_setup failed for "
                    f"setup={setup.id}"
                )
                continue
            waiting = [t for t in trades if t.status == "WAITING"]
            if not waiting:
                continue  # nothing to cancel; FILLED layers protected
                          # by other paths

            for trade in waiting:
                try:
                    self.position_tracker.update_trade_status(
                        trade.id,
                        "CANCELLED",
                        close_reason="ZONE_DEAD_CANCELLED",
                    )
                except Exception:
                    logger.exception(
                        f"zone-death pass: WAITING layer "
                        f"{trade.layer_number} cancel failed "
                        f"for setup {setup.id}"
                    )
                    continue
            logger.info(
                f"ZONE_DEAD_CANCELLED: setup={setup.id} "
                f"zone={zone_row.direction} "
                f"{float(zone_row.bottom):.2f}-{float(zone_row.top):.2f} "
                f"({bodies} bodies through since "
                f"formed_at={zone_row.formed_at}); "
                f"cancelled {len(waiting)} WAITING layer(s)"
            )

    def _try_place_pending(self, *, bid: float, ask: float) -> None:
        """Per-tick placement check (PR #44).

        Iterates the pending pipeline + flipped queues against the
        current tick. For each entry, runs the existing
        :meth:`_try_place_setup` — its price-vs-zone gate (PR #40)
        decides whether to actually fire. Successful placements pop
        from the queue so the next tick doesn't re-attempt.

        Heavy strategy work (pattern detection, persistence,
        lifecycle scan) stays on M5 close in
        :meth:`_detect_new_zones` — only the cheap "is price at this
        zone yet?" check runs here.
        """
        if not self._pending_pipeline_zones and not self._pending_flipped_zones:
            return

        df: pd.DataFrame | None = getattr(self, "_latest_ohlc", None)
        if df is None:
            return

        # Exposure cap.
        try:
            active_setups = self.position_tracker.get_active_setups()
        except Exception:
            logger.exception("main loop: get_active_setups failed")
            return
        active_count = count_active_setups(active_setups)

        for source_label, queue in (
            ("pipeline", self._pending_pipeline_zones),
            ("flipped",  self._pending_flipped_zones),
        ):
            is_flipped_retrade = source_label == "flipped"
            # Iterate over a copy so we can mutate the underlying list.
            remaining: list[tuple[ValidatedZone, UUID]] = []
            for zone, zid in queue:
                exp = check_exposure(
                    active_count=active_count,
                    max_simultaneous=self.config.max_simultaneous_setups,
                    with_candidate=True,
                )
                if not exp.can_open_new:
                    # Out of exposure — leave this and everything
                    # after in the queue for a future tick.
                    remaining.append((zone, zid))
                    remaining.extend(queue[queue.index((zone, zid)) + 1:])
                    break
                if self._try_place_setup(
                    zone, df, zid,
                    bid=bid, ask=ask,
                    is_flipped_retrade=is_flipped_retrade,
                ):
                    active_count += 1
                    self.state.placed_setup_count += 1
                    logger.info(
                        f"setup placed from source={source_label} "
                        f"zone_id={zid}"
                    )
                    # Don't keep this entry on the queue.
                    continue
                # Placement didn't fire (gate deferred, dedup blocked,
                # SL/TP/lot rejected). Keep on the queue for next tick.
                remaining.append((zone, zid))
            if source_label == "pipeline":
                self._pending_pipeline_zones = remaining
            else:
                self._pending_flipped_zones = remaining

    def _persist_zone_if_new(self, zone: ValidatedZone) -> UUID | None:
        """Insert the zone row (status=CONFIRMED) if not already cached.

        Returns the row UUID on success or cache hit; ``None`` on
        Supabase failure (the zone is skipped for this iteration; a
        later detection will retry).
        """
        key = _zone_key_from_validated(zone)
        cached = self._persisted_zone_keys.get(key)
        if cached is not None:
            return cached
        try:
            zone_row = self.supabase.log_zone(_zone_to_input(zone))
            zone_id = UUID(str(zone_row["id"]))
        except Exception:
            logger.exception(
                f"log_zone failed for {zone.direction} "
                f"{zone.bottom:.2f}-{zone.top:.2f}; will retry next iteration"
            )
            return None
        self._persisted_zone_keys[key] = zone_id
        logger.info(
            f"zone persisted: id={zone_id} {zone.direction} "
            f"{zone.bottom:.2f}-{zone.top:.2f} formed_at={zone.formed_at}"
        )
        return zone_id

    # ------------------------------------------------------------------ #
    # SnD Flip trade side path (PR #38)
    # ------------------------------------------------------------------ #

    def _load_confirmed_candidates(
        self,
        df: pd.DataFrame,
        *,
        exclude_zone_ids: set[UUID],
    ) -> list[tuple[ValidatedZone, UUID]]:
        """PR #55: load CONFIRMED zones from DB for the placement queue.

        Mirrors :meth:`_load_flipped_candidates` for the CONFIRMED path.
        The placement queue used to be built solely from the latest
        ``run_strategy_pipeline`` output, which silently dropped any
        CONFIRMED zone the pipeline failed to re-detect on a given M5
        close — even though the zone was still alive in Supabase. This
        method makes Supabase the source of truth: every alive
        CONFIRMED zone enters the queue, regardless of whether the
        pipeline reproduced it.

        ``exclude_zone_ids`` is the set of zone UUIDs already present
        in the pipeline-output queue — passing them in avoids
        duplicate (zone, view) tuples in ``_pending_pipeline_zones``.

        The body-broken-since-formation check is applied against the
        live df. Zones whose bottom (BUY) or top (SELL) has already
        been body-violated by a post-formation bar are skipped — the
        zone is structurally dead. Same rule the original
        ``validate_strong_point`` path applies; mirroring it here
        keeps the queue and the lifecycle scanner in agreement.
        """
        try:
            confirmed_zones = self.supabase.get_zones_by_status(
                ["CONFIRMED"]
            )
        except Exception:
            logger.exception(
                "_load_confirmed_candidates: "
                "get_zones_by_status(['CONFIRMED']) failed"
            )
            return []

        # PR #56: apply the same freshness window we use for dedup.
        # If the bot can't see a zone (older than the window), we
        # don't load it. Symmetric with the dedup block — old burnt
        # zones don't block placement, AND old CONFIRMED zones don't
        # get traded. One config knob governs both.
        freshness_cutoff: datetime | None = None
        freshness_hours = self.strategy_pipeline_config.zone_freshness_hours
        if freshness_hours > 0:
            freshness_cutoff = datetime.now(tz=timezone.utc) - timedelta(
                hours=freshness_hours,
            )

        candidates: list[tuple[ValidatedZone, UUID]] = []
        for cz in confirmed_zones:
            # Defensive: only consider rows whose status is actually
            # CONFIRMED. The Supabase query already filters, but a
            # racing transition (zone moves to CONSUMED between query
            # and processing) shouldn't end up adding a dead zone to
            # the queue.
            if cz.status != "CONFIRMED":
                continue
            if cz.id in exclude_zone_ids:
                continue
            # Freshness window — see PR #56 commentary above.
            if freshness_cutoff is not None and cz.created_at < freshness_cutoff:
                continue
            # Body-broken-since-formation defense. Reuses the same
            # helper the FLIPPED loader uses — the function is
            # direction- + timestamp-keyed, not actually FLIPPED-
            # specific. For a CONFIRMED zone we key off ``formed_at``
            # (when the pattern's impulse_after ended).
            if flipped_zone_body_broken_since_flip(
                zone_top=float(cz.top),
                zone_bottom=float(cz.bottom),
                flipped_direction=cz.direction,
                flipped_at=pd.Timestamp(cz.formed_at),
                df=df,
            ):
                logger.info(
                    f"confirmed zone {cz.id} skipped: body-broken "
                    f"since formation"
                )
                continue
            view = _confirmed_zone_as_validated(cz)
            candidates.append((view, cz.id))

        if candidates:
            logger.debug(
                f"_load_confirmed_candidates: {len(candidates)} "
                f"DB-only candidate(s) after body-break filter"
            )
        return candidates

    def _load_flipped_candidates(
        self, df: pd.DataFrame,
    ) -> list[tuple[ValidatedZone, UUID]]:
        """Load FLIPPED zones from DB and project each into a tradeable view.

        Each returned tuple is ``(synthesised_validated_zone, zone_id)``
        — the zone is already persisted (status=FLIPPED), so we don't
        re-insert; we just need its UUID to FK the setup to.

        Zones that have body-closed past the wrong side of the flipped
        direction since their ``flipped_at`` are filtered out — the
        flip premise has been broken, no trade.
        """
        try:
            flipped_zones = self.supabase.get_zones_by_status(["FLIPPED"])
        except Exception:
            logger.exception(
                "flipped trade detection: get_zones_by_status(['FLIPPED']) failed"
            )
            return []

        candidates: list[tuple[ValidatedZone, UUID]] = []
        for fz in flipped_zones:
            if fz.flipped_direction is None or fz.flipped_at is None:
                logger.warning(
                    f"flipped zone {fz.id} missing flipped_direction/flipped_at; "
                    f"DB CHECK should prevent this — skipping"
                )
                continue
            if flipped_zone_body_broken_since_flip(
                zone_top=float(fz.top),
                zone_bottom=float(fz.bottom),
                flipped_direction=fz.flipped_direction,
                flipped_at=pd.Timestamp(fz.flipped_at),
                df=df,
            ):
                logger.info(
                    f"flipped zone {fz.id} skipped: body-broken since flip"
                )
                continue
            view = _flipped_zone_as_validated(fz)
            candidates.append((view, fz.id))

        if candidates:
            logger.debug(
                f"flipped trade detection: {len(candidates)} candidate(s) "
                f"after body-break filter"
            )
        return candidates

    def _run_zone_lifecycle(self, df: pd.DataFrame) -> None:
        """Apply consumption / violation / flip detection to the last bar.

        Operates on the most recently closed M5 bar (``df.iloc[-1]``).
        Loads all non-terminal zones once per bar; transitions each in
        priority order (consumption → violation → flip) and persists
        the result. CONSUMED is fill-agnostic per design decision Q1.
        FLIP recomputes structure from the current df (option B from
        the design doc) so the BoS target reflects today's structure,
        not the swing recorded at zone formation.

        PR #58: FLIPPED zones now also undergo the consumption check
        (using ``flipped_direction`` to determine the retest side).
        Previously a wick into a flipped zone left it in FLIPPED
        status indefinitely — the dashboard kept advertising it as
        tradeable and the placement queue kept trying to fire on it
        even though the institutional liquidity that defined the
        flipped opportunity had already been triggered. Now the
        first wick correctly transitions FLIPPED → CONSUMED.
        Violation / re-flip transitions are NOT applied to already-
        flipped zones — the loader's body-broken-since-flip filter
        handles those cases by excluding them from the queue, and a
        flipped-zone re-flip isn't a state the strategy currently
        models.
        """
        if len(df) == 0:
            return
        last = df.iloc[-1]
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])

        # Pull non-terminal zones we might transition. PR #58 adds
        # FLIPPED to the list so wicks into flipped zones get
        # transitioned to CONSUMED.
        try:
            zones = self.supabase.get_zones_by_status(
                ["CONFIRMED", "ACTIVE", "CONSUMED", "VIOLATED", "FLIPPED"],
            )
        except Exception:
            logger.exception("zone lifecycle: get_zones_by_status failed")
            return

        for z in zones:
            # PR #58: FLIPPED zones get their own narrow consumption-
            # only path. The retest direction is ``flipped_direction``,
            # not the original ``direction``. Once consumed, no further
            # transitions run (a flipped zone re-flipping isn't a
            # state the strategy models).
            if z.status == "FLIPPED":
                if z.flipped_direction is None:
                    # DB CHECK should prevent this — defensive skip.
                    continue
                flipped_ref = ZoneRef(
                    direction=z.flipped_direction,
                    top=float(z.top),
                    bottom=float(z.bottom),
                )
                if check_consumption(
                    flipped_ref, bar_high=bar_high, bar_low=bar_low,
                ):
                    self._safe_update_zone_status(z, "CONSUMED")
                continue

            ref = ZoneRef(
                direction=z.direction,
                top=float(z.top),
                bottom=float(z.bottom),
            )

            # 1. Consumption — any touch consumes (Q1). Only zones not
            # already CONSUMED/VIOLATED can transition here. CONSUMED
            # zones move on to the violation check below.
            if z.status in ("CONFIRMED", "ACTIVE"):
                if check_consumption(ref, bar_high=bar_high, bar_low=bar_low):
                    self._safe_update_zone_status(z, "CONSUMED")
                    # Status changed in this iteration; refresh local
                    # view so subsequent checks see the new state.
                    z = z.model_copy(update={"status": "CONSUMED"})

            # 2. Violation — body close past the wrong-side bound.
            # Reachable from CONFIRMED (gap-through), ACTIVE (rare),
            # or CONSUMED (the common path: touch then break).
            if z.status in ("CONFIRMED", "ACTIVE", "CONSUMED"):
                if check_violation(ref, bar_close=bar_close):
                    self._safe_update_zone_status(z, "VIOLATED")
                    z = z.model_copy(update={"status": "VIOLATED"})

            # 3. Flip — only meaningful on VIOLATED zones. The flip
            # detector scans forward from the violation bar; here the
            # violation just happened, so violation_index = last bar.
            if z.status == "VIOLATED":
                violation_index = len(df) - 1
                flip = check_flip(ref, df, violation_index)
                if flip.flipped:
                    self._safe_update_zone_status(
                        z, "FLIPPED",
                        flipped_direction=flip.new_direction,
                    )

    def _safe_update_zone_status(
        self,
        zone: Zone,
        new_status: str,
        *,
        flipped_direction: str | None = None,
    ) -> None:
        """Persist a zone transition; log + swallow on failure.

        Single zone's failure shouldn't poison the rest of the loop.
        Also logs the transition at INFO so the operator can see the
        full lifecycle in bot_logs / loguru output.
        """
        try:
            self.supabase.update_zone_status(
                zone.id,
                new_status,  # type: ignore[arg-type]
                flipped_direction=flipped_direction,  # type: ignore[arg-type]
            )
            log_transition(
                str(zone.id), zone.status, new_status,  # type: ignore[arg-type]
            )
        except Exception:
            logger.exception(
                f"zone lifecycle: update_zone_status failed for {zone.id} "
                f"({zone.status} → {new_status})"
            )

    def _try_place_setup(
        self, zone: ValidatedZone, ohlc_df: pd.DataFrame, zone_id: UUID,
        *,
        bid: float, ask: float,
        is_flipped_retrade: bool = False,
    ) -> bool:
        """Compute SL + TP1 + lot size, validate, place orders.

        ``zone_id`` is supplied by the caller — the zone row was already
        inserted (or looked up from cache) by ``_persist_zone_if_new``
        before we got here. Setup row FKs to it.

        ``bid`` / ``ask`` are the current tick prices, used by the
        price-vs-zone gate to skip setups when price hasn't reached
        the planned Layer 1 entry yet. (Bug 2 fix.)

        ``is_flipped_retrade`` flags the SnD Flip side path (PR #45):
        the candidate comes from a status='FLIPPED' zone whose
        ``flipped_direction`` is being traded. Dedup is skipped for
        these because the dedup re-trade guard reliably trips on the
        previously-violated counter-direction zone at the same price
        band (which is what the original demand/supply broke to form),
        even though that's not the same physical zone. The flipped
        path has its own safety guard
        (``flipped_zone_body_broken_since_flip``) plus the FLIPPED →
        ACTIVE transition that pops the zone off the flipped queue
        after placement.

        Returns True iff ``order_manager.place_layered_orders`` reported
        ``PLACED`` (not FAILED, not SKIPPED-via-gap).
        """
        # 0. Dedup — skip if a CONFIRMED/ACTIVE/CONSUMED/VIOLATED/FLIPPED
        # zone with overlapping bounds already exists (within the
        # freshness window). PR #59 added CONFIRMED + ACTIVE to the
        # check; previously only the dead-zone statuses blocked, so
        # the bot could fire multiple setups on the same band when
        # the pipeline kept re-detecting the same structural zone
        # across consecutive M5 closes with shifting ``formed_at``.
        # ``zone_id`` is excluded from the overlap scan so the
        # candidate doesn't dedup against itself.
        if not is_flipped_retrade and self._zone_already_used(
            zone, candidate_zone_id=zone_id,
        ):
            logger.info(
                f"new setup skipped: zone {zone.direction} "
                f"{zone.bottom:.2f}-{zone.top:.2f} overlaps an existing "
                f"recent zone"
            )
            return False

        # 0a. PR #48: zone-visibility gate (pipeline path only).
        # Refuse to place a fresh setup if any candle since formation
        # has bodied through the zone — the zone has been obscured /
        # is no longer respected by price. The flipped path keeps its
        # own existing safety guard
        # (``flipped_zone_body_broken_since_flip``) so we skip this
        # check for flipped retrades.
        if not is_flipped_retrade:
            bodies = count_bodies_through_zone(
                ohlc_df,
                zone_top=zone.top,
                zone_bottom=zone.bottom,
                since_time=pd.Timestamp(zone.formed_at),
            )
            if bodies >= 1:
                logger.info(
                    f"new setup skipped: zone {zone.direction} "
                    f"{zone.bottom:.2f}-{zone.top:.2f} is obscured "
                    f"({bodies} candle(s) bodied through since "
                    f"formed_at={zone.formed_at})"
                )
                return False

        # 0b. Price-vs-zone gate (Bug 2 fix). Don't place Layer 1 as a
        # market order if current price hasn't reached the zone yet —
        # we'd fill at the wrong price (potentially many points away
        # from the planned Layer 1 entry). Skip this iteration; the
        # next M5 close re-evaluates.
        #
        # * BUY  Layer 1 = zone.top.  Fires when price drops in from
        #        above → bid must be ≤ zone.top.
        # * SELL Layer 1 = zone.bottom. Fires when price rises in from
        #        below → ask must be ≥ zone.bottom.
        #
        # Overshoot (price past the FAR edge) is caught by the existing
        # ``_detect_gap_through`` check in order_manager; this gate
        # only covers the "not yet at zone" case.
        if zone.direction == "BUY" and bid > zone.top:
            logger.info(
                f"new setup deferred: BUY zone {zone.bottom:.2f}-"
                f"{zone.top:.2f}, current bid {bid:.2f} above zone.top; "
                f"waiting for price to retest"
            )
            return False
        if zone.direction == "SELL" and ask < zone.bottom:
            logger.info(
                f"new setup deferred: SELL zone {zone.bottom:.2f}-"
                f"{zone.top:.2f}, current ask {ask:.2f} below zone.bottom; "
                f"waiting for price to retest"
            )
            return False

        # 1. SL — zone-bound formula (loosened-rules PR).
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

        # 2. TP chain (PR #41): TP1 from L1 entry, TP2 from TP1, TP3
        # from TP2. TP1 is required — skip the setup if no peak. TP2
        # / TP3 are best-effort; NULLs are recomputed by tp_manager
        # at the previous layer's TP hit. Q-B decision: a layer with
        # no TP rides on the cascaded SL until external close.
        lookback = self.strategy_pipeline_config.tp1_local_peak_lookback_bars
        tp1_price = find_nearest_local_peak(
            ohlc_df,
            entry_price=entry_price,
            direction=zone.direction,
            lookback_bars=lookback,
        )
        if tp1_price is None:
            logger.info(
                f"new setup skipped: no local "
                f"{'peak' if zone.direction == 'BUY' else 'low'} "
                f"within {lookback} "
                f"bars {'above' if zone.direction == 'BUY' else 'below'} "
                f"entry {entry_price:.2f}"
            )
            return False
        tp2_price = find_nearest_local_peak(
            ohlc_df,
            entry_price=tp1_price,
            direction=zone.direction,
            lookback_bars=lookback,
        )
        tp3_price = (
            find_nearest_local_peak(
                ohlc_df,
                entry_price=tp2_price,
                direction=zone.direction,
                lookback_bars=lookback,
            ) if tp2_price is not None else None
        )

        # 3. Lot size.
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

        # 4. Place orders. (Zone row already persisted by
        # ``_persist_zone_if_new``; ``zone_id`` was passed in.)
        try:
            result = place_layered_orders(
                zone, zone_id,
                lot_size=lot_result.lot_size,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                tp3_price=tp3_price,
                mt5=self.mt5,
                supabase=self.supabase,
                tracker=self.position_tracker,
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
    # Per-layer TP recompute (PR #41, Q-C: only when NULL)
    # ------------------------------------------------------------------ #

    def _maybe_recompute_next_tp(
        self, setup: Setup, closed_layer_number: int,
    ) -> None:
        """Recompute ``planned_tp{N+1}_price`` if NULL on the setup row.

        Called by the main loop after ``tp_manager`` reports a layer
        close with ``needs_next_tp_recompute=True``. Reference price
        for the search is the **just-closed layer's TP**, so the new
        peak is guaranteed to be strictly above (BUY) / below (SELL)
        the previous one. If no peak exists in the current df, the
        slot stays NULL and the corresponding layer rides on the
        cascaded SL until external close (Q-B decision).
        """
        if closed_layer_number >= 3:
            return  # no TP4

        next_layer = closed_layer_number + 1
        # Reference = the layer that just closed at its TP price.
        if closed_layer_number == 1:
            reference = float(setup.planned_tp1_price)
        elif closed_layer_number == 2:
            if setup.planned_tp2_price is None:
                # Defensive: TP2 was the layer that fired, so it
                # should be set. Nothing to recompute against.
                return
            reference = float(setup.planned_tp2_price)
        else:  # closed_layer_number == 3 (handled above)
            return

        df: pd.DataFrame | None = getattr(self, "_latest_ohlc", None)
        if df is None or len(df) == 0:
            logger.warning(
                f"tp recompute: no OHLC available for setup {setup.id}"
            )
            return

        new_tp = find_nearest_local_peak(
            df,
            entry_price=reference,
            direction=setup.direction,
            lookback_bars=self.strategy_pipeline_config.tp1_local_peak_lookback_bars,
        )
        if new_tp is None:
            logger.info(
                f"tp recompute: no local "
                f"{'peak' if setup.direction == 'BUY' else 'low'} "
                f"above reference {reference:.2f} for setup {setup.id} "
                f"layer {next_layer} — layer rides cascaded SL"
            )
            return

        field = f"planned_tp{next_layer}_price"
        try:
            self.supabase.update_setup(setup.id, **{field: Decimal(str(new_tp))})
            logger.info(
                f"tp recompute: setup={setup.id} {field}={new_tp:.2f} "
                f"(reference={reference:.2f})"
            )
        except Exception:
            logger.exception(
                f"tp recompute: persist failed for setup {setup.id} {field}"
            )

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

    def _zone_already_used(
        self,
        candidate: ValidatedZone,
        *,
        candidate_zone_id: UUID | None = None,
    ) -> bool:
        """True iff a *recent* overlapping zone already exists in the DB.

        Best-effort: if the lookup fails we let the setup proceed
        (logging the error). Better to occasionally re-trade than to
        miss real entries because of a Supabase blip.

        PR #56 / PR #62: two-tier freshness cap. The dedup lookup
        applies different age cutoffs depending on whether the
        overlapping zone is **live** (CONFIRMED / ACTIVE) or **dead**
        (CONSUMED / VIOLATED / FLIPPED):

        * Live zones — capped at ``zone_freshness_hours`` (default 6h).
          Symmetric with ``_load_confirmed_candidates``: if the loader
          ignores a CONFIRMED zone older than 6h, dedup should ignore
          it too. Otherwise the zone becomes a zombie — invisible to
          trading but still blocking new patterns in its band.
        * Dead zones — capped at ``dead_zone_dedup_freshness_hours``
          (default 0.0 = no cap). Restores the pre-PR-#56 protective
          filter where stale burnt zones in the DB filter out the
          loose pattern-detection false positives (PR #44 / PR #46
          lowered thresholds enough that "DBD" patterns get marked
          inside continuous impulse legs; the pre-PR-#56 strict-era
          dead zones quietly dedup-blocked them).

        PR #59: extends the dedup status set to include CONFIRMED and
        ACTIVE in addition to the dead-zone statuses. Previously the
        pipeline could detect the same structural zone on consecutive
        M5 closes with shifting ``formed_at`` (the impulse_after's
        end bar moves as new bars accumulate), producing multiple
        CONFIRMED rows with identical bounds and different UUIDs.
        Dedup didn't catch them because it only checked dead statuses
        → the bot fired a fresh setup on every duplicate.
        ``candidate_zone_id`` is passed so the candidate doesn't dedup
        against its own row (which is in DB with status CONFIRMED at
        this point).
        """
        try:
            existing = self.supabase.get_zones_by_status(
                ["CONFIRMED", "ACTIVE", "CONSUMED", "VIOLATED", "FLIPPED"],
            )
        except Exception:
            logger.exception(
                "dedup: get_zones_by_status failed; proceeding without dedup"
            )
            return False

        now = datetime.now(tz=timezone.utc)
        live_cutoff: datetime | None = None
        live_hours = self.strategy_pipeline_config.zone_freshness_hours
        if live_hours > 0:
            live_cutoff = now - timedelta(hours=live_hours)
        dead_cutoff: datetime | None = None
        dead_hours = (
            self.strategy_pipeline_config.dead_zone_dedup_freshness_hours
        )
        if dead_hours > 0:
            dead_cutoff = now - timedelta(hours=dead_hours)

        candidate_ref = ZoneRef(
            direction=candidate.direction,
            top=float(candidate.top),
            bottom=float(candidate.bottom),
        )
        live_statuses = ("CONFIRMED", "ACTIVE")
        for z in existing:
            # PR #59: self-exclude. The candidate's own zone row is
            # already persisted (status CONFIRMED) at this point, so
            # without this check every candidate would dedup against
            # itself and skip placement.
            if candidate_zone_id is not None and z.id == candidate_zone_id:
                continue
            cutoff = live_cutoff if z.status in live_statuses else dead_cutoff
            if cutoff is not None and z.created_at < cutoff:
                continue
            existing_ref = ZoneRef(
                direction=z.direction,
                top=float(z.top),
                bottom=float(z.bottom),
            )
            if zone_bounds_overlap(candidate_ref, existing_ref):
                return True
        return False

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


def _zone_key_from_validated(
    zone: ValidatedZone,
) -> tuple[str, float, float, str]:
    """Idempotency key for an in-memory ``ValidatedZone``.

    Matches the natural composite key on the ``zones`` table:
    ``(direction, top, bottom, formed_at)``. Top / bottom are rounded
    to 2 decimal places to match the ``NUMERIC(10,2)`` DB precision
    — pipeline re-runs may produce float-equal-but-not-identical
    values for the same physical zone, so unrounded comparisons would
    miss cache hits.
    """
    return (
        zone.direction,
        round(float(zone.top), 2),
        round(float(zone.bottom), 2),
        zone.formed_at.isoformat(),
    )


def _zone_key_from_row(
    direction: str, top: Any, bottom: Any, formed_at: datetime,
) -> tuple[str, float, float, str]:
    """Idempotency key for a DB row (from :class:`Zone` read model).

    Symmetric with :func:`_zone_key_from_validated`. ``top`` and
    ``bottom`` arrive as ``Decimal`` from Supabase; ``formed_at`` as
    a ``datetime``. Both get normalised the same way as the in-memory
    side so cache hits work across the boundary.
    """
    return (
        direction,
        round(float(top), 2),
        round(float(bottom), 2),
        formed_at.isoformat(),
    )


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


def _confirmed_zone_as_validated(zone_row: Any) -> ValidatedZone:
    """PR #55: synthesise a tradeable ``ValidatedZone`` view of a
    CONFIRMED zone row.

    Counterpart to :func:`_flipped_zone_as_validated`. Same idea:
    the placement code reads only ``direction``, ``top``, ``bottom``,
    ``refined_zone.is_tradeable``, and ``is_strong_point``; the rest
    is stubbed for traceability. The CONFIRMED case uses
    ``zone_row.direction`` and ``zone_row.formed_at`` directly (no
    flipped indirection).
    """
    direction = zone_row.direction
    formed_at = zone_row.formed_at
    if direction is None or formed_at is None:
        raise ValueError(
            f"zone {zone_row.id} missing direction or formed_at; "
            f"cannot synthesise ValidatedZone view"
        )

    ts = pd.Timestamp(formed_at)
    top = float(zone_row.top)
    bottom = float(zone_row.bottom)
    pt = _PT(zone_row.pattern_type)
    impulse_dir = "RALLY" if direction == "BUY" else "DROP"
    impulse = _Impulse(
        direction=impulse_dir,
        start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=0.0, largest_body=0.0, candle_count=1,
    )
    base = _Base(
        start_index=0, end_index=0, candle_count=1,
        top=top, bottom=bottom,
        range_size=top - bottom, largest_body=0.0,
    )
    pattern = _Pattern(
        pattern_type=pt,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction=direction,
        formed_at=ts,
    )
    zone_shape = _ZoneShape(
        direction=direction,
        top=top, bottom=bottom,
        formed_at=ts, source_pattern=pattern,
    )
    refined = _RefinedZone(
        direction=direction,
        top=top, bottom=bottom,
        formed_at=ts, source_pattern=pattern,
        is_tradeable=True,
        rejection_reason=None,
        original_zone=zone_shape,
    )
    return ValidatedZone(
        direction=direction,
        top=top, bottom=bottom,
        formed_at=ts, source_pattern=pattern,
        refined_zone=refined,
        is_strong_point=True,
        validation_failures=[],
        broken_swing=None, broken_at=None, sl_anchor_swing=None,
    )


def _flipped_zone_as_validated(zone_row: Any) -> ValidatedZone:
    """Synthesise a tradeable ``ValidatedZone`` view of a FLIPPED zone row.

    The SnD Flip side path (PR #38) trades flipped zones in their
    ``flipped_direction`` without a fresh pattern detection. We need a
    ValidatedZone-shaped object to feed ``_try_place_setup`` —
    everything in the placement path reads only ``direction``, ``top``,
    ``bottom``, ``refined_zone.is_tradeable``, and ``is_strong_point``.
    Other fields are stubbed.

    Pattern fields aren't read by the placement code (``_zone_to_input``
    would read ``source_pattern.pattern_type.value``, but that's only
    called from ``_persist_zone_if_new`` — which we skip for already-
    persisted flipped zones). The stub Pattern is built to be
    internally consistent anyway.

    ``formed_at`` is set to ``flipped_at`` (per Q3 lock-in) — the
    moment the new direction came into existence.
    """
    direction = zone_row.flipped_direction
    flipped_at = zone_row.flipped_at
    if direction is None or flipped_at is None:
        raise ValueError(
            f"zone {zone_row.id} is not properly FLIPPED "
            f"(missing flipped_direction or flipped_at)"
        )

    ts = pd.Timestamp(flipped_at)
    top = float(zone_row.top)
    bottom = float(zone_row.bottom)
    # Keep the original pattern_type from the DB row — historical
    # truth ("this position formed as an RBR demand"), even though the
    # flipped trade goes in the opposite direction. Not read by the
    # placement path; kept for traceability.
    pt = _PT(zone_row.pattern_type)
    impulse_dir = "RALLY" if direction == "BUY" else "DROP"
    impulse = _Impulse(
        direction=impulse_dir,
        start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=0.0, largest_body=0.0, candle_count=1,
    )
    base = _Base(
        start_index=0, end_index=0, candle_count=1,
        top=top, bottom=bottom,
        range_size=top - bottom, largest_body=0.0,
    )
    pattern = _Pattern(
        pattern_type=pt,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction=direction,
        formed_at=ts,
    )
    zone_shape = _ZoneShape(
        direction=direction,
        top=top, bottom=bottom,
        formed_at=ts, source_pattern=pattern,
    )
    refined = _RefinedZone(
        direction=direction,
        top=top, bottom=bottom,
        formed_at=ts, source_pattern=pattern,
        is_tradeable=True,
        rejection_reason=None,
        original_zone=zone_shape,
    )
    return ValidatedZone(
        direction=direction,
        top=top, bottom=bottom,
        formed_at=ts, source_pattern=pattern,
        refined_zone=refined,
        is_strong_point=True,
        validation_failures=[],
        broken_swing=None, broken_at=None, sl_anchor_swing=None,
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
