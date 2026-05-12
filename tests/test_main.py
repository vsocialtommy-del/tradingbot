"""Smoke tests for ``bot.main`` — orchestration logic, not full E2E.

The Bot owns every manager; these tests replace each manager with a
``MagicMock(spec=...)`` after Bot construction so we can verify the
orchestrator's wiring (call counts, branch gating, cadence firing)
without dragging real Supabase / MT5 plumbing into test setup. End-to-
end integration is Phase F's job.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.data.ohlc_provider import OHLCProvider
from bot.execution.entry_trigger import EntryTrigger, FiredTrigger
from bot.execution.mt5_connector import MT5Connector
from bot.execution.position_tracker import PositionTracker, ReconcileResult
from bot.exits.sl_manager import (
    SLCalculation,
    SLManager,
    SLValidation,
)
from bot.exits.tp1_manager import TP1Manager, TP1Result
from bot.filters.news_filter import (
    NewsCheckResult,
    NewsFilter,
)
from bot.logging.supabase_logger import Setup, SupabaseLogger
from bot.main import (
    Bot,
    BotLoopConfig,
    _elapsed,
    _parse_pause_until,
    _zone_to_input,
    main,
)
from bot.risk.daily_halt import DailyHaltResult
from bot.strategy.pattern_detection import (
    Base as _Base,
    Impulse as _Impulse,
    Pattern as _Pattern,
    PatternType as _PT,
)
from bot.strategy.strong_point import ValidatedZone as _ValidatedZone
from bot.strategy.structure import Swing
from bot.strategy.zone_marking import Zone as _Zone
from bot.strategy.zone_refinement import RefinedZone as _RefinedZone


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_setup(
    *,
    id: UUID | None = None,
    direction: str = "BUY",
    status: str = "ACTIVE",
) -> Setup:
    return Setup(
        id=id or uuid4(),
        zone_id=uuid4(),
        direction=direction,  # type: ignore[arg-type]
        entry_mode="STRONG_POINT_FIRST_TOUCH",
        planned_layer1_price=Decimal("1900"),
        planned_layer2_price=Decimal("1897.5"),
        planned_layer3_price=Decimal("1895"),
        planned_sl_price=Decimal("1880"),
        planned_tp1_price=Decimal("1907"),
        status=status,  # type: ignore[arg-type]
        skip_reason=None,
        activated_at=NOW,
        closed_at=None,
        created_at=NOW,
        updated_at=NOW,
    )


def make_ohlc(n: int = 30, last_time: str = "2026-05-08T12:00:00Z") -> pd.DataFrame:
    times = pd.date_range(end=last_time, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1900.0] * n, "high": [1901.0] * n,
            "low": [1899.0] * n, "close": [1900.0] * n,
            "volume": [100] * n,
        },
        index=times,
    )


# --------------------------------------------------------------------------- #
# Fixtures — Bot with every manager mocked
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_mt5(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=MT5Connector)
    m.get_balance.return_value = 10000.0
    m.get_current_price.return_value = {
        "bid": 1900.0, "ask": 1900.1, "time": NOW, "time_msc": 0,
    }
    m.get_open_positions.return_value = []
    return m


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    # Default: kill switch off, no pause.
    m.check_bot_config.side_effect = lambda key: {
        "kill_switch": False, "pause_until": None,
    }.get(key, None)
    m.get_news_events_in_window.return_value = []
    m.log_zone.return_value = {"id": str(uuid4())}
    m.log_setup.return_value = {"id": str(uuid4())}
    # Empty by default: no zones to dedup against, no zones to scan
    # for lifecycle transitions. Specific tests override this.
    m.get_zones_by_status.return_value = []
    m.log_trade.side_effect = [
        {"id": str(uuid4())} for _ in range(20)
    ]
    return m


def _replace_managers(
    bot: Bot, mocker: MockerFixture,
) -> dict[str, MagicMock]:
    """Swap each manager attribute with a typed MagicMock and return them."""
    bot.position_tracker = mocker.MagicMock(spec=PositionTracker)
    bot.position_tracker.get_active_setups.return_value = []
    bot.position_tracker.reconcile_with_mt5.return_value = ReconcileResult(
        ghost_tickets=[], lost_trade_ids=[],
        closed_externally_count=0, matched_count=0,
    )
    bot.position_tracker.detect_closed_positions.return_value = []

    bot.ohlc_provider = mocker.MagicMock(spec=OHLCProvider)
    bot.ohlc_provider.get.return_value = make_ohlc(n=30)

    bot.tp1_manager = mocker.MagicMock(spec=TP1Manager)
    bot.tp1_manager.check.return_value = TP1Result(triggered=False)

    bot.sl_manager = mocker.MagicMock(spec=SLManager)
    bot.sl_manager.calculate_initial_sl.return_value = SLCalculation(
        sl_price=1882.5, reference_swing_price=1900.0,
        buffer_used=17.5, lookback_used=20, direction="BUY",
    )
    bot.sl_manager.validate_sl_distance.return_value = SLValidation(
        is_valid=True, distance_points=17.5,
        is_too_close=False, is_too_far=False,
    )

    bot.entry_trigger = mocker.MagicMock(spec=EntryTrigger)
    bot.entry_trigger.check_live.return_value = []

    bot.news_filter = mocker.MagicMock(spec=NewsFilter)
    bot.news_filter.check.return_value = NewsCheckResult(is_blocked=False)

    return {
        "position_tracker": bot.position_tracker,
        "ohlc_provider": bot.ohlc_provider,
        "tp1_manager": bot.tp1_manager,
        "sl_manager": bot.sl_manager,
        "entry_trigger": bot.entry_trigger,
        "news_filter": bot.news_filter,
    }


@pytest.fixture
def bot(
    mock_mt5: MagicMock, mock_supabase: MagicMock,
    mocker: MockerFixture,
) -> tuple[Bot, dict[str, MagicMock]]:
    b = Bot(mt5=mock_mt5, supabase=mock_supabase)
    mgrs = _replace_managers(b, mocker)
    # Initialise the runtime state as if startup ran (without actually
    # calling .initialize() — tests should be able to drive a single
    # iteration without the connect/reconcile side effects).
    b.state.starting_balance = 10000.0
    return b, mgrs


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


class TestElapsed:
    def test_none_returns_inf(self) -> None:
        assert _elapsed(None, NOW) == float("inf")

    def test_seconds_diff(self) -> None:
        last = NOW - timedelta(seconds=42)
        assert _elapsed(last, NOW) == pytest.approx(42.0)


class TestParsePauseUntil:
    def test_none_passthrough(self) -> None:
        assert _parse_pause_until(None) is None

    def test_iso_string_with_z_suffix(self) -> None:
        dt = _parse_pause_until("2026-05-08T13:00:00Z")
        assert dt == datetime(2026, 5, 8, 13, 0, tzinfo=timezone.utc)

    def test_iso_string_with_offset(self) -> None:
        dt = _parse_pause_until("2026-05-08T13:00:00+00:00")
        assert dt == datetime(2026, 5, 8, 13, 0, tzinfo=timezone.utc)

    def test_naive_iso_string_assumed_utc(self) -> None:
        dt = _parse_pause_until("2026-05-08T13:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_garbage_returns_none(self) -> None:
        assert _parse_pause_until("not a date") is None
        assert _parse_pause_until(42) is None


# --------------------------------------------------------------------------- #
# initialize() — startup sequence
# --------------------------------------------------------------------------- #


class TestInitialize:
    def test_connects_captures_balance_reconciles(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock,
    ) -> None:
        b, mgrs = bot
        b.state.starting_balance = None  # reset so we can verify capture
        b.initialize()

        mock_mt5.connect.assert_called_once()
        assert b.state.starting_balance == 10000.0
        mgrs["position_tracker"].reconcile_with_mt5.assert_called_once()
        # Last reconcile + last config refresh should be set.
        assert b.state.last_reconcile is not None
        assert b.state.last_config_refresh is not None

    def test_balance_failure_does_not_crash_initialize(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock,
    ) -> None:
        b, _ = bot
        b.state.starting_balance = None  # undo fixture pre-set
        mock_mt5.get_balance.side_effect = RuntimeError("auth")
        # Should not raise.
        b.initialize()
        # starting_balance stays None; daily halt will read it lazily.
        assert b.state.starting_balance is None

    def test_reconcile_failure_does_not_crash_initialize(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        mgrs["position_tracker"].reconcile_with_mt5.side_effect = (
            RuntimeError("DB down")
        )
        b.initialize()  # no raise


# --------------------------------------------------------------------------- #
# _safe_get_active_setups — transient httpx errors get brief logs, not
# full tracebacks. Regression guard for the 10K-request connection-cycle
# noise reported during Windows demo trading.
# --------------------------------------------------------------------------- #


class TestSafeGetActiveSetupsLogging:
    def test_httpx_request_error_logged_briefly_no_traceback(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        import httpx
        b, mgrs = bot
        mgrs["position_tracker"].get_active_setups.side_effect = (
            httpx.RemoteProtocolError("ConnectionTerminated error_code:0")
        )
        # Spy on loguru handlers.
        exception_spy = mocker.patch.object(
            __import__("bot.main", fromlist=["logger"]).logger, "exception",
        )
        warning_spy = mocker.patch.object(
            __import__("bot.main", fromlist=["logger"]).logger, "warning",
        )

        result = b._safe_get_active_setups()

        assert result == []
        # WARN-level brief message, not exception-level full traceback.
        warning_spy.assert_called_once()
        exception_spy.assert_not_called()

    def test_unexpected_error_still_gets_full_traceback(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # Anything that's NOT an httpx.RequestError keeps the full
        # exception logger so genuine bugs surface.
        b, mgrs = bot
        mgrs["position_tracker"].get_active_setups.side_effect = (
            RuntimeError("something genuinely broken")
        )
        exception_spy = mocker.patch.object(
            __import__("bot.main", fromlist=["logger"]).logger, "exception",
        )

        result = b._safe_get_active_setups()

        assert result == []
        exception_spy.assert_called_once()


# --------------------------------------------------------------------------- #
# run_iteration() — orchestration core
# --------------------------------------------------------------------------- #


class TestRunIterationEmpty:
    def test_no_active_setups_no_managers_called_for_setup_loops(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        mgrs["position_tracker"].get_active_setups.return_value = []
        b.run_iteration(NOW)
        # entry_trigger.check_live runs every tick (it iterates internally).
        mgrs["entry_trigger"].check_live.assert_called_once()
        # tp1_manager.check is per-active-setup → 0 calls when none active.
        mgrs["tp1_manager"].check.assert_not_called()

    def test_get_current_price_failure_returns_early(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock,
    ) -> None:
        b, mgrs = bot
        mock_mt5.get_current_price.side_effect = RuntimeError("disconnected")
        # Doesn't raise.
        b.run_iteration(NOW)
        # entry_trigger / tp1_manager not called when tick read fails.
        mgrs["entry_trigger"].check_live.assert_not_called()
        mgrs["tp1_manager"].check.assert_not_called()


class TestRunIterationActiveSetups:
    def test_tp1_manager_called_per_active_setup(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        s1, s2 = make_setup(), make_setup()
        mgrs["position_tracker"].get_active_setups.return_value = [s1, s2]
        b.run_iteration(NOW)
        # ACTIVE → tp1_manager.check called once per setup.
        assert mgrs["tp1_manager"].check.call_count == 2

    def test_pending_setup_skipped_for_tp1(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        s_pending = make_setup(status="PENDING")
        s_active = make_setup(status="ACTIVE")
        mgrs["position_tracker"].get_active_setups.return_value = [
            s_pending, s_active,
        ]
        b.run_iteration(NOW)
        # Only the ACTIVE one gets a TP1 check.
        assert mgrs["tp1_manager"].check.call_count == 1

    def test_entry_trigger_fired_layer_increments_counter(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        mgrs["entry_trigger"].check_live.return_value = [
            FiredTrigger(
                setup_id=uuid4(), trade_id=uuid4(),
                layer_number=2, mt5_ticket=22222, fill_price=1897.5,
            ),
        ]
        b.run_iteration(NOW)
        assert b.state.fired_layer_count == 1

    def test_tp1_triggered_increments_counter(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        s = make_setup(status="ACTIVE")
        mgrs["position_tracker"].get_active_setups.return_value = [s]
        mgrs["tp1_manager"].check.return_value = TP1Result(
            triggered=True, tp1_price=1907.0,
            closed_lots=0.0, new_sl_price=1900.0,
        )
        b.run_iteration(NOW)
        assert b.state.tp1_count == 1

    def test_tp1_manager_exception_does_not_kill_iteration(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        s = make_setup(status="ACTIVE")
        mgrs["position_tracker"].get_active_setups.return_value = [s]
        mgrs["tp1_manager"].check.side_effect = RuntimeError("boom")
        # Doesn't raise.
        b.run_iteration(NOW)


# --------------------------------------------------------------------------- #
# Pause / kill switch / news / daily halt
# --------------------------------------------------------------------------- #


class TestPauseGating:
    def test_kill_switch_blocks_strategy_pipeline(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.kill_switch = True
        # Strategy pipeline shouldn't run; OHLC fetch shouldn't either.
        # Force config refresh skip.
        b.state.last_config_refresh = NOW

        # Force has_new_m5_close branch to be reachable (would be if it ran).
        run_pipeline = mocker.patch("bot.main.run_strategy_pipeline")
        b.run_iteration(NOW)
        run_pipeline.assert_not_called()

    def test_pause_until_in_future_blocks_strategy_pipeline(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        b.state.pause_until = NOW + timedelta(minutes=5)
        b.state.last_config_refresh = NOW

        run_pipeline = mocker.patch("bot.main.run_strategy_pipeline")
        b.run_iteration(NOW)
        run_pipeline.assert_not_called()

    def test_pause_until_in_past_does_not_block(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.pause_until = NOW - timedelta(minutes=5)  # expired
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None  # ensure new-M5 path runs

        run_pipeline = mocker.patch(
            "bot.main.run_strategy_pipeline", return_value=[],
        )
        b.run_iteration(NOW)
        # Pipeline did run (no zones → no order placement, but ran).
        run_pipeline.assert_called_once()


class TestNewsBlock:
    def test_news_blocked_skips_strategy_pipeline(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        mgrs["news_filter"].check.return_value = NewsCheckResult(
            is_blocked=True, block_reason="HIGH USD NFP at 13:30Z",
        )
        run_pipeline = mocker.patch("bot.main.run_strategy_pipeline")
        b.run_iteration(NOW)
        run_pipeline.assert_not_called()

    def test_news_blocked_does_not_skip_existing_setup_management(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        mgrs["news_filter"].check.return_value = NewsCheckResult(
            is_blocked=True,
        )
        s = make_setup(status="ACTIVE")
        mgrs["position_tracker"].get_active_setups.return_value = [s]
        b.run_iteration(NOW)
        # TP1 still checked even during blackout.
        mgrs["tp1_manager"].check.assert_called_once()


class TestDailyHalt:
    def test_drawdown_over_limit_blocks_strategy_pipeline(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.starting_balance = 10000.0
        mock_mt5.get_balance.return_value = 8000.0  # -20% drawdown
        run_pipeline = mocker.patch("bot.main.run_strategy_pipeline")
        b.run_iteration(NOW)
        run_pipeline.assert_not_called()

    def test_within_drawdown_limit_pipeline_runs(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.starting_balance = 10000.0
        mock_mt5.get_balance.return_value = 9500.0  # -5%
        b.state.last_m5_bar_time = None  # new-M5 path
        run_pipeline = mocker.patch(
            "bot.main.run_strategy_pipeline", return_value=[],
        )
        b.run_iteration(NOW)
        run_pipeline.assert_called_once()


# --------------------------------------------------------------------------- #
# Strategy pipeline → order placement
# --------------------------------------------------------------------------- #


class TestStrategyPipeline:
    def test_only_runs_on_new_m5_bar(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        # Same df last_time on both calls — last bar timestamp doesn't advance.
        df = make_ohlc(n=30, last_time="2026-05-08T12:00:00Z")
        mgrs["ohlc_provider"].get.return_value = df

        run_pipeline = mocker.patch(
            "bot.main.run_strategy_pipeline", return_value=[],
        )
        # First call → new M5 → pipeline runs.
        b.run_iteration(NOW)
        # Second call same bar → no new M5 → pipeline doesn't run again.
        b.run_iteration(NOW + timedelta(seconds=2))
        assert run_pipeline.call_count == 1

    def test_zone_detected_calls_place_layered_orders(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        # Real ValidatedZone (the loosened-rules shape — all break-and-
        # close fields are None; SL/TP1 are computed in main from the
        # zone bounds + ohlc).
        zone = _make_validated_for_persistence(_PT.RBR)

        mocker.patch(
            "bot.main.run_strategy_pipeline", return_value=[zone],
        )
        # The local-peak finder needs at least one peak above Layer 1's
        # entry (= zone.top = 1900.5). Patch it directly to keep the
        # OHLC fixture flat and the test focused on the place-orders
        # branch.
        mocker.patch(
            "bot.main.find_nearest_local_peak", return_value=1907.5,
        )
        place = mocker.patch(
            "bot.main.place_layered_orders",
        )
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
            error_messages=[],
        )

        b.run_iteration(NOW)
        place.assert_called_once()
        # tp1_price flows through as a kwarg.
        assert place.call_args.kwargs["tp1_price"] == 1907.5
        assert b.state.placed_setup_count == 1

    def test_exposure_cap_blocks_further_placements(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        # Already at 3 active setups (== max).
        mgrs["position_tracker"].get_active_setups.return_value = [
            make_setup(), make_setup(), make_setup(),
        ]

        zone = mocker.MagicMock()
        zone.direction = "BUY"
        zone.top, zone.bottom = 1900.0, 1895.0
        mocker.patch(
            "bot.main.run_strategy_pipeline", return_value=[zone, zone],
        )
        place = mocker.patch("bot.main.place_layered_orders")
        b.run_iteration(NOW)
        # Already at cap → never tries to place.
        place.assert_not_called()

    def test_invalid_sl_skips_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = mocker.MagicMock()
        zone.direction = "BUY"
        zone.top, zone.bottom = 1900.0, 1895.0
        zone.is_imbalance, zone.is_strong_point = False, True
        zone.approach_count = 0
        zone.qualified_at = None
        zone.formed_at = pd.Timestamp("2026-05-08T12:00:00Z")

        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mgrs["sl_manager"].validate_sl_distance.return_value = SLValidation(
            is_valid=False, distance_points=2.0,
            is_too_close=True, is_too_far=False,
            error="SL distance 2.0 below minimum 5.0",
        )
        place = mocker.patch("bot.main.place_layered_orders")
        b.run_iteration(NOW)
        place.assert_not_called()


# --------------------------------------------------------------------------- #
# Cadences
# --------------------------------------------------------------------------- #


class TestCadences:
    def test_reconcile_runs_after_interval(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_reconcile = NOW - timedelta(seconds=400)  # past 5min
        b.run_iteration(NOW)
        mgrs["position_tracker"].reconcile_with_mt5.assert_called_once()

    def test_reconcile_skipped_within_interval(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_reconcile = NOW - timedelta(seconds=60)  # within 5min
        b.run_iteration(NOW)
        mgrs["position_tracker"].reconcile_with_mt5.assert_not_called()

    def test_detect_closed_runs_after_30s(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_detect_closed = NOW - timedelta(seconds=40)
        s = make_setup(status="ACTIVE")
        mgrs["position_tracker"].get_active_setups.return_value = [s]
        b.run_iteration(NOW)
        mgrs["position_tracker"].detect_closed_positions.assert_called_once()

    def test_heartbeat_emitted_after_interval(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock,
    ) -> None:
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_heartbeat = NOW - timedelta(seconds=400)
        b.run_iteration(NOW)
        # Heartbeat writes a bot_logs INFO row.
        msgs = [
            c for c in mock_supabase.log_event.call_args_list
            if c.args[1] == "heartbeat"
        ]
        assert len(msgs) == 1


# --------------------------------------------------------------------------- #
# Live config refresh
# --------------------------------------------------------------------------- #


class TestConfigRefresh:
    def test_kill_switch_picked_up_on_refresh(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock,
    ) -> None:
        b, _ = bot
        # Simulate dashboard flipping the switch on.
        mock_supabase.check_bot_config.side_effect = lambda key: {
            "kill_switch": True, "pause_until": None,
        }.get(key, None)
        # Force refresh by clearing last_config_refresh.
        b.state.last_config_refresh = None
        b.run_iteration(NOW)
        assert b.state.kill_switch is True

    def test_pause_until_iso_string_parsed(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock,
    ) -> None:
        b, _ = bot
        future = "2026-05-08T13:00:00Z"
        mock_supabase.check_bot_config.side_effect = lambda key: {
            "kill_switch": False, "pause_until": future,
        }.get(key, None)
        b.state.last_config_refresh = None
        b.run_iteration(NOW)
        assert b.state.pause_until == datetime(
            2026, 5, 8, 13, 0, tzinfo=timezone.utc,
        )

    def test_supabase_failure_keeps_last_known_value(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock,
    ) -> None:
        b, _ = bot
        b.state.kill_switch = True  # last known
        mock_supabase.check_bot_config.side_effect = RuntimeError("DB down")
        b.state.last_config_refresh = None
        # Doesn't crash; kill_switch unchanged.
        b.run_iteration(NOW)
        assert b.state.kill_switch is True


# --------------------------------------------------------------------------- #
# run() — the actual loop wrapper
# --------------------------------------------------------------------------- #


class TestRunLoop:
    def test_stop_exits_loop_calls_shutdown(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        # Make initialize a no-op (already set up by fixture).
        mocker.patch.object(b, "initialize")
        mocker.patch("bot.main.time.sleep")  # don't actually sleep

        call_count = {"n": 0}

        def fake_iteration(now):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                b.stop()

        mocker.patch.object(b, "run_iteration", side_effect=fake_iteration)
        b.run()

        assert call_count["n"] == 3
        # shutdown disconnects MT5.
        mock_mt5.disconnect.assert_called_once()

    def test_iteration_exception_does_not_break_loop(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        mocker.patch.object(b, "initialize")
        mocker.patch("bot.main.time.sleep")

        call_count = {"n": 0}

        def flaky_iteration(now):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            if call_count["n"] >= 3:
                b.stop()

        mocker.patch.object(b, "run_iteration", side_effect=flaky_iteration)
        b.run()
        # Survived the exception; ran 3 iterations.
        assert call_count["n"] == 3

    def test_shutdown_called_even_on_unexpected_run_error(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        mocker.patch.object(
            b, "initialize", side_effect=RuntimeError("auth"),
        )
        mocker.patch("bot.main.time.sleep")
        with pytest.raises(RuntimeError):
            b.run()
        # finally branch runs shutdown even when initialize blew up.
        mock_mt5.disconnect.assert_called_once()


# --------------------------------------------------------------------------- #
# main() entry point — .env loading
# --------------------------------------------------------------------------- #


class TestMainEntryPoint:
    """Regression guard for the ``.env`` loading fix.

    Pre-fix, ``python -m bot.main`` crashed with ``KeyError: 'MT5_LOGIN'``
    on a fresh install because main() called ``from_env()`` without
    first calling ``dotenv.load_dotenv()``. The shell-env path
    (Docker / VPS) worked; the dev workflow (local ``.env`` file)
    did not. The fix is two lines: ``from dotenv import load_dotenv``
    + ``load_dotenv()`` at the top of main(). These tests lock the
    invariants:

      1. load_dotenv() is called before either ``from_env`` call.
      2. main() doesn't crash when env vars are present (positive
         path).
    """

    def test_main_calls_load_dotenv_before_from_env(
        self, mocker: MockerFixture,
    ) -> None:
        # Order matters: load_dotenv MUST run BEFORE either from_env
        # call, otherwise the env vars from .env aren't visible.
        # Use a shared MagicMock to record relative call order.
        call_order: list[str] = []
        mocker.patch(
            "bot.main.load_dotenv",
            side_effect=lambda *a, **kw: call_order.append("load_dotenv"),
        )
        mocker.patch(
            "bot.main.MT5Connector.from_env",
            side_effect=lambda: (call_order.append("mt5_from_env") or
                                 mocker.MagicMock()),
        )
        mocker.patch(
            "bot.main.SupabaseLogger.from_env",
            side_effect=lambda: (call_order.append("supabase_from_env") or
                                 mocker.MagicMock()),
        )
        # Don't actually run the loop.
        mocker.patch("bot.main.Bot.run")
        mocker.patch("bot.main.signal.signal")

        main()

        assert call_order[0] == "load_dotenv", (
            f"load_dotenv must run before from_env calls; got order {call_order}"
        )
        assert "mt5_from_env" in call_order
        assert "supabase_from_env" in call_order

    def test_main_no_crash_with_env_present(
        self, mocker: MockerFixture,
    ) -> None:
        # Positive path: env vars are set (either from .env or shell),
        # main() constructs Bot and exits cleanly. Asserts Bot.run was
        # invoked once.
        mocker.patch("bot.main.load_dotenv")
        mocker.patch("bot.main.MT5Connector.from_env",
                     return_value=mocker.MagicMock())
        mocker.patch("bot.main.SupabaseLogger.from_env",
                     return_value=mocker.MagicMock())
        run_spy = mocker.patch("bot.main.Bot.run")
        mocker.patch("bot.main.signal.signal")
        main()
        run_spy.assert_called_once()


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


class TestDefaults:
    def test_loop_config_defaults(self) -> None:
        c = BotLoopConfig()
        assert c.symbol == "XAUUSD"
        # 1 Hz loop pacing — see BotLoopConfig docstring for rationale.
        # Tests that change this must justify why the bot needs faster
        # polling, given Supabase's ~10K-request HTTP/2 connection limit.
        assert c.main_loop_sleep_ms == 1000
        assert c.config_refresh_seconds == 30
        assert c.detect_closed_seconds == 30
        assert c.reconcile_seconds == 300
        assert c.heartbeat_seconds == 300
        assert c.max_simultaneous_setups == 3
        assert c.daily_loss_limit_pct == 10.0
        assert c.ohlc_count == 200
        assert c.ohlc_timeframe == "M5"


# --------------------------------------------------------------------------- #
# Persistence — pattern_type retains the real S&D code (PR #31, fix-up after
# Tommy's Q1 verification: dropped the legacy W/M mapping at the storage
# boundary so analytics can distinguish continuation vs reversal patterns).
# Requires migration 006 (relaxes the CHECK constraint to accept RBR/DBD/DBR/RBD).
# --------------------------------------------------------------------------- #


def _make_validated_for_persistence(pattern_type: _PT) -> _ValidatedZone:
    ts = pd.Timestamp("2026-05-08T12:00:00Z")
    direction = "BUY" if pattern_type in (_PT.RBR, _PT.DBR) else "SELL"
    impulse = _Impulse(
        direction="RALLY" if direction == "BUY" else "DROP",
        start_index=0, end_index=0,
        start_time=ts, end_time=ts,
        range_size=5.0, largest_body=5.0, candle_count=1,
    )
    base = _Base(
        start_index=1, end_index=1, candle_count=1,
        top=1900.5, bottom=1900.0, range_size=0.5, largest_body=0.5,
    )
    pattern = _Pattern(
        pattern_type=pattern_type,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction=direction,  # type: ignore[arg-type]
        formed_at=ts,
    )
    zone = _Zone(
        direction=direction,  # type: ignore[arg-type]
        top=1900.5, bottom=1900.0, formed_at=ts, source_pattern=pattern,
    )
    refined = _RefinedZone(
        direction=direction,  # type: ignore[arg-type]
        top=1900.5, bottom=1900.0, formed_at=ts, source_pattern=pattern,
        is_tradeable=True, rejection_reason=None, original_zone=zone,
    )
    return _ValidatedZone(
        direction=direction,  # type: ignore[arg-type]
        top=1900.5, bottom=1900.0, formed_at=ts, source_pattern=pattern,
        refined_zone=refined,
        is_strong_point=True, validation_failures=[],
        broken_swing=None, broken_at=None, sl_anchor_swing=None,
    )


class TestZoneToInputPersistsRealPatternCode:
    @pytest.mark.parametrize("pt", [_PT.RBR, _PT.DBD, _PT.DBR, _PT.RBD])
    def test_pattern_type_round_trips(self, pt: _PT) -> None:
        # The legacy W/M mapping is gone; we persist the actual S&D
        # pattern code so post-demo analytics can compare continuation
        # (RBR/DBD) vs reversal (DBR/RBD) performance.
        zi = _zone_to_input(_make_validated_for_persistence(pt))
        assert zi.pattern_type == pt.value
        assert zi.zone_type == "STRONG_POINT"


# --------------------------------------------------------------------------- #
# Zone lifecycle — per-bar pass that drives CONSUMED / VIOLATED / FLIPPED.
# Item 2 of the wicks+lifecycle PR.
# --------------------------------------------------------------------------- #


def _make_zone_row(
    *,
    direction: str = "BUY",
    top: float = 1905.0,
    bottom: float = 1900.0,
    status: str = "CONFIRMED",
    consumed_at: datetime | None = None,
    violated_at: datetime | None = None,
    flipped_at: datetime | None = None,
    flipped_direction: str | None = None,
):
    """Build a Zone read model for lifecycle tests."""
    from bot.logging.supabase_logger import Zone
    return Zone(
        id=uuid4(),
        symbol="XAUUSD",
        direction=direction,  # type: ignore[arg-type]
        zone_type="STRONG_POINT",
        pattern_type="RBR",
        top=Decimal(str(top)),
        bottom=Decimal(str(bottom)),
        approach_count=0,
        formed_at=NOW,
        status=status,  # type: ignore[arg-type]
        consumed_at=consumed_at,
        violated_at=violated_at,
        flipped_at=flipped_at,
        flipped_direction=flipped_direction,  # type: ignore[arg-type]
        created_at=NOW,
        updated_at=NOW,
    )


class TestZoneLifecycleLoop:
    """Bot._run_zone_lifecycle runs on every new M5 close.

    The strategy pipeline is patched to return [] in each test so we
    can isolate the lifecycle behaviour from setup placement.
    """

    def test_no_zones_no_writes(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        b.state.last_config_refresh = NOW
        mock_supabase.get_zones_by_status.return_value = []
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])

        b.run_iteration(NOW)

        mock_supabase.update_zone_status.assert_not_called()

    def test_wick_touch_consumes_buy_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW

        zone = _make_zone_row(direction="BUY", top=1905.0, bottom=1900.0)
        mock_supabase.get_zones_by_status.return_value = [zone]
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])

        # Build an OHLC frame whose LAST bar wicks INTO the zone.
        last_time = "2026-05-08T12:00:00Z"
        times = pd.date_range(end=last_time, periods=30, freq="5min", tz="UTC")
        opens = [1910.0] * 30
        highs = [1911.0] * 30
        lows = [1909.0] * 30
        closes = [1910.0] * 30
        # Last bar: low pokes down to 1902 (inside the 1900-1905 zone).
        lows[-1] = 1902.0
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [100] * 30},
            index=times,
        )
        mgrs["ohlc_provider"].get.return_value = df

        b.run_iteration(NOW)

        # Exactly one CONSUMED transition.
        calls = [
            c for c in mock_supabase.update_zone_status.call_args_list
            if c.args[1] == "CONSUMED"
        ]
        assert len(calls) == 1
        assert calls[0].args[0] == zone.id

    def test_body_close_below_buy_zone_violates(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, mgrs = bot
        b.state.last_config_refresh = NOW

        zone = _make_zone_row(direction="BUY", top=1905.0, bottom=1900.0)
        mock_supabase.get_zones_by_status.return_value = [zone]
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])

        last_time = "2026-05-08T12:00:00Z"
        times = pd.date_range(end=last_time, periods=30, freq="5min", tz="UTC")
        opens = [1910.0] * 30
        highs = [1911.0] * 30
        lows = [1909.0] * 30
        closes = [1910.0] * 30
        # Last bar: gaps through the zone — low 1890, close 1895
        # (CONSUMED via touch on the same bar, then VIOLATED on body).
        highs[-1] = 1910.0
        lows[-1] = 1890.0
        closes[-1] = 1895.0
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [100] * 30},
            index=times,
        )
        mgrs["ohlc_provider"].get.return_value = df

        b.run_iteration(NOW)

        statuses = [c.args[1] for c in mock_supabase.update_zone_status.call_args_list]
        # Both transitions land on the same bar — order: CONSUMED then VIOLATED.
        assert statuses == ["CONSUMED", "VIOLATED"]

    def test_already_consumed_zone_only_violates(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Zone is already CONSUMED from a prior bar. Today's bar
        # body-closes below it. Expect only the CONSUMED → VIOLATED
        # transition (not a redundant CONSUMED rewrite).
        b, mgrs = bot
        b.state.last_config_refresh = NOW

        zone = _make_zone_row(
            direction="BUY", top=1905.0, bottom=1900.0,
            status="CONSUMED", consumed_at=NOW,
        )
        mock_supabase.get_zones_by_status.return_value = [zone]
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])

        times = pd.date_range(
            end="2026-05-08T12:00:00Z", periods=30, freq="5min", tz="UTC",
        )
        opens = [1910.0] * 30
        highs = [1911.0] * 30
        lows = [1909.0] * 30
        closes = [1910.0] * 30
        closes[-1] = 1890.0  # body close below zone bottom
        lows[-1] = 1890.0
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [100] * 30},
            index=times,
        )
        mgrs["ohlc_provider"].get.return_value = df

        b.run_iteration(NOW)

        statuses = [c.args[1] for c in mock_supabase.update_zone_status.call_args_list]
        assert statuses == ["VIOLATED"]

    def test_lifecycle_failure_does_not_block_strategy(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # If get_zones_by_status blows up, the strategy pipeline still
        # runs (lifecycle is best-effort).
        b, _ = bot
        b.state.last_config_refresh = NOW
        mock_supabase.get_zones_by_status.side_effect = RuntimeError("DB down")
        run_pipeline = mocker.patch(
            "bot.main.run_strategy_pipeline", return_value=[],
        )

        b.run_iteration(NOW)  # no raise

        run_pipeline.assert_called_once()


# --------------------------------------------------------------------------- #
# Dedup pre-flight — _try_place_setup skips when a zone with overlapping
# bounds is already CONSUMED/VIOLATED/FLIPPED.
# --------------------------------------------------------------------------- #


class TestDedupSkip:
    def test_skip_when_consumed_zone_overlaps(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)
        # Anchor swing so we get past the SL anchor check.
        ts = pd.Timestamp("2026-05-08T12:00:00Z")
        anchor = Swing(index=0, time=ts, price=1880.0, kind="LOW")
        zone = _ValidatedZone(
            direction=zone.direction, top=zone.top, bottom=zone.bottom,
            formed_at=zone.formed_at, source_pattern=zone.source_pattern,
            refined_zone=zone.refined_zone,
            is_strong_point=True, validation_failures=[],
            broken_swing=None, broken_at=None,
            sl_anchor_swing=anchor,
        )

        # Existing CONSUMED zone with overlapping bounds.
        existing = _make_zone_row(
            direction="BUY", top=1900.6, bottom=1900.0,
            status="CONSUMED", consumed_at=NOW,
        )
        # Two calls: lifecycle pass (no transitions wanted) + dedup.
        # The lifecycle pass might process the existing CONSUMED zone
        # and try to transition it; we don't care for this assertion.
        mock_supabase.get_zones_by_status.return_value = [existing]
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_no_overlap_proceeds(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)

        # Existing CONSUMED zone far away — no overlap.
        existing = _make_zone_row(
            direction="BUY", top=1800.0, bottom=1795.0,
            status="CONSUMED", consumed_at=NOW,
        )
        mock_supabase.get_zones_by_status.return_value = [existing]
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1862.5, tp1_price=1907.5,
            error_messages=[],
        )

        b.run_iteration(NOW)

        place.assert_called_once()

    def test_skip_when_flipped_zone_overlaps(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # FLIPPED zones in the same direction also block re-trade
        # (the flipped zone's original direction = the candidate's).
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)
        ts = pd.Timestamp("2026-05-08T12:00:00Z")
        anchor = Swing(index=0, time=ts, price=1880.0, kind="LOW")
        zone = _ValidatedZone(
            direction=zone.direction, top=zone.top, bottom=zone.bottom,
            formed_at=zone.formed_at, source_pattern=zone.source_pattern,
            refined_zone=zone.refined_zone,
            is_strong_point=True, validation_failures=[],
            broken_swing=None, broken_at=None,
            sl_anchor_swing=anchor,
        )

        existing = _make_zone_row(
            direction="BUY",  # Original direction
            top=1900.5, bottom=1900.0,
            status="FLIPPED",
            violated_at=NOW, flipped_at=NOW, flipped_direction="SELL",
        )
        mock_supabase.get_zones_by_status.return_value = [existing]
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()


# --------------------------------------------------------------------------- #
# Loosened-rules entry flow (May 2026)
# Test the new TP1 path + zone-bound SL flow that lives in _try_place_setup.
# --------------------------------------------------------------------------- #


class TestLoosenedRulesEntry:
    def test_no_local_peak_skips_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # When find_nearest_local_peak returns None, we skip the zone
        # — no log_zone, no place_layered_orders.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=None)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()
        assert b.state.placed_setup_count == 0

    def test_sl_formula_uses_zone_bound(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # Loosened-rules SL = zone.bottom - sl_buffer_points (BUY).
        # Default buffer 17.5 → SL = 1900.0 - 17.5 = 1882.5
        # (zone built by _make_validated_for_persistence: 1900.0 - 1900.5).
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
            error_messages=[],
        )

        b.run_iteration(NOW)

        place.assert_called_once()
        assert place.call_args.kwargs["sl_price"] == pytest.approx(1882.5)

    def test_tp1_lookup_uses_layer_1_entry_for_buy(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # BUY → reference = zone.top.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)  # BUY, top=1900.5
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        peak = mocker.patch(
            "bot.main.find_nearest_local_peak", return_value=1907.5,
        )
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
                error_messages=[],
            )
        )

        b.run_iteration(NOW)

        # Verify the entry_price passed into the peak finder is the
        # zone's Layer 1 entry (= zone.top for BUY).
        call_kwargs = peak.call_args.kwargs
        assert call_kwargs["entry_price"] == pytest.approx(1900.5)
        assert call_kwargs["direction"] == "BUY"

    def test_tp1_lookup_uses_layer_1_entry_for_sell(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # SELL → reference = zone.bottom.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.DBD)  # SELL, bottom=1900.0
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        peak = mocker.patch(
            "bot.main.find_nearest_local_peak", return_value=1890.0,
        )
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1918.0, tp1_price=1890.0,
                error_messages=[],
            )
        )

        b.run_iteration(NOW)

        call_kwargs = peak.call_args.kwargs
        assert call_kwargs["entry_price"] == pytest.approx(1900.0)
        assert call_kwargs["direction"] == "SELL"

    def test_tp1_lookback_threaded_from_config(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # Operator can tune the lookback via StrategyPipelineConfig and
        # the value flows through to the peak finder.
        from bot.strategy.pipeline import StrategyPipelineConfig
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        b.strategy_pipeline_config = StrategyPipelineConfig(
            tp1_local_peak_lookback_bars=123,
        )

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        peak = mocker.patch(
            "bot.main.find_nearest_local_peak", return_value=1907.5,
        )
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
                error_messages=[],
            )
        )

        b.run_iteration(NOW)

        assert peak.call_args.kwargs["lookback_bars"] == 123
