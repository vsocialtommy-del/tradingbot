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
from bot.exits.tp_manager import TPManager, LayerCloseResult
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

    bot.tp_manager = mocker.MagicMock(spec=TPManager)
    bot.tp_manager.check.return_value = []

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
        "tp_manager": bot.tp_manager,
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
        mgrs["tp_manager"].check.assert_not_called()

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
        mgrs["tp_manager"].check.assert_not_called()


class TestRunIterationActiveSetups:
    def test_tp1_manager_called_per_active_setup(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        s1, s2 = make_setup(), make_setup()
        mgrs["position_tracker"].get_active_setups.return_value = [s1, s2]
        b.run_iteration(NOW)
        # ACTIVE → tp1_manager.check called once per setup.
        assert mgrs["tp_manager"].check.call_count == 2

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
        assert mgrs["tp_manager"].check.call_count == 1

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

    def test_tp_close_increments_counter(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        s = make_setup(status="ACTIVE")
        mgrs["position_tracker"].get_active_setups.return_value = [s]
        # PR #41: tp_manager.check returns a list of LayerCloseResult.
        # Each entry increments the (still legacy-named) tp1_count
        # counter so the heartbeat reflects "TP layer closes this run".
        mgrs["tp_manager"].check.return_value = [
            LayerCloseResult(
                setup_id=s.id, trade_id=uuid4(),
                layer_number=1, tp_price=1907.0, close_price=1907.0,
                cascaded_sl=1900.0, needs_next_tp_recompute=False,
            ),
        ]
        b.run_iteration(NOW)
        assert b.state.tp1_count == 1

    def test_tp1_manager_exception_does_not_kill_iteration(
        self, bot: tuple[Bot, dict[str, MagicMock]],
    ) -> None:
        b, mgrs = bot
        s = make_setup(status="ACTIVE")
        mgrs["position_tracker"].get_active_setups.return_value = [s]
        mgrs["tp_manager"].check.side_effect = RuntimeError("boom")
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
        mgrs["tp_manager"].check.assert_called_once()


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
        # 1000 M5 bars ≈ 3.5 days of history — wide enough for zones
        # that formed 24-48 h ago to still be detectable. Pre-2026-05
        # default was 200 (~16 h). See BotLoopConfig docstring.
        assert c.ohlc_count == 1000
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
# PR #45: dedup skipped for SnD Flip retrades.
#
# A flipped zone trades in its ``flipped_direction``, which is by S&D
# construction the same direction as the supply/demand that was broken
# to form the original zone. That broken counter-direction zone is
# usually still in the DB at status=VIOLATED at the exact same price
# band, so the dedup pre-flight reliably trips on it and blocks the
# flipped retrade. The flipped path has its own re-trade guards
# (status='FLIPPED' filter on load, ``flipped_zone_body_broken_since_flip``,
# FLIPPED → ACTIVE on placement) so we skip dedup for that branch.
# --------------------------------------------------------------------------- #


class TestFlippedRetradeDedupSkip:
    def test_flipped_retrade_skips_dedup_against_counter_direction_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mock_mt5: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        # Reproduces the production case: zone 77fceff6 (BUY DBR
        # FLIPPED to SELL at 4704.32-4711.47). A previously-VIOLATED
        # SELL supply sits in the DB at the same price band — that's
        # the row dedup picks up. With PR #45 the flipped retrade
        # proceeds past dedup and places.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        # Price already at zone.bottom so the SELL gate fires
        # immediately on the same iteration (per-tick placement).
        mock_mt5.get_current_price.return_value = {
            "bid": 4704.31, "ask": 4704.32, "time": NOW, "time_msc": 0,
        }

        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=4711.47, bottom=4704.32,
            flipped_at=NOW - timedelta(hours=1),
        )
        # The dedup blocker: the supply that was broken when the
        # original BUY demand formed at this band is still in the DB.
        prior_supply_violated = _make_zone_row(
            direction="SELL",  # same direction as the flipped retrade
            top=4711.47, bottom=4704.32,
            status="VIOLATED",
            violated_at=NOW - timedelta(hours=2),
        )

        def fake_get(statuses):
            if "FLIPPED" in statuses and len(statuses) == 1:
                return [fz]
            return [prior_supply_violated]

        mock_supabase.get_zones_by_status.side_effect = fake_get
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch(
            "bot.main.find_nearest_local_peak", return_value=4690.0,
        )
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=4728.97, tp1_price=4690.0,
            error_messages=[],
        )

        b.run_iteration(NOW)

        # Placement fires — dedup did NOT block the flipped retrade.
        place.assert_called_once()
        assert b._pending_flipped_zones == []

    def test_pipeline_path_still_blocked_by_dedup(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mock_mt5: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        # Pipeline (fresh-detection) path is unchanged: a CONSUMED
        # same-direction zone at the same band still blocks placement.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        # Price already at zone so the gate isn't the reason we skip.
        mock_mt5.get_current_price.return_value = {
            "bid": 1900.5, "ask": 1900.6, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.RBR)
        existing_consumed = _make_zone_row(
            direction="BUY",  # same direction as the candidate
            top=1900.5, bottom=1900.0,
            status="CONSUMED", consumed_at=NOW - timedelta(hours=1),
        )

        def fake_get(statuses):
            # No FLIPPED zones to load; the dedup lookup gets the
            # CONSUMED row.
            if "FLIPPED" in statuses and len(statuses) == 1:
                return []
            return [existing_consumed]

        mock_supabase.get_zones_by_status.side_effect = fake_get
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_try_place_pending_passes_flag_per_queue(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # Direct assertion on the call site: pipeline-queue entries
        # get is_flipped_retrade=False; flipped-queue entries get
        # is_flipped_retrade=True.
        b, _ = bot
        b._latest_ohlc = make_ohlc()

        pipeline_zone = _make_validated_for_persistence(_PT.RBR)
        flipped_zone = _make_validated_for_persistence(_PT.RBR)
        pipeline_id, flipped_id = uuid4(), uuid4()
        b._pending_pipeline_zones = [(pipeline_zone, pipeline_id)]
        b._pending_flipped_zones = [(flipped_zone, flipped_id)]

        # Stub the placement so it doesn't drain the queues; we only
        # care about the call-site kwargs.
        try_place = mocker.patch.object(
            b, "_try_place_setup", return_value=False,
        )

        b._try_place_pending(bid=1900.0, ask=1900.1)

        calls = try_place.call_args_list
        assert len(calls) == 2
        # First call: pipeline queue.
        assert calls[0].args[2] == pipeline_id
        assert calls[0].kwargs["is_flipped_retrade"] is False
        # Second call: flipped queue.
        assert calls[1].args[2] == flipped_id
        assert calls[1].kwargs["is_flipped_retrade"] is True


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

        # PR #41: find_nearest_local_peak is called 3× (TP1/TP2/TP3
        # chain). Assert on the FIRST call — that's the TP1 lookup
        # that uses Layer 1's entry as the reference.
        first_call = peak.call_args_list[0]
        assert first_call.kwargs["entry_price"] == pytest.approx(1900.5)
        assert first_call.kwargs["direction"] == "BUY"

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

        # PR #41: first call = TP1 lookup with Layer 1's entry.
        first_call = peak.call_args_list[0]
        assert first_call.kwargs["entry_price"] == pytest.approx(1900.0)
        assert first_call.kwargs["direction"] == "SELL"

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

        # All 3 TP-chain calls share the same lookback config.
        assert peak.call_args_list[0].kwargs["lookback_bars"] == 123


# --------------------------------------------------------------------------- #
# Zone persistence (PR #36 follow-up: log_zone hoisted out of
# _try_place_setup, idempotency cache, startup hydration).
# --------------------------------------------------------------------------- #


def _make_non_tradeable_validated(direction: str = "BUY") -> _ValidatedZone:
    """ValidatedZone with refined.is_tradeable=False (size-filter reject)."""
    base = _make_validated_for_persistence(_PT.RBR if direction == "BUY" else _PT.DBD)
    rejected_refined = _RefinedZone(
        direction=base.refined_zone.direction,
        top=base.refined_zone.top,
        bottom=base.refined_zone.bottom,
        formed_at=base.refined_zone.formed_at,
        source_pattern=base.refined_zone.source_pattern,
        is_tradeable=False,
        rejection_reason="ZONE_TOO_NARROW",
        original_zone=base.refined_zone.original_zone,
    )
    return _ValidatedZone(
        direction=base.direction, top=base.top, bottom=base.bottom,
        formed_at=base.formed_at, source_pattern=base.source_pattern,
        refined_zone=rejected_refined,
        is_strong_point=False,
        validation_failures=["NOT_TRADEABLE"],
        broken_swing=None, broken_at=None, sl_anchor_swing=None,
    )


class TestZonePersistence:
    def test_tradeable_zone_persisted_with_status_confirmed(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
                error_messages=[],
            )
        )

        b.run_iteration(NOW)

        # log_zone fired exactly once, with status defaulted to CONFIRMED.
        mock_supabase.log_zone.assert_called_once()
        zi = mock_supabase.log_zone.call_args.args[0]
        assert zi.status == "CONFIRMED"
        assert zi.direction == zone.direction

    def test_non_tradeable_zone_not_persisted(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Pipeline now emits non-tradeable zones too; main.py persists
        # only the tradeable subset.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        rejected = _make_non_tradeable_validated()
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[rejected])

        b.run_iteration(NOW)

        mock_supabase.log_zone.assert_not_called()

    def test_same_zone_persisted_once_across_iterations(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        b.state.last_config_refresh = NOW

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
                error_messages=[],
            )
        )

        # Three iterations, each with a fresh M5 bar advance so the
        # strategy pipeline runs every time.
        for i in range(3):
            b.state.last_m5_bar_time = None  # force "new bar"
            b.run_iteration(NOW + timedelta(seconds=i))

        # Pipeline ran 3 times but the same zone gets logged only once.
        assert mock_supabase.log_zone.call_count == 1

    def test_try_place_setup_no_longer_calls_log_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Regression guard: the call moved upstream into
        # _persist_zone_if_new. _try_place_setup must not log_zone
        # again (would race the cache + create duplicates).
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
                error_messages=[],
            )
        )

        b.run_iteration(NOW)

        # Exactly one log_zone call (from _persist_zone_if_new), not two.
        assert mock_supabase.log_zone.call_count == 1

    def test_initialize_hydrates_cache_from_db(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        # Pre-populate the DB with a zone that matches the one the
        # pipeline will emit later.
        zone = _make_validated_for_persistence(_PT.RBR)
        existing_id = uuid4()
        existing = _make_zone_row(
            direction=zone.direction,
            top=float(zone.top),
            bottom=float(zone.bottom),
            status="CONFIRMED",
        )
        # Override the auto-generated id so we can verify it's the one
        # used downstream.
        existing = existing.model_copy(update={"id": existing_id})
        mock_supabase.get_zones_by_status.return_value = [existing]

        # initialize() runs reconcile_with_mt5 and then hydrates.
        b.initialize()

        # Cache now contains the existing zone — re-detecting it
        # should NOT trigger a new log_zone call.
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
                error_messages=[],
            )
        )
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        b.run_iteration(NOW)

        mock_supabase.log_zone.assert_not_called()

    def test_lifecycle_scanner_sees_persisted_zones(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # End-to-end coupling test: pipeline persists a zone, then the
        # NEXT iteration's lifecycle pass loads it and (with a touching
        # bar) transitions it to CONSUMED.
        b, mgrs = bot
        b.state.last_config_refresh = NOW

        zone = _make_validated_for_persistence(_PT.RBR)
        new_zone_id = uuid4()
        mock_supabase.log_zone.return_value = {"id": str(new_zone_id)}

        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        mocker.patch("bot.main.place_layered_orders").return_value = (
            mocker.MagicMock(
                status="PLACED", setup_id=uuid4(),
                layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
                error_messages=[],
            )
        )

        # Iteration 1: pipeline persists the zone.
        b.state.last_m5_bar_time = None
        b.run_iteration(NOW)
        mock_supabase.log_zone.assert_called_once()

        # Iteration 2: lifecycle scanner now sees the zone in DB and a
        # bar that wicks into it consumes it.
        persisted = _make_zone_row(
            direction=zone.direction,
            top=float(zone.top), bottom=float(zone.bottom),
            status="CONFIRMED",
        ).model_copy(update={"id": new_zone_id})
        mock_supabase.get_zones_by_status.return_value = [persisted]

        # OHLC with last bar's low inside the zone.
        last_time = "2026-05-08T12:05:00Z"
        times = pd.date_range(end=last_time, periods=30, freq="5min", tz="UTC")
        opens = [1910.0] * 30
        highs = [1911.0] * 30
        lows = [1909.0] * 30
        closes = [1910.0] * 30
        lows[-1] = 1900.2  # wicks into zone [1900.0, 1900.5]
        df_touch = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [100] * 30},
            index=times,
        )
        mgrs["ohlc_provider"].get.return_value = df_touch

        b.state.last_m5_bar_time = None
        b.run_iteration(NOW + timedelta(seconds=1))

        # Zone now transitioned to CONSUMED.
        consumed_calls = [
            c for c in mock_supabase.update_zone_status.call_args_list
            if c.args[1] == "CONSUMED"
        ]
        assert len(consumed_calls) == 1
        assert consumed_calls[0].args[0] == new_zone_id


# --------------------------------------------------------------------------- #
# SnD Flip trading (PR #38) — end-to-end side path that trades FLIPPED
# zones in their flipped_direction.
# --------------------------------------------------------------------------- #


def _make_flipped_zone_row(
    *,
    original_direction: str = "BUY",
    flipped_direction: str = "SELL",
    top: float = 1905.0,
    bottom: float = 1900.0,
    flipped_at: datetime | None = None,
):
    """Build a Zone read model in status=FLIPPED with all the
    PR #35 lifecycle columns populated."""
    from bot.logging.supabase_logger import Zone
    if flipped_at is None:
        flipped_at = NOW
    return Zone(
        id=uuid4(),
        symbol="XAUUSD",
        direction=original_direction,  # type: ignore[arg-type]
        zone_type="STRONG_POINT",
        pattern_type="RBR" if original_direction == "BUY" else "DBD",
        top=Decimal(str(top)),
        bottom=Decimal(str(bottom)),
        approach_count=0,
        formed_at=NOW - timedelta(hours=2),
        status="FLIPPED",
        consumed_at=None,
        violated_at=NOW - timedelta(minutes=30),
        flipped_at=flipped_at,
        flipped_direction=flipped_direction,  # type: ignore[arg-type]
        created_at=NOW - timedelta(hours=2),
        updated_at=NOW,
    )


class TestFlippedZoneTrading:
    """The bot loads FLIPPED zones from DB on each pipeline run and
    treats them as tradeable opportunities in flipped_direction.
    First retest fires a setup; the FLIPPED zone is FK-ed to the
    setup so position_tracker's promotion hook drives FLIPPED →
    ACTIVE."""

    def test_flipped_zone_fires_trade_in_flipped_direction(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        # A demand zone (BUY) that's been flipped to SELL — now a
        # supply candidate.
        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=1905.0, bottom=1900.0,
            flipped_at=NOW - timedelta(hours=1),
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )

        # Pipeline returns no fresh zones — the only candidate is the
        # flipped one.
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        # TP1 in the flipped direction (SELL) — below zone.bottom.
        peak = mocker.patch(
            "bot.main.find_nearest_local_peak", return_value=1890.0,
        )
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1922.5, tp1_price=1890.0,
            error_messages=[],
        )

        b.run_iteration(NOW)

        # Setup placed with the flipped direction's geometry.
        place.assert_called_once()
        call = place.call_args
        # zone arg is the synthesised view: direction=SELL, top=1905, bottom=1900
        synth = call.args[0]
        assert synth.direction == "SELL"
        assert synth.top == 1905.0
        assert synth.bottom == 1900.0
        # zone_id arg is the FLIPPED zone's own UUID — we don't re-persist.
        assert call.args[1] == fz.id
        # SL = zone.top + 17.5 for SELL = 1922.5
        assert call.kwargs["sl_price"] == pytest.approx(1922.5)
        # TP1 search ran in the flipped direction with the SELL entry
        # reference (= zone.bottom). PR #41: subsequent calls in the
        # same iteration are for TP2 / TP3 — assert on the first call.
        first_call = peak.call_args_list[0]
        assert first_call.kwargs["direction"] == "SELL"
        assert first_call.kwargs["entry_price"] == pytest.approx(1900.0)

    def test_body_break_since_flip_rejects_dead_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Zone flipped at T-1h. A bar between then and now body-closed
        # past zone.top (wrong side for the flipped SELL). The
        # candidate should be filtered out before any placement
        # attempt.
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        flipped_at_ts = pd.Timestamp("2026-05-08T11:00:00Z")
        # OHLC: 30 bars ending at NOW (2026-05-08T12:00:00Z), one bar
        # after flipped_at body-closes above zone.top for our SELL
        # candidate (top=1905, body close at 1910).
        times = pd.date_range(
            end="2026-05-08T12:00:00Z", periods=30, freq="5min", tz="UTC",
        )
        opens = [1900.0] * 30
        highs = [1901.0] * 30
        lows = [1899.0] * 30
        closes = [1900.0] * 30
        # Pick a bar strictly AFTER flipped_at to embed the break.
        for i, t in enumerate(times):
            if t > flipped_at_ts:
                closes[i] = 1910.0  # close above zone.top
                highs[i] = 1911.0
                break
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [100] * 30},
            index=times,
        )
        mgrs["ohlc_provider"].get.return_value = df

        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=1905.0, bottom=1900.0,
            flipped_at=flipped_at_ts.to_pydatetime(),
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        # Body-broken since flip → no placement attempt.
        place.assert_not_called()

    def test_break_before_flip_does_not_disqualify(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Symmetric to the above: the break happened BEFORE flipped_at
        # — it's pre-flip history. Should NOT disqualify the trade.
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        flipped_at_ts = pd.Timestamp("2026-05-08T11:30:00Z")
        times = pd.date_range(
            end="2026-05-08T12:00:00Z", periods=30, freq="5min", tz="UTC",
        )
        opens = [1900.0] * 30
        highs = [1901.0] * 30
        lows = [1899.0] * 30
        closes = [1900.0] * 30
        # Pre-flip break: pick a bar before flipped_at and put a close above top.
        for i, t in enumerate(times):
            if t < flipped_at_ts:
                closes[i] = 1910.0
                highs[i] = 1911.0
                break
        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [100] * 30},
            index=times,
        )
        mgrs["ohlc_provider"].get.return_value = df

        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=1905.0, bottom=1900.0,
            flipped_at=flipped_at_ts.to_pydatetime(),
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1890.0)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1922.5, tp1_price=1890.0,
            error_messages=[],
        )

        b.run_iteration(NOW)

        # Pre-flip break ignored → placement proceeds.
        place.assert_called_once()

    def test_no_tp1_peak_skips_flipped_trade(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # The TP1 skip path in _try_place_setup also applies to flipped
        # trades — no local peak in the lookback → no placement.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        fz = _make_flipped_zone_row(flipped_at=NOW - timedelta(hours=1))
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=None)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_exposure_cap_blocks_flipped_trades_too(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Both pipeline + flipped candidates compete for the same
        # max_simultaneous_setups budget. With 3 already active, no
        # new placements — including flipped.
        b, mgrs = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mgrs["position_tracker"].get_active_setups.return_value = [
            make_setup(), make_setup(), make_setup(),
        ]

        fz = _make_flipped_zone_row(flipped_at=NOW - timedelta(hours=1))
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1890.0)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_no_flipped_zones_in_db_means_no_extra_work(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # If get_zones_by_status(['FLIPPED']) returns [], we don't
        # attempt to construct any candidates.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        mock_supabase.get_zones_by_status.return_value = []
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_synthesised_view_uses_flipped_at_as_formed_at(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Q3 lock-in: formed_at on the synthesised stub is flipped_at.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None

        flipped_at_ts = NOW - timedelta(hours=2)
        # Use a zone where the default fixture OHLC (closes=1900) sits
        # inside the bounds — body-break check should NOT disqualify
        # the candidate. (flipped BUY rejects on close < bottom.)
        fz = _make_flipped_zone_row(
            flipped_at=flipped_at_ts,
            original_direction="SELL", flipped_direction="BUY",
            top=1901.0, bottom=1895.0,
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1925.0)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1887.5, tp1_price=1925.0,
            error_messages=[],
        )

        b.run_iteration(NOW)

        synth = place.call_args.args[0]
        # pd.Timestamp(flipped_at_ts) — compare directly.
        assert synth.formed_at == pd.Timestamp(flipped_at_ts)
        # Direction is the flipped direction.
        assert synth.direction == "BUY"


# --------------------------------------------------------------------------- #
# _flipped_zone_as_validated — module-level helper unit tests
# --------------------------------------------------------------------------- #


class TestFlippedZoneAsValidated:
    def test_builds_view_with_flipped_direction(self) -> None:
        from bot.main import _flipped_zone_as_validated
        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=1905.0, bottom=1900.0,
        )
        view = _flipped_zone_as_validated(fz)
        assert view.direction == "SELL"
        assert view.top == 1905.0
        assert view.bottom == 1900.0
        assert view.is_strong_point is True
        assert view.refined_zone.is_tradeable is True
        # Break-and-close fields are stubbed None (loosened-rules shape).
        assert view.broken_swing is None
        assert view.broken_at is None
        assert view.sl_anchor_swing is None

    def test_raises_when_zone_not_properly_flipped(self) -> None:
        from bot.main import _flipped_zone_as_validated
        fz = _make_flipped_zone_row().model_copy(update={
            "flipped_direction": None,
        })
        with pytest.raises(ValueError, match="not properly FLIPPED"):
            _flipped_zone_as_validated(fz)


# --------------------------------------------------------------------------- #
# Price-vs-zone gate (Bug 2 fix) — defer placement when current price
# hasn't reached the planned Layer 1 entry yet. Applies uniformly to
# pipeline-detected and flipped zones. The "overshoot past far edge"
# case stays with order_manager._detect_gap_through.
# --------------------------------------------------------------------------- #


class TestPriceVsZoneGate:
    def test_buy_pipeline_defers_when_price_above_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Default mock returns bid=1900.0. For a BUY zone [1895, 1900.5],
        # bid (1900.0) ≤ zone.top (1900.5) → would normally fire. Move
        # the tick well above the zone → gate defers.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1925.0, "ask": 1925.1, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_sell_pipeline_defers_when_price_below_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # SELL zone [1900, 1900.5]. ask = 1850 → below zone.bottom →
        # gate defers (price hasn't risen to the supply zone yet).
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1849.9, "ask": 1850.0, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.DBD)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1840.0)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_buy_pipeline_fires_when_price_at_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mocker: MockerFixture,
    ) -> None:
        # Default fixture has bid=1900.0; BUY zone top=1900.5 → bid ≤
        # zone.top → gate passes. The default flow runs.
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

    def test_sell_pipeline_fires_when_price_at_zone_bottom(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # SELL Layer 1 = zone.bottom = 1900.0. ask = 1900.0 → ask ≥
        # zone.bottom → gate passes.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1899.9, "ask": 1900.0, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.DBD)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1890.0)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1918.0, tp1_price=1890.0,
            error_messages=[],
        )

        b.run_iteration(NOW)

        place.assert_called_once()

    def test_flipped_buy_respects_gate(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Originally-SELL zone flipped to BUY. Layer 1 = zone.top.
        # Price far above zone → defer (would have been catastrophic
        # before this fix: BUY at current price way above the zone).
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1925.0, "ask": 1925.1, "time": NOW, "time_msc": 0,
        }

        fz = _make_flipped_zone_row(
            original_direction="SELL", flipped_direction="BUY",
            top=1901.0, bottom=1895.0,
            flipped_at=NOW - timedelta(hours=1),
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_flipped_sell_respects_gate(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Originally-BUY zone flipped to SELL. Layer 1 = zone.bottom.
        # Price far below zone → defer. This is the exact scenario from
        # the production incident: zone at 4733-4739, price at 4713 →
        # would have fired SELL at 4713 (20 pts of negative slippage).
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1849.9, "ask": 1850.0, "time": NOW, "time_msc": 0,
        }

        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=1905.0, bottom=1900.0,
            flipped_at=NOW - timedelta(hours=1),
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1890.0)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()

    def test_flipped_sell_fires_when_price_reaches_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Same flipped-SELL setup as above, but price has risen to
        # zone.bottom. Gate now passes; setup fires.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1899.9, "ask": 1900.0, "time": NOW, "time_msc": 0,
        }

        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=1905.0, bottom=1900.0,
            flipped_at=NOW - timedelta(hours=1),
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1890.0)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1922.5, tp1_price=1890.0,
            error_messages=[],
        )

        b.run_iteration(NOW)

        place.assert_called_once()
        # Sanity: the SELL direction propagates through.
        assert place.call_args.args[0].direction == "SELL"


# --------------------------------------------------------------------------- #
# PR #44 — per-tick placement
# Detection runs on M5 close and populates ``self._pending_*``; the
# per-tick ``_try_place_pending`` fires placement when price reaches
# a zone, regardless of M5 close. Solves the mid-bar-wick race.
# --------------------------------------------------------------------------- #


class TestPerTickPlacement:
    def test_pending_list_populated_on_m5_close_without_placement(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # First tick: M5 close fires, pipeline returns one zone.
        # Price is well above the zone → gate defers, zone goes onto
        # the pending list. No placement attempt this iteration.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1925.0, "ask": 1925.1, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()
        # Zone now sitting on the queue for future tick evaluation.
        assert len(b._pending_pipeline_zones) == 1

    def test_subsequent_tick_with_price_at_zone_fires_placement(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Iteration 1: detection populates pending; price above zone.
        # Iteration 2: same M5 bar still — price wicks down to zone
        # top → gate passes → placement fires. No new M5 close.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1925.0, "ask": 1925.1, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1882.5, tp1_price=1907.5,
            error_messages=[],
        )

        # Iter 1 — detection only.
        b.run_iteration(NOW)
        place.assert_not_called()
        assert len(b._pending_pipeline_zones) == 1

        # Iter 2 — same M5 bar (last_m5_bar_time stays set so the
        # M5-close gate is False; only per-tick placement runs).
        # Tick now puts bid AT zone.top.
        mock_mt5.get_current_price.return_value = {
            "bid": 1900.5, "ask": 1900.6, "time": NOW, "time_msc": 0,
        }
        b.run_iteration(NOW + timedelta(seconds=1))

        place.assert_called_once()
        # Successful placement pops the zone off the queue.
        assert b._pending_pipeline_zones == []

    def test_pending_zone_survives_across_ticks_until_placed(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mocker: MockerFixture,
    ) -> None:
        # Many ticks with price stubbornly above the zone — pending
        # entry persists. No placement, no spurious DB writes.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1925.0, "ask": 1925.1, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")

        for i in range(5):
            b.run_iteration(NOW + timedelta(seconds=i))

        place.assert_not_called()
        # The zone is still on the queue after 5 ticks.
        assert len(b._pending_pipeline_zones) == 1

    def test_pending_list_refreshes_on_next_m5_close(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        # An M5 close re-runs detection and rebuilds the pending
        # list from scratch. Same zone detected on both closes → one
        # entry on the queue (idempotency via the persistence cache).
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        mock_mt5.get_current_price.return_value = {
            "bid": 1925.0, "ask": 1925.1, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.RBR)
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        mocker.patch("bot.main.place_layered_orders")

        # First M5 close: pending = 1.
        b.run_iteration(NOW)
        assert len(b._pending_pipeline_zones) == 1

        # Force a "new M5 bar" by advancing the fixture's last_time
        # and resetting the bot's bar-tracking state.
        next_bar_df = make_ohlc(n=30, last_time="2026-05-08T12:05:00Z")
        bot[1]["ohlc_provider"].get.return_value = next_bar_df
        b.state.last_m5_bar_time = None  # treat next call as a new bar
        b.run_iteration(NOW + timedelta(minutes=5))

        # Same physical zone → idempotency cache hit → one entry.
        assert len(b._pending_pipeline_zones) == 1

    def test_flipped_candidate_fires_on_tick_when_price_reaches_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_supabase: MagicMock, mock_mt5: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        # Flipped-zone path also rides the pending queue.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        # Price starts away from the zone.
        mock_mt5.get_current_price.return_value = {
            "bid": 1849.9, "ask": 1850.0, "time": NOW, "time_msc": 0,
        }

        fz = _make_flipped_zone_row(
            original_direction="BUY", flipped_direction="SELL",
            top=1905.0, bottom=1900.0,
            flipped_at=NOW - timedelta(hours=1),
        )
        mock_supabase.get_zones_by_status.side_effect = (
            lambda statuses: [fz] if "FLIPPED" in statuses else []
        )
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1890.0)
        place = mocker.patch("bot.main.place_layered_orders")
        place.return_value = mocker.MagicMock(
            status="PLACED", setup_id=uuid4(),
            layer_1_ticket=11111, sl_price=1922.5, tp1_price=1890.0,
            error_messages=[],
        )

        # Iter 1: detection populates flipped queue; price below
        # zone → gate defers.
        b.run_iteration(NOW)
        place.assert_not_called()
        assert len(b._pending_flipped_zones) == 1

        # Iter 2: price rises to zone.bottom → fires.
        mock_mt5.get_current_price.return_value = {
            "bid": 1899.9, "ask": 1900.0, "time": NOW, "time_msc": 0,
        }
        b.run_iteration(NOW + timedelta(seconds=1))

        place.assert_called_once()
        assert b._pending_flipped_zones == []

    def test_dedup_still_blocks_placement_on_pending_zone(
        self, bot: tuple[Bot, dict[str, MagicMock]],
        mock_mt5: MagicMock, mock_supabase: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        # A pending zone whose bounds overlap an existing CONSUMED
        # zone in the same direction stays in the queue but never
        # fires — dedup blocks every attempt.
        b, _ = bot
        b.state.last_config_refresh = NOW
        b.state.last_m5_bar_time = None
        # Price already at zone (so the gate wouldn't otherwise defer).
        mock_mt5.get_current_price.return_value = {
            "bid": 1900.5, "ask": 1900.6, "time": NOW, "time_msc": 0,
        }

        zone = _make_validated_for_persistence(_PT.RBR)
        existing_consumed = _make_zone_row(
            direction="BUY", top=1900.5, bottom=1900.0,
            status="CONSUMED", consumed_at=NOW,
        )
        mock_supabase.get_zones_by_status.return_value = [existing_consumed]
        mocker.patch("bot.main.run_strategy_pipeline", return_value=[zone])
        mocker.patch("bot.main.find_nearest_local_peak", return_value=1907.5)
        place = mocker.patch("bot.main.place_layered_orders")

        b.run_iteration(NOW)

        place.assert_not_called()
        # Pending list keeps the zone — dedup is run inside
        # _try_place_setup which returned False; it'll keep retrying
        # on subsequent ticks (cheap; cache absorbs the Supabase hit).
        assert len(b._pending_pipeline_zones) == 1
