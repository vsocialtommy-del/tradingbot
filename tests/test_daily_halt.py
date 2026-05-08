"""Tests for ``bot.risk.daily_halt``.

Tests use timezone-aware datetimes throughout. ``ET`` is a shorthand
for ``America/New_York``. UTC is also tested to confirm cross-zone
behaviour (the orchestrator might pass times in any zone).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from bot.risk.daily_halt import (
    DailyHaltConfig,
    DailyHaltResult,
    check_daily_halt,
    current_trading_date,
    is_halt_expired,
    next_daily_reset,
)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Drawdown math
# --------------------------------------------------------------------------- #


class TestDrawdownMath:
    def test_at_threshold_exactly_halted(self) -> None:
        # $1000 → $900 = -10% exactly, threshold -10% → boundary inclusive.
        r = check_daily_halt(starting_balance=1000, current_balance=900)
        assert r.current_drawdown_pct == pytest.approx(-10.0)
        assert r.is_halted is True
        assert r.reason == "DAILY_LOSS_LIMIT_REACHED"

    def test_just_below_threshold_not_halted(self) -> None:
        # $1000 → $900.01 = -9.999% — under threshold, not halted.
        r = check_daily_halt(starting_balance=1000, current_balance=900.01)
        assert r.current_drawdown_pct > -10.0
        assert r.is_halted is False
        assert r.reason is None

    def test_just_above_threshold_halted(self) -> None:
        # $1000 → $899.99 = -10.001% — over threshold, halted.
        r = check_daily_halt(starting_balance=1000, current_balance=899.99)
        assert r.current_drawdown_pct < -10.0
        assert r.is_halted is True
        assert r.reason == "DAILY_LOSS_LIMIT_REACHED"

    def test_winning_day_not_halted(self) -> None:
        r = check_daily_halt(starting_balance=1000, current_balance=1100)
        assert r.current_drawdown_pct == pytest.approx(10.0)
        assert r.is_halted is False
        assert r.resume_at is None

    def test_severe_loss_halted(self) -> None:
        # 50% drawdown — way past the 10% halt.
        r = check_daily_halt(starting_balance=1000, current_balance=500)
        assert r.current_drawdown_pct == pytest.approx(-50.0)
        assert r.is_halted is True

    def test_total_loss_halted(self) -> None:
        r = check_daily_halt(starting_balance=1000, current_balance=0)
        assert r.current_drawdown_pct == pytest.approx(-100.0)
        assert r.is_halted is True

    def test_negative_balance_handled(self) -> None:
        # Over-leveraged blow-up — drawdown < -100%, still works.
        r = check_daily_halt(starting_balance=1000, current_balance=-50)
        assert r.current_drawdown_pct < -100
        assert r.is_halted is True

    def test_unchanged_balance_not_halted(self) -> None:
        r = check_daily_halt(starting_balance=1000, current_balance=1000)
        assert r.current_drawdown_pct == 0.0
        assert r.is_halted is False


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_zero_starting_balance_rejected(self) -> None:
        with pytest.raises(ValueError, match="starting_balance"):
            check_daily_halt(starting_balance=0, current_balance=900)

    def test_negative_starting_balance_rejected(self) -> None:
        with pytest.raises(ValueError, match="starting_balance"):
            check_daily_halt(starting_balance=-100, current_balance=900)


# --------------------------------------------------------------------------- #
# Custom config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_tighter_threshold_halts_smaller_drawdown(self) -> None:
        # -5% drawdown halted only when threshold is 5% (not default 10%).
        balances = dict(starting_balance=1000, current_balance=950)
        default = check_daily_halt(**balances)
        tight = check_daily_halt(
            **balances, config=DailyHaltConfig(daily_loss_limit_pct=5.0)
        )
        assert default.is_halted is False
        assert tight.is_halted is True

    def test_threshold_passed_through_in_result(self) -> None:
        r = check_daily_halt(
            starting_balance=1000,
            current_balance=900,
            config=DailyHaltConfig(daily_loss_limit_pct=8.0),
        )
        assert r.threshold_pct == 8.0


# --------------------------------------------------------------------------- #
# Resume_at — populated when halted, None otherwise
# --------------------------------------------------------------------------- #


class TestResumeAt:
    def test_resume_at_none_when_not_halted(self) -> None:
        r = check_daily_halt(starting_balance=1000, current_balance=950)
        assert r.resume_at is None

    def test_resume_at_today_when_called_before_reset(self) -> None:
        now = datetime(2026, 5, 6, 12, 0, tzinfo=ET)  # noon ET
        r = check_daily_halt(
            starting_balance=1000, current_balance=900, now=now
        )
        # Still before today's 17:00 reset → next reset is today.
        assert r.resume_at == datetime(2026, 5, 6, 17, 0, tzinfo=ET)

    def test_resume_at_tomorrow_when_called_after_reset(self) -> None:
        now = datetime(2026, 5, 6, 18, 0, tzinfo=ET)  # 6 PM ET
        r = check_daily_halt(
            starting_balance=1000, current_balance=900, now=now
        )
        assert r.resume_at == datetime(2026, 5, 7, 17, 0, tzinfo=ET)

    def test_resume_at_works_with_utc_input(self) -> None:
        # Late winter: 22:00 UTC = 17:00 EST.
        # 21:00 UTC on Jan 7 → 16:00 ET on Jan 7 → before today's reset
        # → next reset is Jan 7 17:00 ET == Jan 7 22:00 UTC.
        now_utc = datetime(2026, 1, 7, 21, 0, tzinfo=UTC)
        r = check_daily_halt(
            starting_balance=1000, current_balance=900, now=now_utc
        )
        assert r.resume_at is not None
        # Verify against ET equivalent.
        expected = datetime(2026, 1, 7, 17, 0, tzinfo=ET)
        assert r.resume_at == expected


# --------------------------------------------------------------------------- #
# Trading-day boundary
# --------------------------------------------------------------------------- #


class TestTradingDateBoundary:
    def test_just_before_17et_is_yesterday(self) -> None:
        now = datetime(2026, 5, 6, 16, 59, tzinfo=ET)
        assert current_trading_date(now) == date(2026, 5, 5)

    def test_exactly_17et_is_today(self) -> None:
        # The reset has just happened — we're now in today's trading day.
        now = datetime(2026, 5, 6, 17, 0, tzinfo=ET)
        assert current_trading_date(now) == date(2026, 5, 6)

    def test_after_17et_is_today(self) -> None:
        now = datetime(2026, 5, 6, 23, 0, tzinfo=ET)
        assert current_trading_date(now) == date(2026, 5, 6)

    def test_next_morning_still_in_prior_trading_day(self) -> None:
        # 9 AM ET on May 7 — before May 7's 17:00 reset, still in May 6's
        # trading day.
        now = datetime(2026, 5, 7, 9, 0, tzinfo=ET)
        assert current_trading_date(now) == date(2026, 5, 6)

    def test_works_with_utc_input(self) -> None:
        # 21:59 UTC in winter (EST) = 16:59 ET → still in yesterday's day.
        now = datetime(2026, 1, 7, 21, 59, tzinfo=UTC)
        assert current_trading_date(now) == date(2026, 1, 6)

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            current_trading_date(datetime(2026, 5, 6, 12, 0))


# --------------------------------------------------------------------------- #
# next_daily_reset
# --------------------------------------------------------------------------- #


class TestNextDailyReset:
    def test_before_today_reset_returns_today(self) -> None:
        now = datetime(2026, 5, 6, 12, 0, tzinfo=ET)
        assert next_daily_reset(now) == datetime(2026, 5, 6, 17, 0, tzinfo=ET)

    def test_after_today_reset_returns_tomorrow(self) -> None:
        now = datetime(2026, 5, 6, 18, 0, tzinfo=ET)
        assert next_daily_reset(now) == datetime(2026, 5, 7, 17, 0, tzinfo=ET)

    def test_at_exactly_reset_returns_tomorrow(self) -> None:
        # We just crossed it — next reset is tomorrow's.
        now = datetime(2026, 5, 6, 17, 0, tzinfo=ET)
        assert next_daily_reset(now) == datetime(2026, 5, 7, 17, 0, tzinfo=ET)

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            next_daily_reset(datetime(2026, 5, 6, 12, 0))

    def test_dst_spring_forward(self) -> None:
        # March 8, 2026: DST begins in the US. Clocks jump from 2 AM EST
        # to 3 AM EDT. The 17:00 boundary shouldn't be affected — it's
        # always "5 PM local clock". On March 7 evening (still EST),
        # next reset is March 8 17:00 EDT.
        now = datetime(2026, 3, 7, 18, 0, tzinfo=ET)
        result = next_daily_reset(now)
        assert result.date() == date(2026, 3, 8)
        assert result.hour == 17
        # March 8 17:00 EDT == 21:00 UTC. Verify by converting.
        assert result.astimezone(UTC) == datetime(2026, 3, 8, 21, 0, tzinfo=UTC)

    def test_dst_fall_back(self) -> None:
        # November 1, 2026: DST ends. Clocks fall back from 2 AM EDT to
        # 1 AM EST. Same principle — 17:00 ET is whatever the local
        # clock reads. On Oct 31 evening (still EDT), next reset is
        # Nov 1 17:00 EST.
        now = datetime(2026, 10, 31, 18, 0, tzinfo=ET)
        result = next_daily_reset(now)
        assert result.date() == date(2026, 11, 1)
        assert result.hour == 17
        # Nov 1 17:00 EST == 22:00 UTC.
        assert result.astimezone(UTC) == datetime(2026, 11, 1, 22, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# is_halt_expired
# --------------------------------------------------------------------------- #


class TestHaltExpired:
    def test_same_trading_day_not_expired(self) -> None:
        # Both within May 6's trading day (after 17:00 ET on May 6).
        halted = datetime(2026, 5, 6, 18, 0, tzinfo=ET)
        now = datetime(2026, 5, 6, 23, 0, tzinfo=ET)
        assert is_halt_expired(halted, now) is False

    def test_across_calendar_dates_but_same_trading_day(self) -> None:
        # Halted at 18:00 ET on May 6 (in May 6's trading day).
        # Now at 14:00 ET on May 7 (still in May 6's trading day, before
        # May 7's reset).
        halted = datetime(2026, 5, 6, 18, 0, tzinfo=ET)
        now = datetime(2026, 5, 7, 14, 0, tzinfo=ET)
        assert is_halt_expired(halted, now) is False

    def test_after_next_daily_reset_expired(self) -> None:
        # Halted at noon on May 6 (in May 5's trading day per the boundary
        # rule). Now at 18:00 ET on May 6 (in May 6's trading day).
        halted = datetime(2026, 5, 6, 12, 0, tzinfo=ET)
        now = datetime(2026, 5, 6, 18, 0, tzinfo=ET)
        assert is_halt_expired(halted, now) is True

    def test_long_duration_obviously_expired(self) -> None:
        halted = datetime(2026, 1, 1, 18, 0, tzinfo=ET)
        now = datetime(2026, 5, 6, 18, 0, tzinfo=ET)
        assert is_halt_expired(halted, now) is True


# --------------------------------------------------------------------------- #
# Result metadata pass-through
# --------------------------------------------------------------------------- #


class TestResultMetadata:
    def test_balances_passed_through(self) -> None:
        r = check_daily_halt(starting_balance=1234.56, current_balance=1100.00)
        assert r.starting_balance == 1234.56
        assert r.current_balance == 1100.00

    def test_drawdown_calculation_precision(self) -> None:
        # Common Gold scenario: $5000 starting, $4500 current = -10%.
        r = check_daily_halt(starting_balance=5000, current_balance=4500)
        assert r.current_drawdown_pct == pytest.approx(-10.0)
        assert r.is_halted is True

    def test_returns_dailyhaltresult_dataclass(self) -> None:
        r = check_daily_halt(starting_balance=1000, current_balance=950)
        assert isinstance(r, DailyHaltResult)
