"""Tests for ``bot.filters.news_filter``.

Same pattern as test_sl_manager / test_tp1_manager:
``MagicMock(spec=...)`` for the Supabase dependency; module-level
helper builds a typed ``NewsEvent`` for fixtures.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pytest_mock import MockerFixture

from bot.filters.news_filter import (
    NewsCheckResult,
    NewsFilter,
    NewsFilterConfig,
    _meets_threshold,
    _pick_earliest,
    _pick_most_severe,
)
from bot.logging.supabase_logger import NewsEvent, SupabaseLogger


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


def make_event(
    *,
    id: UUID | None = None,
    event_time: datetime,
    currency: str = "USD",
    impact_level: str = "HIGH",
    title: str = "Non-Farm Payrolls",
    forecast: str | None = "200K",
    actual: str | None = None,
) -> NewsEvent:
    return NewsEvent(
        id=id or uuid4(),
        event_time=event_time,
        currency=currency,
        title=title,
        impact_level=impact_level,  # type: ignore[arg-type]
        forecast=forecast,
        actual=actual,
        fetched_at=NOW,
        created_at=NOW,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_supabase(mocker: MockerFixture) -> MagicMock:
    m = mocker.MagicMock(spec=SupabaseLogger)
    m.get_news_events_in_window.return_value = []
    return m


@pytest.fixture
def filter_(mock_supabase: MagicMock) -> NewsFilter:
    return NewsFilter(supabase=mock_supabase)


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


class TestMeetsThreshold:
    @pytest.mark.parametrize(
        "level,threshold,expected",
        [
            ("HIGH", "HIGH", True),
            ("HIGH", "MEDIUM", True),
            ("HIGH", "LOW", True),
            ("MEDIUM", "HIGH", False),
            ("MEDIUM", "MEDIUM", True),
            ("MEDIUM", "LOW", True),
            ("LOW", "HIGH", False),
            ("LOW", "MEDIUM", False),
            ("LOW", "LOW", True),
        ],
    )
    def test_severity_ordering(
        self, level: str, threshold: str, expected: bool,
    ) -> None:
        assert _meets_threshold(level, threshold) is expected


class TestPickHelpers:
    def test_pick_most_severe_prefers_high_over_medium(self) -> None:
        e1 = make_event(event_time=NOW, impact_level="MEDIUM", title="CPI")
        e2 = make_event(event_time=NOW, impact_level="HIGH", title="NFP")
        assert _pick_most_severe([e1, e2]) is e2

    def test_pick_most_severe_ties_break_to_earliest_event_time(self) -> None:
        e1 = make_event(
            event_time=NOW + timedelta(minutes=10),
            impact_level="HIGH", title="Late",
        )
        e2 = make_event(
            event_time=NOW + timedelta(minutes=5),
            impact_level="HIGH", title="Early",
        )
        assert _pick_most_severe([e1, e2]) is e2

    def test_pick_most_severe_empty_returns_none(self) -> None:
        assert _pick_most_severe([]) is None

    def test_pick_earliest_returns_min_event_time(self) -> None:
        e1 = make_event(event_time=NOW + timedelta(minutes=30))
        e2 = make_event(event_time=NOW + timedelta(minutes=5))
        e3 = make_event(event_time=NOW + timedelta(minutes=15))
        assert _pick_earliest([e1, e2, e3]) is e2

    def test_pick_earliest_empty_returns_none(self) -> None:
        assert _pick_earliest([]) is None


# --------------------------------------------------------------------------- #
# check() — blocking events
# --------------------------------------------------------------------------- #


class TestBlockingDetection:
    def test_high_impact_usd_within_window_blocks(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # Event 10 minutes from now → inside the 30-min before window.
        ev = make_event(event_time=NOW + timedelta(minutes=10))
        mock_supabase.get_news_events_in_window.return_value = [ev]

        result = filter_.check(NOW)

        assert result.is_blocked is True
        assert result.blocking_event is ev
        assert "Non-Farm Payrolls" in (result.block_reason or "")
        # resume_at = event_time + 15 (default after).
        assert result.resume_at == ev.event_time + timedelta(minutes=15)

    def test_high_impact_usd_60min_away_not_blocked_but_upcoming(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # 60 min away → outside 30-min blackout, inside 60-min upcoming.
        ev = make_event(event_time=NOW + timedelta(minutes=60))
        mock_supabase.get_news_events_in_window.return_value = [ev]

        result = filter_.check(NOW)

        assert result.is_blocked is False
        assert result.upcoming_event is ev

    def test_high_impact_usd_45min_away_not_blocked_but_upcoming(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # 45 min ahead — outside 30-min before window, but the test
        # description's "31 minutes before NFP" case: not blocked,
        # flagged upcoming.
        ev = make_event(event_time=NOW + timedelta(minutes=45))
        mock_supabase.get_news_events_in_window.return_value = [ev]

        result = filter_.check(NOW)
        assert result.is_blocked is False
        assert result.upcoming_event is ev

    def test_medium_impact_usd_within_window_does_not_block(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # Threshold default = HIGH. MEDIUM event → ignored.
        ev = make_event(
            event_time=NOW + timedelta(minutes=10),
            impact_level="MEDIUM", title="ISM Manufacturing",
        )
        mock_supabase.get_news_events_in_window.return_value = [ev]

        result = filter_.check(NOW)
        assert result.is_blocked is False

    def test_high_impact_eur_within_window_does_not_block(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        ev = make_event(
            event_time=NOW + timedelta(minutes=10),
            currency="EUR", title="ECB Rate Decision",
        )
        mock_supabase.get_news_events_in_window.return_value = [ev]

        result = filter_.check(NOW)
        # USD-only by default → EUR ignored.
        assert result.is_blocked is False

    def test_event_during_post_event_window_still_blocks(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # Event 10 min ago → still inside the 15-min after window.
        ev = make_event(event_time=NOW - timedelta(minutes=10))
        mock_supabase.get_news_events_in_window.return_value = [ev]

        result = filter_.check(NOW)
        assert result.is_blocked is True
        assert result.blocking_event is ev

    def test_event_post_window_expired_does_not_block(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # Event 20 min ago → past the 15-min after window.
        ev = make_event(event_time=NOW - timedelta(minutes=20))
        mock_supabase.get_news_events_in_window.return_value = [ev]

        result = filter_.check(NOW)
        assert result.is_blocked is False


class TestBoundaryInclusive:
    def test_exactly_30_min_before_event_is_blocked(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # now = T - 30min exactly → inside (boundary inclusive).
        ev = make_event(event_time=NOW + timedelta(minutes=30))
        mock_supabase.get_news_events_in_window.return_value = [ev]
        result = filter_.check(NOW)
        assert result.is_blocked is True

    def test_30min_1sec_before_event_not_blocked(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # now = T - 30:01 → just outside.
        ev = make_event(
            event_time=NOW + timedelta(minutes=30, seconds=1),
        )
        mock_supabase.get_news_events_in_window.return_value = [ev]
        result = filter_.check(NOW)
        assert result.is_blocked is False
        # But still upcoming (within 60 min).
        assert result.upcoming_event is ev

    def test_exactly_at_event_time_blocked(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        ev = make_event(event_time=NOW)
        mock_supabase.get_news_events_in_window.return_value = [ev]
        result = filter_.check(NOW)
        assert result.is_blocked is True

    def test_exactly_15_min_after_event_blocked(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # now = T + 15min exactly → inside post-window (inclusive).
        ev = make_event(event_time=NOW - timedelta(minutes=15))
        mock_supabase.get_news_events_in_window.return_value = [ev]
        result = filter_.check(NOW)
        assert result.is_blocked is True


class TestConflictResolution:
    def test_higher_impact_wins_when_multiple_events_overlap(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # Default threshold=HIGH, but a config with threshold=MEDIUM
        # is needed for both to be eligible.
        cfg = NewsFilterConfig(impact_threshold="MEDIUM")
        f = NewsFilter(supabase=mock_supabase, config=cfg)
        nfp = make_event(
            event_time=NOW + timedelta(minutes=10),
            impact_level="HIGH", title="NFP",
        )
        ism = make_event(
            event_time=NOW + timedelta(minutes=5),
            impact_level="MEDIUM", title="ISM",
        )
        mock_supabase.get_news_events_in_window.return_value = [ism, nfp]

        result = f.check(NOW)
        assert result.is_blocked is True
        assert result.blocking_event is nfp  # HIGH beats MEDIUM

    def test_same_impact_earlier_event_wins(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # Two HIGH USD events both inside the window.
        nfp = make_event(
            event_time=NOW + timedelta(minutes=10),
            impact_level="HIGH", title="NFP",
        )
        cpi = make_event(
            event_time=NOW + timedelta(minutes=20),
            impact_level="HIGH", title="CPI",
        )
        mock_supabase.get_news_events_in_window.return_value = [cpi, nfp]

        result = filter_.check(NOW)
        # Tiebreaker = earliest event_time.
        assert result.blocking_event is nfp


# --------------------------------------------------------------------------- #
# Empty / error paths
# --------------------------------------------------------------------------- #


class TestNoEvents:
    def test_empty_news_table_not_blocked(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        mock_supabase.get_news_events_in_window.return_value = []
        result = filter_.check(NOW)
        assert result.is_blocked is False
        assert result.blocking_event is None
        assert result.upcoming_event is None

    def test_supabase_error_returns_safe_default(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # Per spec: graceful degradation — better to skip a blackout
        # than crash the bot's main loop.
        mock_supabase.get_news_events_in_window.side_effect = RuntimeError(
            "DB down"
        )
        result = filter_.check(NOW)
        assert result.is_blocked is False
        assert result.block_reason == "news_query_failed"
        assert result.cache_used is False


# --------------------------------------------------------------------------- #
# Cache behaviour
# --------------------------------------------------------------------------- #


class TestCache:
    def test_second_call_within_ttl_uses_cache_no_query(
        self, mock_supabase: MagicMock,
    ) -> None:
        ev = make_event(event_time=NOW + timedelta(minutes=10))
        mock_supabase.get_news_events_in_window.return_value = [ev]
        f = NewsFilter(supabase=mock_supabase)

        first = f.check(NOW)
        # Move 1 minute forward (well inside the 5-min default TTL).
        second = f.check(NOW + timedelta(minutes=1))

        assert first.cache_used is False
        assert second.cache_used is True
        # Underlying query was hit only once.
        assert mock_supabase.get_news_events_in_window.call_count == 1
        # Same blocking decision (event is still inside the window).
        assert second.is_blocked is True

    def test_call_after_ttl_refreshes_cache(
        self, mock_supabase: MagicMock,
    ) -> None:
        cfg = NewsFilterConfig(cache_ttl_seconds=60.0)
        f = NewsFilter(supabase=mock_supabase, config=cfg)
        ev = make_event(event_time=NOW + timedelta(minutes=10))
        mock_supabase.get_news_events_in_window.return_value = [ev]

        f.check(NOW)
        # Move 2 minutes forward (TTL=1 min → cache expired).
        result = f.check(NOW + timedelta(minutes=2))

        assert result.cache_used is False
        assert mock_supabase.get_news_events_in_window.call_count == 2

    def test_supabase_error_does_not_poison_cache(
        self, mock_supabase: MagicMock,
    ) -> None:
        # First call fails; cache should stay empty so the next call
        # actually re-queries (instead of "using" a None cache).
        mock_supabase.get_news_events_in_window.side_effect = RuntimeError(
            "DB down"
        )
        f = NewsFilter(supabase=mock_supabase)
        first = f.check(NOW)
        assert first.is_blocked is False

        # Recover: second call within TTL — but cache wasn't populated,
        # so a fresh query is attempted again.
        mock_supabase.get_news_events_in_window.side_effect = None
        ev = make_event(event_time=NOW + timedelta(minutes=10))
        mock_supabase.get_news_events_in_window.return_value = [ev]
        second = f.check(NOW + timedelta(seconds=30))
        assert second.is_blocked is True
        assert second.cache_used is False


# --------------------------------------------------------------------------- #
# Naive timestamp handling
# --------------------------------------------------------------------------- #


class TestNaiveTimestamp:
    def test_naive_now_treated_as_utc(
        self, filter_: NewsFilter, mock_supabase: MagicMock,
    ) -> None:
        # If a caller passes a naive datetime, treat it as UTC rather
        # than crash on tz-aware comparison. Production main loop will
        # always pass UTC, but tolerate the alternative.
        ev = make_event(event_time=NOW + timedelta(minutes=10))
        mock_supabase.get_news_events_in_window.return_value = [ev]
        naive = datetime(2026, 5, 8, 12, 0)  # equivalent to NOW in UTC
        result = filter_.check(naive)
        assert result.is_blocked is True


# --------------------------------------------------------------------------- #
# Result + config defaults
# --------------------------------------------------------------------------- #


class TestDefaults:
    def test_result_defaults(self) -> None:
        r = NewsCheckResult(is_blocked=False)
        assert r.blocking_event is None
        assert r.block_reason is None
        assert r.resume_at is None
        assert r.upcoming_event is None
        assert r.cache_used is False

    def test_config_defaults_match_seed_values(self) -> None:
        c = NewsFilterConfig()
        # Match migration 001 seeds + spec Section 8.3.
        assert c.blackout_before_minutes == 30
        assert c.blackout_after_minutes == 15
        assert c.impact_threshold == "HIGH"
        assert c.currencies == ("USD",)
        assert c.cache_ttl_seconds == 300.0
