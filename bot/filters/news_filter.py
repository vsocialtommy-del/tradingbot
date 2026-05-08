"""High-impact USD news blackout (spec Section 8).

Reads the Supabase ``news_events`` table — populated by a Vercel cron
from Finnhub (Section 8.2), never by the bot — and answers two
questions every tick:

1. Is the bot currently inside the blackout window of a high-impact
   tracked event? (``NewsCheckResult.is_blocked``)
2. Is there an upcoming event the operator should know about, even
   if it hasn't entered the blackout window yet?
   (``NewsCheckResult.upcoming_event``)

Behaviour during blackout (spec 8.3):

* ``T - blackout_before`` ≤ now ≤ ``T + blackout_after`` ⇒ ``is_blocked=True``
* The blackout *before* extends 30 min by default; *after* extends 15
  min. Both are tunable in ``bot_config``.

Caller (main loop / order_manager) is responsible for translating
``is_blocked=True`` into action:

* Skip new setups
* Close open Gold positions (per spec — but this module doesn't act
  on it; that's an order-manager concern)
* Display the blocking event on the dashboard

Design decisions called out in the PR
-------------------------------------

1. **Read-side caching, 5-minute TTL.** Every tick would otherwise hit
   Supabase. The Vercel cron refreshes every ~15 min, so a 5-min TTL
   keeps freshness within ¹⁄₃ of an upstream cycle while cutting reads
   by ~99 %. Cache stores the *forward window* (next ``cache_window``
   minutes from fetch time); blackout / upcoming filters slice the
   cached list at query time. ``cache_used`` flag in the result lets
   the caller distinguish a fresh-fetch tick from a cache-hit tick.

2. **Schema-faithful ``NewsEvent``.** ``forecast`` and ``actual`` are
   ``str`` (Finnhub embeds units / qualitative values like "Hawkish");
   no ``description`` field exists in the schema (spec wording was
   slightly inaccurate). The trading decision only needs ``event_time``
   + ``impact_level`` + ``currency`` regardless.

3. **Existing seed key names retained.** ``news_blackout_minutes_before``
   and ``news_blackout_minutes_after`` are already in migration 001 and
   match spec Section 8.3 wording. We keep those names rather than
   introducing parallel ``news_block_*`` keys.

4. **Graceful degradation on Supabase error.** A query failure logs a
   warning and returns ``is_blocked=False``. Rationale: a hard fail
   would crash the bot's main loop, which is worse than missing one
   blackout window. Bot writes (trades, logs) would also be failing
   in that scenario, so the operator will see the alarm fast through
   other channels.

5. **Conflict resolution: highest impact, then closest event_time.**
   When multiple events overlap the blackout window, we pick the most
   severe one (HIGH > MEDIUM > LOW), tiebreaker by earliest
   ``event_time``. This is what surfaces on the dashboard and in
   logs; the bot's gating decision is just "any qualifying event ⇒
   blocked", so the choice only matters for display.

6. **Sessions are orthogonal.** Friday-close / Sunday-open behaviour
   is ``daily_halt``'s domain (broker rollover, not news). News
   filter just answers the news question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from loguru import logger

from bot.logging.supabase_logger import (
    ImpactLevel,
    NewsEvent,
    SupabaseLogger,
)


# --------------------------------------------------------------------------- #
# Result + config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NewsCheckResult:
    """Outcome of a single :meth:`NewsFilter.check` call."""

    is_blocked: bool
    blocking_event: NewsEvent | None = None
    block_reason: str | None = None
    resume_at: datetime | None = None
    upcoming_event: NewsEvent | None = None
    cache_used: bool = False


@dataclass(frozen=True)
class NewsFilterConfig:
    blackout_before_minutes: int = 30
    """Minutes before a qualifying event when entries are blocked."""
    blackout_after_minutes: int = 15
    """Minutes after a qualifying event when entries resume."""
    upcoming_warning_minutes: int = 60
    """How far ahead the ``upcoming_event`` field looks. ≥
    ``blackout_before_minutes`` so an event doesn't pop into "blocking"
    without ever appearing as "upcoming"."""
    impact_threshold: ImpactLevel = "HIGH"
    """Minimum severity that triggers a blackout. Lower-severity events
    are returned by the Supabase query iff this is lowered, but only
    threshold-or-higher events block."""
    currencies: tuple[str, ...] = ("USD",)
    """Tracked currencies. v1: USD only — Gold is USD-denominated and
    other-currency events have negligible direct impact."""
    cache_ttl_seconds: float = 300.0
    """How long an in-memory event-list snapshot is reused. 5 min default
    matches the Vercel cron's ~15-min cadence (catches each refresh on
    the next-but-one tick at worst)."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class NewsFilter:
    """Read-only news gate. Single ``check(now)`` method."""

    def __init__(
        self,
        supabase: SupabaseLogger,
        config: NewsFilterConfig | None = None,
    ) -> None:
        self._supabase = supabase
        self._config = config or NewsFilterConfig()
        self._cache: list[NewsEvent] | None = None
        self._cache_fetched_at: datetime | None = None

    def check(self, now: datetime) -> NewsCheckResult:
        """Evaluate whether ``now`` is inside any qualifying event's blackout."""
        if now.tzinfo is None:
            # Treat naive timestamps as UTC; the Supabase column is
            # timestamptz, so all stored values are tz-aware.
            now = now.replace(tzinfo=timezone.utc)

        cfg = self._config
        events, cache_used = self._get_events(now)
        if events is None:
            # Hard failure on the underlying query — return safe default.
            return NewsCheckResult(
                is_blocked=False,
                block_reason="news_query_failed",
                cache_used=False,
            )

        before = timedelta(minutes=cfg.blackout_before_minutes)
        after = timedelta(minutes=cfg.blackout_after_minutes)
        upcoming = timedelta(minutes=cfg.upcoming_warning_minutes)

        blocking_candidates: list[NewsEvent] = []
        upcoming_candidates: list[NewsEvent] = []

        for ev in events:
            if not _meets_threshold(ev.impact_level, cfg.impact_threshold):
                continue
            if ev.currency not in cfg.currencies:
                continue
            blackout_start = ev.event_time - before
            blackout_end = ev.event_time + after
            if blackout_start <= now <= blackout_end:
                blocking_candidates.append(ev)
            elif now < ev.event_time <= now + upcoming:
                upcoming_candidates.append(ev)

        blocking = _pick_most_severe(blocking_candidates)
        upcoming_pick = _pick_earliest(upcoming_candidates)

        if blocking is None:
            return NewsCheckResult(
                is_blocked=False,
                upcoming_event=upcoming_pick,
                cache_used=cache_used,
            )

        return NewsCheckResult(
            is_blocked=True,
            blocking_event=blocking,
            block_reason=(
                f"{blocking.impact_level} {blocking.currency} event "
                f"'{blocking.title}' at {blocking.event_time.isoformat()}"
            ),
            resume_at=blocking.event_time + after,
            upcoming_event=upcoming_pick,
            cache_used=cache_used,
        )

    # ----------------------------------------------------------------- #
    # Cache + fetch
    # ----------------------------------------------------------------- #

    def _get_events(
        self, now: datetime,
    ) -> tuple[list[NewsEvent] | None, bool]:
        """Return (events, cache_used). events=None ⇒ Supabase fault.

        Cache stores a forward-only window: from ``now - blackout_after``
        (so an event still inside the post-window is captured) through
        ``now + max(upcoming_warning_minutes, blackout_before_minutes)``.
        """
        cfg = self._config
        if (
            self._cache is not None
            and self._cache_fetched_at is not None
            and (now - self._cache_fetched_at).total_seconds()
            < cfg.cache_ttl_seconds
        ):
            return self._cache, True

        # Pull a forward+backward window that covers any conceivable
        # query origin within the cache TTL.
        window_back = timedelta(minutes=cfg.blackout_after_minutes)
        window_forward = timedelta(
            minutes=max(cfg.upcoming_warning_minutes,
                        cfg.blackout_before_minutes),
        ) + timedelta(seconds=cfg.cache_ttl_seconds)

        try:
            events = self._supabase.get_news_events_in_window(
                start=now - window_back,
                end=now + window_forward,
                currencies=list(cfg.currencies),
                min_impact=cfg.impact_threshold,
            )
        except Exception:
            logger.exception(
                "news_filter: get_news_events_in_window failed; "
                "returning is_blocked=False (graceful degradation)"
            )
            return None, False

        if not events:
            # Empty result is a valid (but suspicious) state — log once
            # per cache refresh, not every tick.
            logger.warning(
                "news_filter: no events in Supabase for the next "
                f"{int(window_forward.total_seconds() // 60)} min — "
                "is the Vercel cron healthy?"
            )

        self._cache = events
        self._cache_fetched_at = now
        return events, False


# --------------------------------------------------------------------------- #
# Module-level helpers (testable without a NewsFilter instance)
# --------------------------------------------------------------------------- #


_SEVERITY_RANK: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _meets_threshold(level: str, threshold: str) -> bool:
    """``level >= threshold`` per the LOW/MEDIUM/HIGH ordering."""
    return _SEVERITY_RANK.get(level, -1) >= _SEVERITY_RANK.get(threshold, 99)


def _pick_most_severe(events: list[NewsEvent]) -> NewsEvent | None:
    """Highest ``impact_level``; tiebreak earliest ``event_time``."""
    if not events:
        return None
    return max(
        events,
        key=lambda e: (
            _SEVERITY_RANK.get(e.impact_level, -1),
            -e.event_time.timestamp(),
        ),
    )


def _pick_earliest(events: list[NewsEvent]) -> NewsEvent | None:
    """Closest ``event_time`` (smallest)."""
    if not events:
        return None
    return min(events, key=lambda e: e.event_time)
