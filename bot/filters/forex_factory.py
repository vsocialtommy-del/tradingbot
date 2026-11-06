"""Forex Factory calendar as the news source (Nov 2026).

The bot downloads Forex Factory's free weekly calendar feed itself (no
API key, no Supabase table, no Vercel cron) and hands the events to
:class:`bot.filters.news_filter.NewsFilter`. Red folder = ``"High"``.

Feed format (one entry per event)::

    {"title": "Non-Farm Employment Change", "country": "USD",
     "date": "2026-11-06T08:30:00-05:00", "impact": "High",
     "forecast": "...", "previous": "..."}

* Refreshed every ``refresh_minutes`` (60). Between refreshes the last
  good copy is used, so a failed download never wipes known events.
* If it can't be downloaded at all, a warning is logged (at most every
  ``warn_every_minutes``) so the bot never goes quiet about trading
  without news protection.
* The feed is unofficial. If Forex Factory changes it, the warning is
  what tells you.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable
from uuid import NAMESPACE_URL, uuid5

from loguru import logger

from bot.logging.supabase_logger import NewsEvent

FF_THIS_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

_IMPACT = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _default_fetch(url: str) -> Any:
    import requests

    r = requests.get(url, timeout=15, headers={"User-Agent": "xautism-bot/1.0"})
    r.raise_for_status()
    return r.json()


def parse_ff_events(raw: Iterable[dict], fetched_at: datetime) -> list[NewsEvent]:
    """Feed entries -> NewsEvent. Skips holidays / unparseable rows."""
    out: list[NewsEvent] = []
    for row in raw or []:
        try:
            impact = _IMPACT.get(str(row.get("impact", "")).strip().lower())
            if impact is None:
                continue
            when = datetime.fromisoformat(str(row["date"]))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            when = when.astimezone(timezone.utc)
            title = str(row.get("title", "")).strip()
            currency = str(row.get("country", "")).strip().upper()
            out.append(NewsEvent(
                id=uuid5(NAMESPACE_URL, f"ff|{currency}|{title}|{when.isoformat()}"),
                event_time=when,
                currency=currency,
                title=title,
                impact_level=impact,  # type: ignore[arg-type]
                forecast=(str(row["forecast"]) if row.get("forecast") else None),
                actual=(str(row["actual"]) if row.get("actual") else None),
                fetched_at=fetched_at,
                created_at=fetched_at,
            ))
        except Exception:
            continue
    return out


class ForexFactoryCalendar:
    """Drop-in news source for :class:`NewsFilter` (same method as
    ``SupabaseLogger.get_news_events_in_window``)."""

    def __init__(
        self,
        *,
        url: str = FF_THIS_WEEK_URL,
        refresh_minutes: float = 60.0,
        retry_minutes: float = 5.0,
        warn_every_minutes: float = 30.0,
        fetch: Callable[[str], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._url = url
        self._refresh = timedelta(minutes=refresh_minutes)
        self._retry = timedelta(minutes=retry_minutes)
        self._warn_every = timedelta(minutes=warn_every_minutes)
        self._fetch = fetch or _default_fetch
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._events: list[NewsEvent] = []
        self._loaded_at: datetime | None = None
        self._last_try: datetime | None = None
        self._last_warn: datetime | None = None

    @property
    def events(self) -> list[NewsEvent]:
        return list(self._events)

    def refresh(self, force: bool = False) -> None:
        now = self._clock()
        due = self._loaded_at is None or now - self._loaded_at >= self._refresh
        if not force and not due:
            return
        if not force and self._last_try is not None and now - self._last_try < self._retry:
            return  # failed recently: don't hammer the feed
        self._last_try = now
        try:
            events = parse_ff_events(self._fetch(self._url), fetched_at=now)
        except Exception as e:
            if self._last_warn is None or now - self._last_warn >= self._warn_every:
                logger.warning(
                    f"news: couldn't download the Forex Factory calendar ({e}); "
                    + ("using the last copy" if self._events else
                       "NO NEWS PROTECTION until it downloads")
                )
                self._last_warn = now
            return
        self._events = events
        self._loaded_at = now
        usd_red = [e for e in events if e.currency == "USD" and e.impact_level == "HIGH"]
        logger.info(
            f"news: Forex Factory calendar loaded, {len(events)} events, "
            f"{len(usd_red)} USD red folder"
        )

    def get_news_events_in_window(
        self,
        *,
        start: datetime,
        end: datetime,
        currencies: list[str] | None = None,
        min_impact: str = "HIGH",
    ) -> list[NewsEvent]:
        self.refresh()
        floor = _RANK.get(min_impact, 2)
        cur = {c.upper() for c in currencies} if currencies else None
        return sorted(
            (
                e for e in self._events
                if start <= e.event_time <= end
                and _RANK.get(e.impact_level, -1) >= floor
                and (cur is None or e.currency in cur)
            ),
            key=lambda e: e.event_time,
        )
