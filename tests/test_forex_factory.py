"""Forex Factory calendar news source (Nov 2026)."""

from datetime import datetime, timedelta, timezone

from bot.filters.forex_factory import ForexFactoryCalendar, parse_ff_events
from bot.filters.news_filter import NewsFilter, NewsFilterConfig

UTC = timezone.utc
NOW = datetime(2026, 11, 6, 13, 0, tzinfo=UTC)

FEED = [
    {"title": "Non-Farm Employment Change", "country": "USD",
     "date": "2026-11-06T08:30:00-05:00", "impact": "High",
     "forecast": "120K", "previous": "254K"},
    {"title": "Employment Change", "country": "CAD",
     "date": "2026-11-06T08:30:00-05:00", "impact": "High"},
    {"title": "Prelim UoM Consumer Sentiment", "country": "USD",
     "date": "2026-11-06T10:00:00-05:00", "impact": "Medium"},
    {"title": "Bank Holiday", "country": "USD",
     "date": "2026-11-11T00:00:00-05:00", "impact": "Holiday"},
    {"title": "broken", "country": "USD", "date": "not a date", "impact": "High"},
]

CFG = NewsFilterConfig(blackout_before_minutes=15, blackout_after_minutes=15,
                       impact_threshold="HIGH", currencies=("USD",))


def _cal(feed=FEED, clock=lambda: NOW):
    calls = []

    def fetch(url):
        calls.append(url)
        if isinstance(feed, Exception):
            raise feed
        return feed

    return ForexFactoryCalendar(fetch=fetch, clock=clock), calls


class TestParse:
    def test_times_converted_to_utc_and_holidays_skipped(self):
        ev = parse_ff_events(FEED, NOW)
        assert len(ev) == 3
        nfp = ev[0]
        assert nfp.event_time == datetime(2026, 11, 6, 13, 30, tzinfo=UTC)
        assert (nfp.currency, nfp.impact_level) == ("USD", "HIGH")


class TestBlackout:
    def test_usd_red_only(self):
        cal, _ = _cal()
        f = NewsFilter(cal, config=CFG, warn_when_empty=False)
        # 13:14 UTC: 16 min before NFP -> clear.
        assert not f.check(NOW + timedelta(minutes=14)).is_blocked

    def test_window_15_before_to_15_after(self):
        cal, _ = _cal()
        for mins, blocked in [(14, False), (15, True), (30, True), (45, True), (46, False)]:
            f = NewsFilter(cal, config=CFG, warn_when_empty=False)
            r = f.check(NOW + timedelta(minutes=mins))
            assert r.is_blocked is blocked, mins
            if blocked:
                assert "Non-Farm" in r.block_reason

    def test_medium_usd_and_red_cad_do_not_block(self):
        cad_only = [FEED[1], FEED[2]]
        cal, _ = _cal(cad_only)
        f = NewsFilter(cal, config=CFG, warn_when_empty=False)
        assert not f.check(NOW + timedelta(minutes=30)).is_blocked


class TestRefresh:
    def test_downloads_once_an_hour(self):
        t = [NOW]
        cal, calls = _cal(clock=lambda: t[0])
        cal.get_news_events_in_window(start=NOW, end=NOW + timedelta(hours=1))
        t[0] = NOW + timedelta(minutes=59)
        cal.get_news_events_in_window(start=NOW, end=NOW + timedelta(hours=1))
        assert len(calls) == 1
        t[0] = NOW + timedelta(minutes=61)
        cal.get_news_events_in_window(start=NOW, end=NOW + timedelta(hours=1))
        assert len(calls) == 2

    def test_failed_download_keeps_last_copy(self):
        t = [NOW]
        feed = list(FEED)
        state = {"fail": False}

        def fetch(url):
            if state["fail"]:
                raise ConnectionError("down")
            return feed

        cal = ForexFactoryCalendar(fetch=fetch, clock=lambda: t[0])
        cal.refresh()
        state["fail"] = True
        t[0] = NOW + timedelta(hours=2)
        cal.refresh()
        assert len(cal.events) == 3

    def test_never_downloaded_means_no_events_but_no_crash(self):
        cal, _ = _cal(ConnectionError("no internet"))
        assert cal.get_news_events_in_window(start=NOW, end=NOW + timedelta(hours=1)) == []

    def test_retry_waits_5_minutes_after_failure(self):
        t = [NOW]
        calls = []

        def fetch(url):
            calls.append(1)
            raise ConnectionError("down")

        cal = ForexFactoryCalendar(fetch=fetch, clock=lambda: t[0])
        cal.refresh()
        t[0] = NOW + timedelta(minutes=2)
        cal.refresh()
        assert len(calls) == 1
        t[0] = NOW + timedelta(minutes=6)
        cal.refresh()
        assert len(calls) == 2
