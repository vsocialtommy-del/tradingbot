"""OHLC bar fetcher with TTL caching.

Wraps ``MT5Connector.get_ohlc`` with a small in-memory cache so the
strategy modules can ask "give me the last 200 1H bars" repeatedly
without a network round-trip every time.

Cache strategy
--------------
* Keyed by ``(symbol, timeframe, count)``.
* Time-to-live in seconds; default 30 s. Short enough that the most
  recent (still-forming) bar is refreshed often; long enough that a
  single strategy iteration never re-fetches the same window.
* ``invalidate()`` clears the cache — call it from the bar-close hook
  in Phase B if you want strategy code to see settled bars instantly.

DataFrames are returned ``copy()``-ed so a caller mutating the frame
can't poison the cache.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from bot.execution.mt5_connector import MT5Connector, Timeframe


class OHLCProvider:
    """Cached OHLC fetcher built on top of :class:`MT5Connector`."""

    def __init__(self, connector: MT5Connector, cache_ttl_sec: float = 30.0) -> None:
        self._connector = connector
        self._cache_ttl = cache_ttl_sec
        self._cache: dict[tuple[str, str, int], tuple[float, pd.DataFrame]] = {}

    def get(
        self,
        symbol: str,
        timeframe: Timeframe,
        count: int,
    ) -> pd.DataFrame:
        """Return the last ``count`` bars; cached if fresh, else fetched."""
        key = (symbol, timeframe, count)
        now = time.monotonic()

        cached = self._cache.get(key)
        if cached is not None and now - cached[0] < self._cache_ttl:
            return cached[1].copy()

        df = self._connector.get_ohlc(symbol, timeframe, count)
        self._cache[key] = (now, df)
        return df.copy()

    def invalidate(self, symbol: str | None = None) -> None:
        """Drop cache entries (all, or only those matching ``symbol``)."""
        if symbol is None:
            self._cache.clear()
            return
        self._cache = {k: v for k, v in self._cache.items() if k[0] != symbol}

    def stats(self) -> dict[str, Any]:
        """Diagnostic: cache size + age of each entry."""
        now = time.monotonic()
        return {
            "ttl_sec": self._cache_ttl,
            "entries": [
                {
                    "symbol": s,
                    "timeframe": tf,
                    "count": c,
                    "age_sec": round(now - cached_at, 2),
                }
                for (s, tf, c), (cached_at, _) in self._cache.items()
            ],
        }
