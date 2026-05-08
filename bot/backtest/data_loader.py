"""Historical OHLC loader for the backtest engine.

Loads CSVs of M5 OHLC data, validates them, and returns a tz-aware
DataFrame indexed by timestamp. Designed for **Dukascopy** exports
(which are the simplest free source of XAUUSD M5 data) but accepts
any CSV with the standard ``open / high / low / close`` columns.

Dukascopy quirks handled
------------------------

* Their default header is ``Gmt time`` (with a space). We accept that
  alongside more conventional names like ``timestamp``, ``time``,
  ``date``.
* Timestamps come in UTC but are sometimes naive (no tz suffix). We
  always force-tag them ``UTC`` after parsing.
* Their format is ``dd.mm.yyyy hh:mm:ss.fff``. ``pd.to_datetime``'s
  default parser handles it via ``dayfirst=True``.
* Weekends and broker holidays produce **gaps** (no rows for Sat/Sun)
  rather than zero-volume bars. We preserve the gaps as-is — the
  engine's bar-close detection works on actual rows, not a regular
  grid.
* Volume is sometimes in lots, sometimes in tick-count. We don't
  trust it for anything; the strategy doesn't read it.

What we explicitly do NOT do
----------------------------

* No resampling. The strategy is M5-only; if you upload H1 data the
  loader will accept it and the backtest will run nonsense. Document
  the timeframe of the data you're loading.
* No forward-fill across gaps. Weekend gaps stay as-is — see above.
* No data quality scoring. If your CSV has spikes/bad ticks, the
  backtest reflects that. ``validate_ohlc`` only flags structural
  problems (high < low, NaNs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger


# Canonical columns the engine expects after loading.
REQUIRED_OHLC = ("open", "high", "low", "close")
OPTIONAL_VOLUME = "volume"


# Header name aliases → canonical lowercase column names.
# The keys are matched case-insensitively after stripping whitespace.
_HEADER_ALIASES: dict[str, str] = {
    "gmt time": "timestamp",
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "timestamp": "timestamp",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "vol": "volume",
}


def load_dukascopy_csv(
    path: str | Path,
    *,
    dayfirst: bool = True,
) -> pd.DataFrame:
    """Read an OHLC CSV → tz-aware DataFrame indexed by timestamp.

    Parameters
    ----------
    path
        Path to the CSV file. Reads with pandas defaults.
    dayfirst
        Whether to interpret ``dd.mm.yyyy`` (Dukascopy default, True)
        or ``mm.dd.yyyy`` (US-style, False) when ambiguous.

    Returns
    -------
    DataFrame indexed by UTC ``DatetimeIndex``, with columns
    ``open / high / low / close`` and (when present) ``volume``.
    Rows are sorted by timestamp; duplicate timestamps drop the later
    row (the first-seen wins).

    Raises
    ------
    ValueError
        Missing required columns, OHLC consistency violations
        (``high < low`` etc.), or NaNs in price columns.
    FileNotFoundError
        Path doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OHLC CSV not found: {p}")

    raw = pd.read_csv(p)
    df = _normalise_columns(raw)
    df = _parse_timestamps(df, dayfirst=dayfirst)
    df = _drop_duplicate_index(df)
    validate_ohlc(df)
    logger.info(
        f"data_loader: loaded {len(df)} bars from {p.name} "
        f"({df.index[0]} → {df.index[-1]})"
    )
    return df


def validate_ohlc(df: pd.DataFrame) -> None:
    """Structural sanity checks. Raises ``ValueError`` on the first failure."""
    missing = [c for c in REQUIRED_OHLC if c not in df.columns]
    if missing:
        raise ValueError(
            f"OHLC CSV missing required columns: {missing}. "
            f"Got: {list(df.columns)}"
        )
    if df.index.tz is None:
        raise ValueError("DataFrame index must be tz-aware (UTC)")
    if df[list(REQUIRED_OHLC)].isna().any().any():
        # Get the first NaN row for the error message.
        first_bad = df[df[list(REQUIRED_OHLC)].isna().any(axis=1)].index[0]
        raise ValueError(
            f"OHLC CSV contains NaN price at {first_bad}"
        )
    high_lt_low = df["high"] < df["low"]
    if high_lt_low.any():
        bad = df[high_lt_low].index[0]
        raise ValueError(
            f"OHLC consistency violation at {bad}: high < low"
        )
    # close / open must lie within [low, high].
    out_of_range = (
        (df["close"] > df["high"])
        | (df["close"] < df["low"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
    )
    if out_of_range.any():
        bad = df[out_of_range].index[0]
        raise ValueError(
            f"OHLC consistency violation at {bad}: open/close outside "
            f"[low, high]"
        )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + alias-resolve headers; drop unknown columns."""
    rename: dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower()
        canonical = _HEADER_ALIASES.get(key)
        if canonical is not None:
            rename[col] = canonical
    df = df.rename(columns=rename)
    keep = [c for c in df.columns if c in {*REQUIRED_OHLC, "timestamp", OPTIONAL_VOLUME}]
    return df[keep]


def _parse_timestamps(
    df: pd.DataFrame, *, dayfirst: bool,
) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError(
            "OHLC CSV missing timestamp column "
            "(expected one of: gmt time, time, date, datetime, timestamp)"
        )
    # ``dayfirst`` only applies to ambiguous formats (Dukascopy's
    # dd.mm.yyyy). For unambiguous ISO 8601 (yyyy-mm-dd / yyyy-mm-ddT...)
    # pandas would otherwise still swap month/day when dayfirst=True is
    # passed — so we sniff the first value and disable dayfirst for ISO.
    sample = str(df["timestamp"].dropna().iloc[0]).strip()
    looks_iso = (
        len(sample) >= 4
        and sample[:4].isdigit()
        and (len(sample) == 4 or sample[4] in "-/")
    )
    parse_kwargs: dict = {"errors": "coerce"}
    if not looks_iso:
        parse_kwargs["dayfirst"] = dayfirst
    ts = pd.to_datetime(df["timestamp"], **parse_kwargs)
    if ts.isna().any():
        bad = df.loc[ts.isna(), "timestamp"].iloc[0]
        raise ValueError(f"Unparseable timestamp: {bad!r}")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df = df.drop(columns=["timestamp"])
    df.index = pd.DatetimeIndex(ts.values, tz="UTC", name="time")
    return df.sort_index()


def _drop_duplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    """First-seen wins on duplicate timestamps."""
    dup_count = int(df.index.duplicated(keep="first").sum())
    if dup_count > 0:
        logger.warning(
            f"data_loader: dropped {dup_count} duplicate-timestamp row(s)"
        )
    return df[~df.index.duplicated(keep="first")]
