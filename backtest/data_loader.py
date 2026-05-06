"""Historical data loader.

Reads Dukascopy CSV exports (M1 by default), normalises timestamps to
UTC, resamples up to M5 / 1H as needed by the strategy, and yields the
same DataFrame shape that ``ohlc_provider`` produces live so the
strategy modules don't care whether they're running historical or live.
"""
