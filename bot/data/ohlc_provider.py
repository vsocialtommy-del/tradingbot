"""OHLC bar fetcher.

Pulls 1H bars (zone / structure timeframe) and M5 bars (entry timeframe)
from MT5 via ``mt5_connector``, returns them as pandas DataFrames keyed
by UTC timestamp, and handles incremental updates so the strategy
modules only ever see a consistent rolling window.
"""
