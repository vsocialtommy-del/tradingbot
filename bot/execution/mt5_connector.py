"""MT5 terminal connection wrapper.

Initialises the ``MetaTrader5`` library, logs into the configured Vantage
account, and exposes thin helpers for symbol info, account info, OHLC
fetches, tick subscription, and order send / modify. All other modules
talk to MT5 through this layer so the broker integration stays
swappable.
"""
