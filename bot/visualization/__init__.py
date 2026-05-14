"""Bot visualization — operator-facing output of bot state.

PR #49: MT5 chart zone overlay. The bot writes a snapshot of the zones
it currently sees to a CSV file that a companion MQL5 EA reads and
renders as colored rectangles on the live MT5 chart.

This package is intentionally minimal — visualization is operator
diagnostics, not a trading-critical path. Failures here MUST NOT
affect the trading loop.
"""
