"""W / M / N pattern detection on the 1H line chart.

Operates on close prices (line chart, not candles). A W pattern = two
swing lows within the configured tolerance (default 0.1 %, spec Section
13) with a higher peak between them; M is the inverse; N is deferred to
v2 (spec Section 2.3). Output feeds ``zone_marking``.
"""
