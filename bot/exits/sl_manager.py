"""Initial SL placement and lifecycle (spec Section 5).

For BUY: SL = recent lowest low (last N M5 candles, configurable) minus
a 15-20 point buffer; for SELL: recent highest high plus the buffer. All
three layers share one SL so a pre-TP1 stop closes the whole setup at
exactly 1 % account risk.

After TP1 the SL on the remaining 50 % is moved to break-even by
``tp1_manager``; this module does not auto-trail (spec Section 5.2).
"""
