"""Market structure tracking on 1H bars.

Identifies swing highs / lows, tags them as HH/HL/LL/LH, and detects
Break of Structure (BoS) events. Used by ``strong_point`` for quality
validation (spec Section 3.1) and by ``sl_manager`` to find the recent
swing for SL placement (spec Section 5.1, default 20-candle lookback).
"""
