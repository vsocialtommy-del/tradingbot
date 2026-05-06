"""Zone marking and candle-body refinement.

Given a detected W / M pattern, draws an initial box around the reversal
area, then refines top and bottom to candle bodies only (wicks
excluded). Outputs a zone with explicit ``top`` and ``bottom`` price
levels consumed by ``strong_point`` and ``imbalance`` (spec Section 3.1).
"""
