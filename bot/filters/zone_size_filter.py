"""Zone-size sanity filter (spec Section 7).

Skips setups where the refined zone is below or above the configured
size band (default 5-80 points; both ends are tunable in backtest per
Section 11.3). Below the lower bound the zone is too narrow to be
meaningful; above the upper bound SL distance bloats and the 1 % risk
math gets unfavourable.
"""
