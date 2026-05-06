"""Strong Point validation — the foundation setup (spec Section 3.1).

Validates a candidate refined zone by checking three quality criteria:
1. The move out of the zone broke a previous structure level (BoS).
2. The base before the impulsive move = small consolidation candles.
3. The impulsive move = strong-bodied candles, not just wicks.

A passing zone becomes a tradeable Strong Point; failures are discarded.
"""
