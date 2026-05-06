"""Real-time tick stream.

Powers Layer 1 entry detection (spec Section 4.2): the moment a tick
prints at or beyond a refined zone edge, the order manager fires the
market order. Also feeds the gap-protection check (Section 4.3) by
exposing the prior tick alongside the current one so a single-update
gap through the zone can be recognised.
"""
