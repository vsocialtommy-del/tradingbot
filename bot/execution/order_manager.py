"""Layered entry order placement (spec Section 4).

On first touch of a refined zone:
- Layer 1: market order (real-time tick trigger).
- Layer 2: pending limit at zone midpoint.
- Layer 3: pending limit at zone bottom (BUY) / top (SELL).

Implements the gap-protection rule (Section 4.3) — skips the setup if a
single tick gaps through the entire zone — and partial-fill handling
(Section 4.5) — cancels remaining pending layers if price reverses.
"""
