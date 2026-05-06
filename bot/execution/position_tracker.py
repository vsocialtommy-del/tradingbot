"""Filled-layer position state.

Tracks which of the three layers per setup are filled, computes the
average entry across filled layers (used by TP1 break-even), and
reconciles MT5's reported positions against the bot's expected state.
Owns the lifecycle: pending → partially filled → filled → TP1 → runner
→ closed.
"""
