"""Concurrent-setup cap (spec Section 7).

Blocks new setup activation when 2-3 setups are already live so total
risk across simultaneous trades stays bounded. Counts a setup as live
from the moment Layer 1 fills until all layers are closed.
"""
