"""Imbalance Zone tracking (spec Section 3.2).

Tracks how many times price has approached an existing Strong Point zone
without tapping it. A zone with 2+ failed approaches (default 5-10 point
proximity, spec Section 13) qualifies as an Imbalance Zone. The first
actual touch after qualification is treated as a higher-conviction
entry.
"""
