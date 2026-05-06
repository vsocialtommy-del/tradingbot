"""Daily loss limit enforcement (spec Section 7).

At each broker-day rollover (17:00 EST) records ``starting_balance``.
After every closed trade, if ``current_balance < starting_balance * 0.90``
the bot cancels pending orders, refuses new setups, and lets open
positions run their course until next day's reset.
"""
