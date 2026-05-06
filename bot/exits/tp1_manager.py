"""TP1 partial-take and break-even move (spec Section 6.1).

Watches each open setup for price reaching ``zone edge ± $4`` (default,
configurable in the $2-10 range per Section 11.3). On trigger:
1. Closes 50 % of total filled position (half of each filled layer).
2. Moves SL on the remaining 50 % to the average entry of filled layers
   (break-even).

The runner that remains is then handed off to manual management — the
bot does not set TP2 and does not trail (spec Section 6.2).
"""
