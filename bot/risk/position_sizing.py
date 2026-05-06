"""Position sizing per layer.

v1: returns a fixed 0.01 lot for every layer regardless of balance —
used for entry/exit logic validation only (spec Section 4.4).

v1.1+: 0.33 % of account balance per layer (1 % per setup across three
layers), computed from SL distance and the contract spec for XAUUSD
(spec Section 7).
"""
