"""Supabase write layer.

Persists trades, setups, zones, daily PnL, and structured bot logs to
the Supabase tables defined in spec Section 9.2. The dashboard (Phase E)
reads these via Supabase realtime subscriptions, so every write here
also drives the UI.
"""
