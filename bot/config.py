"""Typed bot configuration.

Loads credentials and strategy parameters from ``.env`` via pydantic
settings. Strategy defaults follow spec Section 13 (open items) and are
overridable from the Supabase ``bot_config`` table at runtime so the
dashboard can tune without redeploys.
"""
