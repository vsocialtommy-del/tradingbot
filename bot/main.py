"""Bot entry point.

Wires together the strategy, execution, risk, exits, filters, data, and
logging subsystems and runs the live event loop. Polls Supabase
``bot_config`` every ~30 s for kill-switch / parameter changes, and
``news_events`` every ~1 min for the news filter (spec Sections 8, 9.6).
"""
