"""High-impact USD news blackout (spec Section 8).

Polls the Supabase ``news_events`` table every ~1 min (the table is
populated by a Vercel cron, not by the bot — Section 8.2). Exposes
``is_news_imminent()`` returning True when any tracked high-impact USD
event is within 30 minutes either side. Bot behaviour during the window:
close open Gold positions 30 min before, no new setups, resume 15 min
after.
"""
