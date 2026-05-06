"""Telegram alerts for trade events.

Sends notifications to Tommy's phone on key lifecycle events: setup
detected, layer filled, TP1 hit, runner handed off, SL hit, daily halt
triggered, bot error. Uses the bot token + chat ID from ``.env``;
alerting is best-effort and never blocks trade execution.
"""
