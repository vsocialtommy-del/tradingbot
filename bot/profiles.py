"""The two bots that share this code (Nov 2026).

* ``m5`` (default): the live M5 bot. M5 zones, M15 direction with the
  fresh H1 exception, M5 close exit, £100 a trade, 2 losses a day,
  75/25 runner with a 4 point trail. Reads ``.env``.
* ``m1``: M1 zones (1 point minimum), M5 direction with the fresh M15
  exception, M1 close exit, 10% of the balance a trade (daily cap 30%),
  2 losses a day, 50/50 runner with a 2 point trail. Reads ``.env.m1``,
  so it runs on its own MT5 account and its own Supabase project.

Run them with ``python -m bot.main`` (or ``--bot m5``) and
``python -m bot.main --bot m1``, each in its own window.

M1 settings come from the 28 Jul - 21 Oct 2026 backtests (3 months of
M1 data): 50/50 with a 2 pt trail at 10% beat 4 pt, 75/25 and taking
everything at TP1.
"""

from __future__ import annotations

from dataclasses import replace

from bot.strategy.pipeline import StrategyPipelineConfig

ENV_FILES = {"m5": ".env", "m1": ".env.m1"}


def build_profile(name: str):
    """Return ``(BotLoopConfig, StrategyPipelineConfig)`` for a bot."""
    from bot.main import BotLoopConfig  # local import: main imports this module

    base = BotLoopConfig()
    pipeline = StrategyPipelineConfig()
    if name == "m5":
        return replace(base, bot_name="M5"), pipeline
    if name == "m1":
        cfg = replace(
            base,
            bot_name="M1",
            ohlc_timeframe="M1",
            direction_gate=replace(
                base.direction_gate, trend_timeframe="5min", htf_timeframe="15min",
            ),
            xautism_entries=replace(
                base.xautism_entries, runner_keep_frac=0.5, runner_trail_points=2.0,
            ),
            risk=replace(
                base.risk, risk_pct=0.10, max_daily_loss_pct=0.30, max_losses_per_day=2,
            ),
        )
        return cfg, replace(pipeline, zone_min_size_points=1.0)
    raise ValueError(f"unknown bot {name!r} (expected one of {sorted(ENV_FILES)})")
