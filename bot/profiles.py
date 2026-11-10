"""The two bots that share this code (Nov 2026).

* ``m5`` (default): the live M5 bot. M5 zones, M15 direction with the
  fresh H1 exception, M5 close exit, £100 a trade, 2 losses a day,
  75/25 runner with a 4 point trail. Reads ``.env``.
* ``m1``: M1 zones (1 point minimum), M5 direction with the fresh M15
  exception, M1 close exit, 2 losses a day, 50/50 runner with a 2 point
  trail. Risk is switchable (``RISK=`` in ``.env.m1`` or ``--risk``):
  10% of balance (default, daily cap 30%), 5% (cap 15%) or £100 fixed
  (cap £300). Reads ``.env.m1``, so it runs on its own MT5 account and
  its own Supabase project.

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

M1_DEFAULT_RISK = "10%"


def parse_risk(text: str):
    """M1 risk switch: ``"10%"`` / ``"5%"`` (share of balance per trade) or
    ``"100"`` / ``"£100"`` (fixed). Always 2 losses a day. The daily cap is
    3 trades' worth: 30% / 15% of balance, or £300 for £100.

    Returns ``(risk_pct, risk_per_trade, max_daily_loss, max_daily_loss_pct)``.
    """
    t = str(text).strip().replace("£", "").replace(" ", "").lower()
    try:
        if t.endswith("%"):
            pct = float(t[:-1]) / 100.0
            if not 0 < pct <= 0.20:
                raise ValueError
            return pct, None, None, round(pct * 3, 4)
        amount = float(t)
        if not 0 < amount <= 10000:
            raise ValueError
        return None, amount, amount * 3, None
    except ValueError:
        raise ValueError(
            f"RISK={text!r} isn't valid. Use a percent like 10% or 5%, "
            "or a fixed amount like 100."
        ) from None


def build_profile(name: str, risk: str | None = None):
    """Return ``(BotLoopConfig, StrategyPipelineConfig)`` for a bot.

    ``risk`` (M1 only): see :func:`parse_risk`; default 10%."""
    from bot.main import BotLoopConfig  # local import: main imports this module

    base = BotLoopConfig()
    pipeline = StrategyPipelineConfig()
    if name == "m5":
        if risk is not None:
            raise ValueError("the risk switch is for the M1 bot only; the M5 bot stays at £100")
        return replace(base, bot_name="M5"), pipeline
    if name == "m1":
        pct, amount, daily, daily_pct = parse_risk(risk or M1_DEFAULT_RISK)
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
                base.risk,
                risk_pct=pct,
                max_daily_loss_pct=daily_pct,
                risk_per_trade=amount if amount is not None else base.risk.risk_per_trade,
                max_daily_loss=daily if daily is not None else base.risk.max_daily_loss,
                max_losses_per_day=2,
            ),
        )
        return cfg, replace(pipeline, zone_min_size_points=1.0)
    raise ValueError(f"unknown bot {name!r} (expected one of {sorted(ENV_FILES)})")
