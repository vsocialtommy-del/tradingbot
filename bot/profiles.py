"""The two bots that share this code (Nov 2026).

* ``m5`` (default): the live M5 bot. M5 zones, M15 direction with the
  fresh H1 exception, **no candle close exit** (30% SL only, 10 Nov
  2026), £100 a trade, 2 losses a day, 75/25 runner with a 4 point
  trail. Reads ``.env``.
* ``m1``: M1 zones (1 point minimum), M5 direction with the fresh M15
  exception, **no candle close exit** (only the 30% SL, TP1 and the
  trail close a trade, Nov 2026), 50/50 runner with a 2 point trail.
  Risk is switchable (``RISK=`` in ``.env.m1`` or ``--risk``): 10% of
  balance (default), 5% or £100 fixed. Losses a day are switchable too
  (``MAX_LOSSES=`` or ``--max-losses``, default 2). The daily cap is one
  trade more than the losses allowed: 2 losses at 10% = 30%. Reads
  ``.env.m1``, so it runs on its own MT5 account and its own Supabase
  project.

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
M1_DEFAULT_MAX_LOSSES = 2


def parse_max_losses(text) -> int:
    """M1 losses-a-day switch: a whole number from 1 to 50."""
    t = str(text).strip()
    try:
        n = int(t)
        if not 1 <= n <= 50:
            raise ValueError
        return n
    except ValueError:
        raise ValueError(
            f"MAX_LOSSES={text!r} isn't valid. Use a whole number from 1 to 50."
        ) from None


def parse_risk(text: str, max_losses: int = M1_DEFAULT_MAX_LOSSES):
    """M1 risk switch: ``"10%"`` / ``"5%"`` (share of balance per trade) or
    ``"100"`` / ``"£100"`` (fixed). The daily cap is one trade more than
    the losses allowed: with 2 losses that's 30% / 15% of balance, or
    £300 for £100.

    Returns ``(risk_pct, risk_per_trade, max_daily_loss, max_daily_loss_pct)``.
    """
    t = str(text).strip().replace("£", "").replace(" ", "").lower()
    trades = max_losses + 1
    try:
        if t.endswith("%"):
            pct = float(t[:-1]) / 100.0
            if not 0 < pct <= 0.20:
                raise ValueError
            return pct, None, None, round(pct * trades, 4)
        amount = float(t)
        if not 0 < amount <= 10000:
            raise ValueError
        return None, amount, amount * trades, None
    except ValueError:
        raise ValueError(
            f"RISK={text!r} isn't valid. Use a percent like 10% or 5%, "
            "or a fixed amount like 100."
        ) from None


def risk_label(cfg) -> str:
    """How a RiskConfig reads on the dashboard: "1%" or "100"."""
    if cfg.risk_pct is not None:
        return f"{cfg.risk_pct * 100:g}%"
    return f"{cfg.risk_per_trade:g}"


def live_risk(base, risk=None, max_losses=None):
    """Risk settings from the dashboard (``bot_config`` keys ``risk`` and
    ``max_losses``) applied over the bot's own settings ``base``. Either
    may be None (= keep the bot's own). Works for both bots. The daily cap
    stays one trade more than the losses allowed. Raises ValueError on a
    bad value."""
    if (risk is None or str(risk).strip() == "") and max_losses in (None, ""):
        return base
    losses = (
        parse_max_losses(max_losses) if max_losses not in (None, "")
        else base.max_losses_per_day
    )
    text = risk if (risk is not None and str(risk).strip() != "") else risk_label(base)
    pct, amount, daily, daily_pct = parse_risk(text, losses)
    return replace(
        base,
        risk_pct=pct,
        risk_per_trade=amount if amount is not None else base.risk_per_trade,
        max_daily_loss=daily if daily is not None else base.max_daily_loss,
        max_daily_loss_pct=daily_pct,
        max_losses_per_day=losses,
    )


def build_profile(name: str, risk: str | None = None, max_losses=None):
    """Return ``(BotLoopConfig, StrategyPipelineConfig)`` for a bot.

    ``risk`` (M1 only): see :func:`parse_risk`; default 10%.
    ``max_losses`` (M1 only): losing trades a day before it stops; default 2."""
    from bot.main import BotLoopConfig  # local import: main imports this module

    base = BotLoopConfig()
    pipeline = StrategyPipelineConfig()
    if name == "m5":
        if risk is not None:
            raise ValueError("the risk switch is for the M1 bot only; the M5 bot stays at £100")
        if max_losses is not None:
            raise ValueError("the losses switch is for the M1 bot only; the M5 bot stays at 2 losses a day")
        # 12 Nov 2026: the trend is read on the bot's own chart (M5), with
        # Tommy's BOS rule. Fresh H1 zones still allow a counter-trend trade.
        return replace(
            base, bot_name="M5",
            direction_gate=replace(base.direction_gate, trend_timeframe="5min"),
        ), pipeline
    if name == "m1":
        losses = (
            parse_max_losses(max_losses) if max_losses not in (None, "")
            else M1_DEFAULT_MAX_LOSSES
        )
        pct, amount, daily, daily_pct = parse_risk(risk or M1_DEFAULT_RISK, losses)
        cfg = replace(
            base,
            bot_name="M1",
            ohlc_timeframe="M1",
            direction_gate=replace(
                # 12 Nov 2026: trend read on the M1 chart itself.
                base.direction_gate, trend_timeframe="1min", htf_timeframe="15min",
            ),
            xautism_entries=replace(
                base.xautism_entries, runner_keep_frac=0.5, runner_trail_points=2.0,
                # Nov 2026: no candle close exit on the M1 bot. Only the
                # 30% SL, TP1 and the runner trail close a trade.
                m5_close_exit=False,
            ),
            risk=replace(
                base.risk,
                risk_pct=pct,
                max_daily_loss_pct=daily_pct,
                risk_per_trade=amount if amount is not None else base.risk.risk_per_trade,
                max_daily_loss=daily if daily is not None else base.risk.max_daily_loss,
                max_losses_per_day=losses,
            ),
        )
        return cfg, replace(pipeline, zone_min_size_points=1.0)
    raise ValueError(f"unknown bot {name!r} (expected one of {sorted(ENV_FILES)})")
