"""The two bots that share this code (Nov 2026).

* ``m5`` (default): the live M5 bot. M5 zones, trend filter off by
  default (dashboard switch: off / M5 / M15, with the fresh H1 exception), **no candle close exit** (30% SL only, 10 Nov
  2026), £100 a trade, 2 losses a day, 75/25 runner with a 4 point
  trail. Reads ``.env``.
* ``m1``: M1 zones (1 point minimum), trend filter off by default
  (dashboard switch: off / M1 / M5, with the fresh M15 exception), **no candle close exit** (only the 30% SL, TP1 and the
  trail close a trade, Nov 2026), 50/50 runner with a 2 point trail.
  Risk is switchable (``RISK=`` in ``.env.m1`` or ``--risk``): 10% of
  balance (default), 5% or £100 fixed. Losses a day are switchable too
  (``MAX_LOSSES=`` or ``--max-losses``, default 2). The daily cap is one
  trade more than the losses allowed: 2 losses at 10% = 30%. Reads
  ``.env.m1``, so it runs on its own MT5 account and its own Supabase
  project.

**Claude's plan** (19 Nov 2026, on by default for ``m1``; ``M1_PLAN=off``
in ``.env.m1`` turns it off): the M1 bot also trades M5 zones (M5 bot
rules, 17 months of M5), each with its own runner; one trade at a time
but a risk-free runner doesn't block; £50 a trade; -£200 dead stop with
a room check; no losses-a-day / streak stop; drawdown breaker (half risk
£400 below the peak since the plan started, quarter at £600). See
:mod:`bot.plan`. ``RISK=`` / ``MAX_LOSSES=`` in ``.env.m1`` are ignored
while it's on (the dashboard risk setting still works).

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

# Trend filter (12 Nov 2026): OFF by default on both bots, switchable per
# bot from the dashboard (bot_config "trend_filter"):
#   "off"    no trend filter; the trend is still worked out for the badge
#   "own"    trend read on the bot's own chart (M5 bot: M5, M1 bot: M1)
#   "higher" trend read one step up (M5 bot: M15, M1 bot: M5)
# When on, fresh H1 (M5 bot) / M15 (M1 bot) zones still allow a
# counter-trend trade.
TREND_FILTER_MODES = ("off", "own", "higher")
TREND_TIMEFRAMES = {"M5": ("5min", "15min"), "M1": ("1min", "5min")}


def parse_trend_filter(text) -> str:
    """"" / None (bot default) = "off"; otherwise one of TREND_FILTER_MODES."""
    if text is None or str(text).strip() == "":
        return "off"
    mode = str(text).strip().lower()
    if mode not in TREND_FILTER_MODES:
        raise ValueError(f"unknown trend filter {text!r}; use off, own or higher")
    return mode


def trend_gate(gate, bot_name: str, mode: str):
    """The direction gate config for a trend filter mode."""
    own, higher = TREND_TIMEFRAMES[bot_name]
    return replace(
        gate,
        enabled=mode != "off",
        trend_timeframe=higher if mode == "higher" else own,
    )

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


def build_profile(name: str, risk: str | None = None, max_losses=None, *, plan: bool = False):
    """Return ``(BotLoopConfig, StrategyPipelineConfig)`` for a bot.

    ``risk`` (M1 only): see :func:`parse_risk`; default 10%.
    ``max_losses`` (M1 only): losing trades a day before it stops; default 2.
    ``plan`` (M1 only): Claude's plan (see the module docstring)."""
    from bot.main import BotLoopConfig  # local import: main imports this module

    base = BotLoopConfig()
    pipeline = StrategyPipelineConfig()
    if name == "m5":
        if risk is not None:
            raise ValueError("the risk switch is for the M1 bot only; the M5 bot stays at £100")
        if max_losses is not None:
            raise ValueError("the losses switch is for the M1 bot only; the M5 bot stays at 2 losses a day")
        # 12 Nov 2026: trend filter off by default (switch it on from the
        # dashboard); the trend shown is read on M5 with Tommy's BOS rule.
        return replace(
            base, bot_name="M5",
            direction_gate=trend_gate(base.direction_gate, "M5", "off"),
            # 16 Nov 2026: a 0.01-only loss (layer 2 never filled) doesn't
            # use up one of the 2 losses a day. The daily £ limit still counts it.
            # 24 Nov 2026: max 3 trades a day (see h1_trend_ema_period below).
            risk=replace(base.risk, count_first_layer_only_losses=False, max_trades_per_day=3),
            # 24 Nov 2026 (Tommy): only trade with the H1 trend (50 EMA), max 3
            # trades a day, and skip zones whose TP is under 0.3 x the SL
            # distance. 12-month backtest (£50 a trade): +£737 from 205 trades,
            # £3.60 a trade, max drawdown -£448; live rules then +£358 from 862
            # trades, £0.42 a trade, -£871.
            h1_trend_ema_period=50,
            min_reward_frac=0.3,
            # 19 Nov 2026: look back 100,000 M5 bars (about 17 months, was
            # 5000 = 3.5 weeks) so still-fresh zones from when price was last
            # at these levels (e.g. Sep 2025 demand) can be traded. Needs MT5
            # "Max bars in chart" of 100000 or more; falls back to 5000 if not.
            ohlc_count=100_000,
            # 17 Nov 2026: tight 2-3 pt zones allowed (min 2 pts, was 3).
            # 22-month backtest: +£7,214 vs +£6,915, drawdown -£1,295 vs -£1,551.
            # 18 Nov 2026: M / W zones only, no single spikes. Tommy's rule: a
            # spike is not a zone. 20 Nov 2026: the REAL M / W rule (the W / M
            # on the line chart: two bottom / top closes on base candles, at most
            # half the zone apart, nothing closing lower / higher between, 10%
            # middle top / dip; box wick to wick) is the standard for every bot
            # and timeframe (pipeline default).
            # 19 Nov 2026: tried a 2-candle vertex (PR #116); back to 3 the same
            # evening (3-month backtest: 3 candles +£3,643 vs 2 candles +£2,850).
        ), replace(pipeline, zone_min_size_points=2.0, vertex_require_mw=True)
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
            # 12 Nov 2026: trend filter off by default (dashboard switch).
            direction_gate=trend_gate(
                replace(base.direction_gate, htf_timeframe="15min"), "M1", "off",
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
                # 12 Nov 2026: stop for the day after 4 losses in a row
                # (3-month test: +£11,106 vs +£11,085, drawdown -£1,491 vs -£1,547).
                max_loss_streak=4,
                # 19 Nov 2026: a 0.01-only loss (layer 2 never filled) doesn't
                # count toward the losses a day or the 4-in-a-row streak, same
                # as the M5 bot. The daily £ limit still counts its money.
                count_first_layer_only_losses=False,
            ),
        )
        # 12 Nov 2026: BOS must close beyond the swing's wick (M1 only;
        # 3-month backtest: +£11,085 vs +£10,118, drawdown -£1,547 vs -£2,053).
        # 18 Nov 2026: M / W zones only, no single spikes. 20 Nov 2026: the
        # real M / W rule (W / M on the line chart), same as every bot and
        # timeframe (the 18 Nov 3-month M1 test of the old close check: TP
        # rate 68.9% vs 67.0%, 707 trades vs 870).
        m1_pipeline = replace(pipeline, zone_min_size_points=1.0, bos_close_beyond_wick=True,
                              vertex_require_mw=True)
        if plan:
            # 19 Nov 2026: Claude's plan. £50 a trade (fixed), plan risk rules,
            # M5 zones with the M5 bot's settings.
            from bot.main import plan_risk
            from bot.plan import PlanConfig
            pc = PlanConfig(
                m5_pipeline=replace(pipeline, zone_min_size_points=2.0, vertex_require_mw=True),
                m5_entries=replace(base.xautism_entries, m5_close_exit=False),
            )
            fixed = replace(cfg.risk, risk_pct=None, risk_per_trade=50.0,
                            count_first_layer_only_losses=False)
            # 24 Nov 2026 (Tommy): its M5 W / M zones only trade with the H1
            # trend (50 EMA); M1 zones and the FVG are unchanged. 12-month
            # backtest (£50): +£21,601 vs +£19,822, max drawdown -£535 vs
            # -£744, 5 red weeks of 53 vs 8, no losing month vs 1.
            # Same evening: its M5 zones also skip a TP under 0.3 x the SL
            # distance (as the M5 bot). 12-month backtest with the H1 trend:
            # M5 zones 231 trades +£788 vs 466 trades +£764; M1 zones unchanged.
            cfg = replace(cfg, plan=pc, risk=plan_risk(fixed, pc), h1_trend_ema_period=50,
                          min_reward_frac=0.3)
        return cfg, m1_pipeline
    raise ValueError(f"unknown bot {name!r} (expected one of {sorted(ENV_FILES)})")
