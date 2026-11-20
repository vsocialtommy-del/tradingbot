"""Claude's plan (19 Nov 2026): one bot trading M5 AND M1 zones.

Real M / W zones (20 Nov 2026): both timeframes use Tommy's M / W rule,
the W / M on the line chart (see :func:`bot.strategy.vertex.is_real_mw`).

Runs on the M1 bot's account (``--bot m1``). Backtested 28 Jul - 13 Nov
2026 (3 months) as the merged bot.

* **Zones**: M1 zones exactly as the M1 bot (1 pt min, wick-close BOS,
  real M / W only, 5,000 M1 candles) PLUS M5 zones exactly as the M5 bot
  (2 pt min, normal BOS, real M / W only, 100,000 M5 candles ≈ 17 months).
  Each zone keeps its own timeframe's BOS / TP and its own runner:
  M5 75/25 with a 4 pt trail, M1 50/50 with a 2 pt trail.
* **M5 priority** (19 Nov 2026): an M5 trade can open beside an open M1
  trade; an M1 trade only when nothing else is open. An M1 zone that
  overlaps a fresh M5 zone is traded as M5. A risk-free runner (TP1 hit,
  stop at breakeven or better) doesn't block anything.
* Vertex minimum 3 candles on both timeframes (a 2-candle minimum for
  M5 zones was tried on 19 Nov 2026 and taken back the same evening).
* **Risk**: £50 a trade on both. A -£200 dead stop for the day with a
  room check: no trade opens if the day's P&L minus that trade's risk
  would go below -£200. No losses-a-day count, no losses-in-a-row stop,
  no profit target.
* **Drawdown breaker**: half risk once the balance is £400 below its
  peak, a quarter at £600 below. The peak counts from when the plan
  started on this account (stored in ``logs/plan_state.json``), not the
  account's all-time high.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from bot.execution.layer2_stop import XautismEntryConfig
from bot.strategy.pipeline import StrategyPipelineConfig


@dataclass(frozen=True)
class PlanConfig:
    m5_ohlc_count: int = 100_000
    """M5 candles scanned for M5 zones (about 17 months)."""
    m5_fallback_count: int = 5000
    m5_pipeline: StrategyPipelineConfig = field(default_factory=lambda: StrategyPipelineConfig(
        zone_min_size_points=2.0, vertex_require_mw=True,
    ))
    """Same zone rules as the live M5 bot."""
    m5_entries: XautismEntryConfig = field(default_factory=lambda: XautismEntryConfig(
        runner_keep_frac=0.25, runner_trail_points=4.0, m5_close_exit=False,
    ))
    """M5 trades: 75% at TP1, 25% trails 4 pts (as the M5 bot)."""
    daily_stop: float = 200.0
    runner_frees_slot: bool = True
    m5_priority: bool = True
    """19 Nov 2026: an M5 trade can open beside an open M1 trade (never two
    M5); an M1 trade only when nothing else is open (risk-free runners
    aside). An M1 zone that overlaps a fresh M5 zone is traded as the M5
    trade. A second trade only opens if the day's realised P&L, plus every
    open position and waiting order going to its stop, minus the new
    trade's risk, stays inside the dead stop."""
    drawdown_steps: tuple[tuple[float, float], ...] = ((600.0, 0.25), (400.0, 0.5))
    """(£ below peak, risk multiplier), biggest drawdown first."""
    state_file: str = "logs/plan_state.json"


def drawdown_multiplier(peak: float, balance: float,
                        steps: tuple[tuple[float, float], ...]) -> float:
    dd = peak - balance
    for below, mult in steps:
        if dd >= below:
            return mult
    return 1.0


class DrawdownBreaker:
    """Tracks the balance peak since the plan started on this account and
    gives the risk multiplier (1, 0.5 or 0.25). File-backed so a restart
    keeps the peak; a different account login starts fresh."""

    def __init__(self, path: str | Path, steps: tuple[tuple[float, float], ...],
                 *, refresh_seconds: float = 30.0) -> None:
        self._path = Path(path)
        self._steps = steps
        self._refresh = refresh_seconds
        self._state: dict[str, Any] | None = None
        self._checked = 0.0
        self._mult = 1.0

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _save(self, state: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(state))
        except Exception:
            logger.exception("plan: couldn't save the drawdown peak")

    def update(self, login: Any, balance: float) -> float:
        """Record ``balance`` for ``login``; return the risk multiplier."""
        if self._state is None:
            self._state = self._load()
        st = self._state
        if str(st.get("login")) != str(login) or "peak" not in st:
            st = self._state = {"login": login, "start_balance": balance, "peak": balance}
            self._save(st)
            logger.info(f"plan: drawdown breaker starts from balance {balance:.2f} (account {login})")
        elif balance > float(st["peak"]):
            st["peak"] = balance
            self._save(st)
        mult = drawdown_multiplier(float(st["peak"]), balance, self._steps)
        if mult != self._mult:
            logger.info(
                f"plan: balance {balance:.2f} is {float(st['peak']) - balance:.2f} below the "
                f"peak {float(st['peak']):.2f}, risk x{mult:g}"
            )
        self._mult = mult
        return mult

    def multiplier(self, mt5: Any) -> float:
        """Cached for ``refresh_seconds``; keeps the last value on errors."""
        now = time.monotonic()
        if self._checked and now - self._checked < self._refresh:
            return self._mult
        self._checked = now
        try:
            info = mt5.get_account_info()
            return self.update(info.get("login"), float(info["balance"]))
        except Exception:
            logger.exception("plan: drawdown check failed; keeping the last risk multiplier")
            return self._mult

    @property
    def peak(self) -> float | None:
        if not self._state:
            return None
        return float(self._state.get("peak")) if "peak" in self._state else None
