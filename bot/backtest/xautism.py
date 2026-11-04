"""XAUTISM backtester (Nov 2026, step 8).

Replays historical bars through the **same** decision code the live bot
uses (pipeline, visibility, spread, M15 direction gate, levels, stacked
plan, SL checks, TP1, fixed-risk sizing, daily risk gates) and simulates
the broker side: 0.01 market on first touch, layer 2 as a stop at the
edge (or a limit at the outer zone when stacked), broker SL / TP on both
layers, M5 close exit, cancel layer 2 when layer 1 closes.

Price path inside a bar
-----------------------
M1 bars are used when available (each M1 bar walks O→L→H→C if bullish,
O→H→L→C if bearish); otherwise the M5 bar walks the same way. Segments
are stepped every ``step`` dollars so things that happen inside one bar
(touch, go deeper, come back out, hit TP) fire in the right order.

Prices in the data are BID. ASK = BID + the bar's spread.

Not simulated: news (no historical calendar), slippage, swaps,
commission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger

from bot.execution.layer2_stop import XautismEntryConfig
from bot.risk.daily_halt import current_trading_date
from bot.risk.xautism_risk import RiskConfig, split_lots
from bot.strategy.direction_gate import DirectionGateConfig, check_direction
from bot.strategy.levels import LevelsConfig, compute_levels
from bot.strategy.pipeline import StrategyPipelineConfig, run_strategy_pipeline
from bot.strategy.stacked import StackedConfig, StackedPlan, stacked_plan
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.tp_target import find_nearest_local_peak
from bot.strategy.zone_visibility import count_bodies_through_zone

Direction = Literal["BUY", "SELL"]
USD_PER_POINT_PER_LOT = 100.0


@dataclass(frozen=True)
class XBacktestConfig:
    window_bars: int = 5000
    m1_window_bars: int = 5000
    step: float = 0.05
    min_sl_distance: float = 1.0
    max_sl_distance: float = 200.0
    use_spread_filter: bool = True
    first_touch_lots: float = 0.01
    xautism: XautismEntryConfig = field(default_factory=XautismEntryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    direction_gate: DirectionGateConfig = field(default_factory=DirectionGateConfig)
    stacked: StackedConfig = field(default_factory=StackedConfig)
    pipeline: StrategyPipelineConfig = field(default_factory=StrategyPipelineConfig)


@dataclass
class Layer:
    lots: float
    entry: float | None = None
    exit: float | None = None
    state: str = "OPEN"  # OPEN / UNPLACED / PENDING / CLOSED / CANCELLED / NONE
    pnl: float = 0.0


@dataclass
class Trade:
    direction: Direction
    zone_bottom: float
    zone_top: float
    edge: float
    far_edge: float
    sl: float
    tp: float
    sl_method: str
    opened_at: pd.Timestamp
    stacked: StackedPlan | None
    l1: Layer
    l2: Layer
    l2_price: float
    l2_kind: str  # STOP / LIMIT / NONE
    trend: str
    planned_risk: float
    closed_at: pd.Timestamp | None = None
    exit_reason: str | None = None

    @property
    def pnl(self) -> float:
        return round(self.l1.pnl + self.l2.pnl, 2)

    def row(self) -> dict[str, Any]:
        return {
            "opened_at": self.opened_at, "closed_at": self.closed_at,
            "direction": self.direction,
            "zone": f"{self.zone_bottom:.2f}-{self.zone_top:.2f}",
            "stacked": (
                f"{self.stacked.outer_timeframe} {self.stacked.outer_bottom:.2f}-"
                f"{self.stacked.outer_top:.2f}" if self.stacked else ""
            ),
            "trend": self.trend,
            "entry_edge": self.edge, "sl": self.sl, "sl_method": self.sl_method,
            "tp": self.tp,
            "l1_lots": self.l1.lots, "l1_entry": self.l1.entry, "l1_exit": self.l1.exit,
            "l2_kind": self.l2_kind, "l2_lots": self.l2.lots, "l2_price": self.l2_price,
            "l2_state": self.l2.state, "l2_entry": self.l2.entry, "l2_exit": self.l2.exit,
            "planned_risk": self.planned_risk,
            "exit_reason": self.exit_reason, "pnl": self.pnl,
        }


@dataclass
class XBacktestResult:
    trades: list[Trade]
    skips: dict[str, int]
    blocked_days: list[str]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.row() for t in self.trades])


def load_mt5_csv(path: str) -> pd.DataFrame:
    """MT5 'Export bars' CSV (tab separated, <DATE> <TIME> ... <SPREAD>)."""
    d = pd.read_csv(path, sep="\t")
    d.columns = [c.strip("<>").lower() for c in d.columns]
    idx = pd.to_datetime(d["date"] + " " + d["time"], format="%Y.%m.%d %H:%M:%S")
    d.index = pd.DatetimeIndex(idx).tz_localize("UTC")
    out = d[["open", "high", "low", "close", "tickvol"]].rename(columns={"tickvol": "volume"})
    out["spread"] = d["spread"].astype(float).values * 0.01
    return out[~out.index.duplicated(keep="first")].sort_index()


def _pnl(direction: str, entry: float, exit_: float, lots: float) -> float:
    move = exit_ - entry if direction == "BUY" else entry - exit_
    return round(move * lots * USD_PER_POINT_PER_LOT, 2)


def _bar_path(o: float, h: float, l: float, c: float) -> list[float]:
    return [o, l, h, c] if c >= o else [o, h, l, c]


class XautismBacktest:
    def __init__(self, m5: pd.DataFrame, m1: pd.DataFrame | None = None,
                 config: XBacktestConfig | None = None) -> None:
        self.cfg = config or XBacktestConfig()
        self.m5 = m5
        self.m1 = m1
        self.trades: list[Trade] = []
        self.skips: dict[str, int] = {}
        self.traded_keys: set[tuple] = set()
        self.day_losses: dict[str, int] = {}
        self.day_pnl: dict[str, float] = {}
        self.blocked_days: set[str] = set()
        self.open: Trade | None = None
        self._m1_cache: tuple[pd.Timestamp, list] | None = None

    # ------------------------------------------------------------------ #

    def run(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
            progress_every: int = 2000) -> XBacktestResult:
        m5 = self.m5
        i0 = max(self.cfg.window_bars // 5, 300)
        if start is not None:
            i0 = max(i0, int(m5.index.searchsorted(start)))
        i1 = len(m5) if end is None else int(m5.index.searchsorted(end))
        for n, i in enumerate(range(i0, i1)):
            if progress_every and n % progress_every == 0:
                logger.info(f"backtest: bar {i}/{i1} {m5.index[i]} trades={len(self.trades)}")
            self._bar(i)
        if self.open is not None:
            last = m5.iloc[i1 - 1]
            self._close_all(float(last["close"]), float(last["spread"]), m5.index[i1 - 1], "END_OF_DATA")
        return XBacktestResult(self.trades, self.skips, sorted(self.blocked_days))

    # ------------------------------------------------------------------ #

    def _skip(self, why: str) -> None:
        self.skips[why] = self.skips.get(why, 0) + 1

    def _day(self, t: pd.Timestamp) -> str:
        return str(current_trading_date(t.to_pydatetime()))

    def _blocked(self, t: pd.Timestamp) -> bool:
        d = self._day(t)
        r = self.cfg.risk
        if self.day_losses.get(d, 0) >= r.max_losses_per_day or self.day_pnl.get(d, 0.0) <= -r.max_daily_loss:
            self.blocked_days.add(d)
            return True
        return False

    def _bar(self, i: int) -> None:
        m5 = self.m5
        t = m5.index[i]
        bar = m5.iloc[i]
        spread = float(bar["spread"])
        closed = m5.iloc[max(0, i - self.cfg.window_bars):i]   # bars closed before this one

        queue: list[ValidatedZone] = []
        can_open = self.open is None and not self._blocked(t)
        if can_open:
            zones = run_strategy_pipeline(closed, self.cfg.pipeline)
            queue = [
                z for z in zones
                if z.is_strong_point and z.refined_zone.is_tradeable
                and _key(z) not in self.traded_keys
            ]
            lo, hi = float(bar["low"]), float(bar["high"]) + spread
            queue = [
                z for z in queue
                if (z.direction == "BUY" and lo <= z.top) or (z.direction == "SELL" and hi >= z.bottom)
            ]
            self._m5_zones = [z for z in zones if z.is_strong_point and z.refined_zone.is_tradeable]
        if self.open is None and not queue:
            return

        failed: set[tuple] = set()
        for p in self._path(i):
            if self.open is not None:
                self._manage(p, spread, t)
                continue
            if not can_open or self._blocked(t):
                continue
            for z in queue:
                k = _key(z)
                if k in failed or k in self.traded_keys:
                    continue
                touched = p <= z.top if z.direction == "BUY" else p + spread >= z.bottom
                if not touched:
                    continue
                if self._try_open(z, closed, p, spread, t):
                    break
                failed.add(k)
            if self.open is not None:
                self._manage(p, spread, t)

        # M5 close exit on this bar's close.
        tr = self.open
        if tr is not None and self.cfg.xautism.m5_close_exit:
            c = float(bar["close"])
            beyond = c < tr.far_edge if tr.direction == "BUY" else c > tr.far_edge
            if beyond:
                self._close_all(c, spread, t, "M5_CLOSE_EXIT")

    def _path(self, i: int) -> list[float]:
        t = self.m5.index[i]
        pts: list[float] = []
        if self.m1 is not None:
            sub = self.m1.loc[t: t + pd.Timedelta(minutes=4)]
            if len(sub) >= 3:
                for _, r in sub.iterrows():
                    pts.extend(_bar_path(r["open"], r["high"], r["low"], r["close"]))
        if not pts:
            r = self.m5.iloc[i]
            pts = _bar_path(r["open"], r["high"], r["low"], r["close"])
        out: list[float] = [float(pts[0])]
        step = self.cfg.step
        for a, b in zip(pts, pts[1:]):
            n = max(1, int(abs(b - a) / step))
            out.extend(np.linspace(a, b, n + 1)[1:].tolist())
        return out

    # ------------------------------------------------------------------ #

    def _try_open(self, z: ValidatedZone, closed: pd.DataFrame, p: float,
                  spread: float, t: pd.Timestamp) -> bool:
        cfg = self.cfg
        if count_bodies_through_zone(closed, zone_top=z.top, zone_bottom=z.bottom,
                                     since_time=pd.Timestamp(z.formed_at)) >= 1:
            self._skip("obscured"); return False
        if cfg.use_spread_filter and spread > cfg.risk.max_spread:
            self._skip("spread"); return False
        trend = ""
        if cfg.direction_gate.enabled:
            dec = check_direction(direction=z.direction, zone_bottom=float(z.bottom),
                                  zone_top=float(z.top), df_m5=closed,
                                  config=cfg.direction_gate, pipeline_cfg=cfg.pipeline)
            if not dec.allowed:
                self._skip("direction_gate"); return False
            trend = dec.reason
        lv = compute_levels(z, closed, cfg.levels)
        entry, sl = lv.entry, lv.sl
        stack = None
        if cfg.stacked.enabled:
            cands = [("M5", c) for c in getattr(self, "_m5_zones", [])]
            cands += [("M1", c) for c in self._m1_zones(t)]
            stack = stacked_plan(z, cands, cfg.stacked)
            if stack is not None:
                sl = stack.sl
        dist = abs(entry - sl)
        if dist < cfg.min_sl_distance or dist > cfg.max_sl_distance:
            self._skip("sl_distance"); return False
        tp = lv.tp1 if lv.tp1 is not None else find_nearest_local_peak(
            closed, entry_price=entry, direction=z.direction,
            lookback_bars=cfg.pipeline.tp1_local_peak_lookback_bars)
        if tp is None:
            self._skip("no_tp"); return False
        if (z.direction == "BUY" and tp <= z.top) or (z.direction == "SELL" and tp >= z.bottom):
            self._skip("tp_wrong_side"); return False

        l2_price = stack.outer_entry if stack else entry
        lots = split_lots(
            risk=cfg.risk.risk_per_trade, layer1_entry=entry, layer2_entry=l2_price, sl=sl,
            loss_per_lot=lambda e, s: abs(e - s) * USD_PER_POINT_PER_LOT,
            first_touch_lots=cfg.first_touch_lots, lot_step=cfg.risk.lot_step,
            max_lot=cfg.risk.max_lot,
        )
        # Layer 1 fill: BUY at ask, SELL at bid, at the touch.
        fill = p + spread if z.direction == "BUY" else p
        far = stack.outer_far_edge if stack else (z.bottom if z.direction == "BUY" else z.top)
        z_far = z.bottom if z.direction == "BUY" else z.top
        if (z.direction == "BUY" and fill < z_far - 0.05) or (z.direction == "SELL" and fill > z_far + 0.05):
            self._skip("gap_through"); return False

        self.traded_keys.add(_key(z))
        self.open = Trade(
            direction=z.direction, zone_bottom=float(z.bottom), zone_top=float(z.top),
            edge=entry, far_edge=float(far), sl=float(sl), tp=float(tp),
            sl_method="STACKED_30" if stack else lv.sl_method, opened_at=t, stacked=stack,
            l1=Layer(lots.layer1, entry=fill),
            l2=Layer(lots.layer2, state="UNPLACED" if lots.layer2 > 0 else "NONE"),
            l2_price=float(l2_price),
            l2_kind="NONE" if lots.layer2 <= 0 else ("LIMIT" if stack else "STOP"),
            trend=trend, planned_risk=lots.risk,
        )
        if stack:
            self.traded_keys.add((z.direction, round(stack.outer_top, 2), round(stack.outer_bottom, 2)))
        return True

    def _m1_zones(self, t: pd.Timestamp) -> list[ValidatedZone]:
        if self.m1 is None or not self.cfg.stacked.use_m1_zones:
            return []
        if self._m1_cache is not None and self._m1_cache[0] == t:
            return self._m1_cache[1]
        j = int(self.m1.index.searchsorted(t))
        if j < 300:
            return []
        zones = run_strategy_pipeline(self.m1.iloc[max(0, j - self.cfg.m1_window_bars):j], self.cfg.pipeline)
        out = [z for z in zones if z.is_strong_point and z.refined_zone.is_tradeable]
        self._m1_cache = (t, out)
        return out

    # ------------------------------------------------------------------ #

    def _manage(self, p: float, spread: float, t: pd.Timestamp) -> None:
        tr = self.open
        assert tr is not None
        bid, ask = p, p + spread
        buy = tr.direction == "BUY"
        l2 = tr.l2
        min_d = self.cfg.xautism.layer2_min_distance

        # Layer 2 placement / fill.
        if l2.state == "UNPLACED":
            if tr.l2_kind == "LIMIT":
                l2.state = "PENDING"
            elif (buy and ask <= tr.l2_price - min_d) or (not buy and bid >= tr.l2_price + min_d):
                l2.state = "PENDING"
        if l2.state == "PENDING":
            if tr.l2_kind == "STOP":
                hit = ask >= tr.l2_price if buy else bid <= tr.l2_price
            else:
                hit = ask <= tr.l2_price if buy else bid >= tr.l2_price
            if hit:
                l2.state, l2.entry = "OPEN", tr.l2_price

        # SL / TP on everything open (same levels on both layers).
        if (buy and bid <= tr.sl) or (not buy and ask >= tr.sl):
            self._close_at(tr.sl, t, "SL")
        elif (buy and bid >= tr.tp) or (not buy and ask <= tr.tp):
            self._close_at(tr.tp, t, "TP")

    def _close_at(self, price: float, t: pd.Timestamp, reason: str) -> None:
        tr = self.open
        assert tr is not None
        for layer in (tr.l1, tr.l2):
            if layer.state == "OPEN":
                layer.exit = price
                layer.pnl = _pnl(tr.direction, layer.entry, price, layer.lots)
                layer.state = "CLOSED"
            elif layer.state in ("UNPLACED", "PENDING"):
                layer.state = "CANCELLED"
        tr.closed_at, tr.exit_reason = t, reason
        self.trades.append(tr)
        d = self._day(t)
        self.day_pnl[d] = self.day_pnl.get(d, 0.0) + tr.pnl
        if tr.pnl < 0:
            self.day_losses[d] = self.day_losses.get(d, 0) + 1
        self.open = None

    def _close_all(self, bid_close: float, spread: float, t: pd.Timestamp, reason: str) -> None:
        tr = self.open
        assert tr is not None
        price = bid_close if tr.direction == "BUY" else bid_close + spread
        self._close_at(price, t, reason)


def _key(z: Any) -> tuple:
    return (z.direction, round(float(z.top), 2), round(float(z.bottom), 2))


def summarise(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0}
    pnl = trades["pnl"]
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    streak = worst = 0
    for v in pnl:
        streak = streak + 1 if v < 0 else 0
        worst = max(worst, streak)
    wins = trades[pnl > 0]
    losses = trades[pnl < 0]
    return {
        "trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1),
        "net": round(pnl.sum(), 2),
        "avg_win": round(wins["pnl"].mean(), 2) if len(wins) else 0.0,
        "avg_loss": round(losses["pnl"].mean(), 2) if len(losses) else 0.0,
        "profit_factor": round(wins["pnl"].sum() / -losses["pnl"].sum(), 2) if len(losses) else None,
        "max_drawdown": round(float(dd), 2),
        "worst_losing_streak": worst,
    }
