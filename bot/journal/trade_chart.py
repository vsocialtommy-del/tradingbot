"""Trade photo journal (Tommy, 26 Nov 2026).

One picture per closed trade that tells the whole story: the candles on the
trade's own timeframe, the zone, H1 trend (50 EMA), BOS / TP1, entry, SL,
every layer's entry and exit (runner included), a line chart of closes
underneath showing the M / W shape, and a table of the layers. Plus a
summary dict of the same numbers for analysis later.

Pure: takes data in, gives PNG bytes out. No MT5, no Supabase.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

UK = ZoneInfo("Europe/London")

BG = "#0b0b0c"
PANEL = "#141416"
GRID = "#26262a"
GOLD = "#d4af37"
WHITE = "#f2f2f2"
MUTED = "#9a9aa2"
UP = "#22c55e"
DOWN = "#ef4444"
BLUE = "#60a5fa"
_LABEL_BOX = {"boxstyle": "round,pad=0.15", "facecolor": BG, "edgecolor": "none", "alpha": 0.8}


@dataclass
class Layer:
    name: str
    volume: float
    open_time: datetime          # UTC
    open_price: float
    close_time: datetime         # UTC
    close_price: float
    profit: float
    how: str = ""


@dataclass
class ChartTrade:
    bot_label: str               # "CLAUDE FTMO"
    account: int | None
    setup_key: str
    direction: str               # BUY / SELL
    tf: str                      # M1 / M5 (the zone's timeframe)
    kind: str                    # "W/M" / "FVG"
    layers: list[Layer]
    zone_top: float | None = None
    zone_bottom: float | None = None
    zone_formed: datetime | None = None   # UTC
    entry: float | None = None   # planned entry (zone edge)
    sl: float | None = None
    tp1: float | None = None     # = the BOS level the zone broke
    h1_state: str | None = None  # UPTREND / DOWNTREND at entry
    h1_close: float | None = None
    h1_ema: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def opened(self) -> datetime:
        return min(x.open_time for x in self.layers)

    @property
    def closed(self) -> datetime:
        return max(x.close_time for x in self.layers)

    @property
    def profit(self) -> float:
        return round(sum(x.profit for x in self.layers), 2)


def classify_exit(t: ChartTrade, layer: Layer, tol: float = 0.6) -> str:
    """How a layer closed, from its close price vs the plan."""
    buy = t.direction == "BUY"
    px = layer.close_price
    gain = (px - layer.open_price) if buy else (layer.open_price - px)
    past = None
    if t.tp1 is not None:
        past = (px - t.tp1) if buy else (t.tp1 - px)
    if t.sl is not None and abs(px - t.sl) <= tol:
        return "SL"
    if layer.name.endswith("runner"):
        if abs(gain) <= tol:
            return "runner: breakeven stop"
        if past is not None and past > 0:
            return f"runner: trailed +{past:.1f} past TP1"
        return "runner stopped" if gain < 0 else "runner: trail stop"
    if t.tp1 is not None and abs(px - t.tp1) <= tol:
        return "TP1"
    if abs(gain) <= 0.35:
        return "breakeven"
    if past is not None and past > 0:
        return f"trail (+{past:.1f} past TP1)"
    return "closed early" if gain < 0 else "closed in profit"


def summary(t: ChartTrade, candles: pd.DataFrame | None = None) -> dict[str, Any]:
    """The numbers behind the picture, for analysis later."""
    buy = t.direction == "BUY"
    entry = t.entry if t.entry is not None else t.layers[0].open_price
    risk = abs(entry - t.sl) if t.sl is not None else None
    reward = ((t.tp1 - entry) if buy else (entry - t.tp1)) if t.tp1 is not None else None
    out: dict[str, Any] = {
        "bot": t.bot_label, "account": t.account, "setup_key": t.setup_key,
        "direction": t.direction, "timeframe": t.tf, "kind": t.kind,
        "h1_trend": t.h1_state, "h1_close": t.h1_close, "h1_ema": t.h1_ema,
        "with_h1_trend": (None if t.h1_state is None
                          else (t.h1_state == "UPTREND") == buy),
        "zone_top": t.zone_top, "zone_bottom": t.zone_bottom,
        "zone_height": (round(t.zone_top - t.zone_bottom, 2)
                        if t.zone_top is not None and t.zone_bottom is not None else None),
        "zone_formed": t.zone_formed.isoformat() if t.zone_formed else None,
        "entry": entry, "sl": t.sl, "tp1": t.tp1,
        "risk_pts": round(risk, 2) if risk is not None else None,
        "reward_pts": round(reward, 2) if reward is not None else None,
        "rr": round(reward / risk, 2) if risk and reward is not None else None,
        "opened_at": t.opened.isoformat(), "closed_at": t.closed.isoformat(),
        "minutes": round((t.closed - t.opened).total_seconds() / 60, 1),
        "profit": t.profit,
        "result": "WIN" if t.profit > 0 else ("LOSS" if t.profit < 0 else "BREAKEVEN"),
        "layers": [
            {"name": x.name, "volume": x.volume, "open_time": x.open_time.isoformat(),
             "open_price": x.open_price, "close_time": x.close_time.isoformat(),
             "close_price": x.close_price, "profit": round(x.profit, 2), "how": x.how}
            for x in t.layers
        ],
    }
    if candles is not None and len(candles):
        during = candles[(candles.index >= _floor(t.opened, candles)) & (candles.index <= t.closed)]
        if len(during):
            hi, lo = float(during["high"].max()), float(during["low"].min())
            out["max_against_pts"] = round((entry - lo) if buy else (hi - entry), 2)
            out["max_for_pts"] = round((hi - entry) if buy else (entry - lo), 2)
        after = candles[candles.index > t.closed]
        if len(after):
            hi, lo = float(after["high"].max()), float(after["low"].min())
            out["after_close_for_pts"] = round((hi - entry) if buy else (entry - lo), 2)
            out["after_close_against_pts"] = round((entry - lo) if buy else (hi - entry), 2)
            if t.sl is not None:
                out["sl_hit_after_close"] = bool(lo <= t.sl if buy else hi >= t.sl)
            if t.tp1 is not None:
                out["tp1_hit_after_close"] = bool(hi >= t.tp1 if buy else lo <= t.tp1)
    return out


def _floor(ts: datetime, candles: pd.DataFrame) -> pd.Timestamp:
    i = int(candles.index.searchsorted(pd.Timestamp(ts), side="right")) - 1
    return candles.index[max(i, 0)]


def _uk(ts: Any) -> datetime:
    return pd.Timestamp(ts).tz_convert(UK).to_pydatetime()


def _money(v: float) -> str:
    return f"{'+' if v > 0 else '-' if v < 0 else ''}£{abs(v):,.2f}"


def render(t: ChartTrade, candles: pd.DataFrame, h1_ema: pd.Series | None = None) -> bytes:
    """PNG of the whole trade. ``candles``: the trade timeframe's OHLC with a
    UTC index, already cut to the window to show. ``h1_ema``: the H1 50 EMA
    (UTC index = when each H1 candle closed)."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    c = candles.sort_index()
    n = len(c)
    if n == 0:
        raise ValueError("no candles to draw")
    idx = c.index

    def x_of(ts: Any) -> float:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        i = int(idx.searchsorted(ts, side="right")) - 1
        if i < 0:
            return -0.5
        step = (idx[1] - idx[0]) if n > 1 else pd.Timedelta(minutes=1)
        return i + min(max((ts - idx[i]) / step, 0.0), 1.0) - 0.5

    buy = t.direction == "BUY"
    side = UP if buy else DOWN
    win = t.profit > 0
    loss = t.profit < 0

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(9, 11.5), dpi=120, facecolor=BG)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.15, 5.2, 2.0, 1.75], hspace=0.12,
                          left=0.085, right=0.84, top=0.985, bottom=0.02)
    head = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    lx = fig.add_subplot(gs[2], sharex=ax)
    tab = fig.add_subplot(gs[3])
    for a in (head, tab):
        a.axis("off")
        a.set_facecolor(BG)

    # ---- header ---------------------------------------------------------
    trend = t.h1_state or "H1 TREND ?"
    trend_txt = f"H1 {trend}" if t.h1_state else trend
    trend_col = UP if t.h1_state == "UPTREND" else DOWN if t.h1_state == "DOWNTREND" else MUTED
    acct = f"  ·  #{t.account}" if t.account else ""
    head.text(0.0, 0.97, f"{t.bot_label}{acct}  ·  {t.kind}", color=GOLD, fontsize=11,
              fontweight="bold", va="top", transform=head.transAxes)
    head.text(1.19, 0.97, _money(t.profit) + ("  WIN" if win else "  LOSS" if loss else "  BREAKEVEN"),
              color=UP if win else DOWN if loss else MUTED, fontsize=15, fontweight="bold",
              va="top", ha="right", transform=head.transAxes)
    x0 = 0.0
    renderer = canvas.get_renderer()
    for txt, col in ((t.direction, side), ("  ·  ", MUTED), (f"{t.tf} ZONE", GOLD), ("  ·  ", MUTED),
                     (trend_txt, trend_col)):
        tt = head.text(x0, 0.62, txt, color=col, fontsize=17, fontweight="bold", va="top",
                       transform=head.transAxes)
        x0 += tt.get_window_extent(renderer).width / head.get_window_extent(renderer).width
    o, cl = _uk(t.opened), _uk(t.closed)
    mins = (t.closed - t.opened).total_seconds() / 60
    dur = f"{mins:.0f} min" if mins < 120 else f"{mins / 60:.1f} h"
    head.text(0.0, 0.30, f"{o:%a %d %b %Y}  ·  opened {o:%H:%M}  ·  closed {cl:%H:%M} UK  ·  {dur}",
              color=WHITE, fontsize=10, va="top", transform=head.transAxes)
    bits = []
    entry = t.entry if t.entry is not None else t.layers[0].open_price
    if t.zone_top is not None and t.zone_bottom is not None:
        bits.append(f"Zone {t.zone_bottom:.2f}-{t.zone_top:.2f} ({t.zone_top - t.zone_bottom:.1f} pts)")
    if t.sl is not None:
        bits.append(f"SL {t.sl:.2f} ({abs(entry - t.sl):.1f} pts)")
    if t.tp1 is not None:
        rew = (t.tp1 - entry) if buy else (entry - t.tp1)
        bits.append(f"BOS/TP1 {t.tp1:.2f} ({rew:+.2f} pts)")
        if t.sl is not None and abs(entry - t.sl) > 0:
            bits.append(f"R:R {rew / abs(entry - t.sl):.2f}")
    if t.h1_close is not None and t.h1_ema is not None:
        bits.append(f"H1 close {t.h1_close:.1f} vs 50 EMA {t.h1_ema:.1f}")
    head.text(0.0, 0.08, "  ·  ".join(bits), color=MUTED, fontsize=8.6, va="top",
              transform=head.transAxes)

    # ---- candles ------------------------------------------------------------
    for a in (ax, lx):
        a.set_facecolor(PANEL)
        a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(GRID)
        a.grid(True, color=GRID, linewidth=0.5)
        a.yaxis.tick_left()
    xs = np.arange(n)
    o_, h_, l_, c_ = (c[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    upm = c_ >= o_
    ax.vlines(xs, l_, h_, color=np.where(upm, "#d9d9d9", "#8a8a8a"), linewidth=0.8, zorder=2)
    w = 0.62
    for i in range(n):
        lo, hi = min(o_[i], c_[i]), max(o_[i], c_[i])
        ax.add_patch(Rectangle((i - w / 2, lo), w, max(hi - lo, 1e-3),
                               facecolor="#f2f2f2" if upm[i] else "#1b1b1e",
                               edgecolor="#d9d9d9" if upm[i] else "#8a8a8a", linewidth=0.6, zorder=3))

    xa, xb = x_of(t.opened), x_of(t.closed)
    for a in (ax, lx):
        a.axvspan(xa, xb, color=GOLD, alpha=0.06, zorder=0)

    # zone box from when it formed to the trade's close
    if t.zone_top is not None and t.zone_bottom is not None:
        zx = x_of(t.zone_formed) if t.zone_formed is not None else -0.5
        zx = max(zx, -0.5)
        for a in (ax, lx):
            a.add_patch(Rectangle((zx, t.zone_bottom), max(xb - zx, 1), t.zone_top - t.zone_bottom,
                                  facecolor=side, alpha=0.16, edgecolor=side, linewidth=1.0, zorder=1))
        if t.zone_formed is not None and zx > 0:
            ax.annotate(f"zone formed {_uk(t.zone_formed):%H:%M}", (zx, t.zone_top if not buy else t.zone_bottom),
                        xytext=(2, 8 if not buy else -14), textcoords="offset points",
                        color=side, fontsize=7.5,
                        bbox=_LABEL_BOX)

    # H1 50 EMA
    if h1_ema is not None and len(h1_ema):
        e = h1_ema[(h1_ema.index >= idx[0] - pd.Timedelta(hours=1)) & (h1_ema.index <= idx[-1])]
        if len(e):
            ex = [x_of(ts) for ts in e.index] + [n - 0.5]
            ey = list(e.to_numpy(float)) + [float(e.iloc[-1])]
            ax.step(ex, ey, where="post", color=BLUE, linewidth=1.3, alpha=0.9, zorder=4)
            ax.text(n - 0.5, ey[-1], " H1 50 EMA", color=BLUE, fontsize=7.5, va="center",
                    clip_on=False)

    levels = [(y, col, lab, ls) for y, col, lab, ls in (
        (t.sl, DOWN, "SL", "--"), (t.tp1, UP, "BOS / TP1", "--"), (t.entry, GOLD, "ENTRY", ":"))
        if y is not None]
    for y, col, _, ls in levels:
        for a in (ax, lx):
            a.axhline(y, color=col, linestyle=ls, linewidth=1.1, alpha=0.95, zorder=4)

    # layers: entry arrow, exit cross, dotted path between
    for k, ly in enumerate(t.layers):
        ox, cx = x_of(ly.open_time), x_of(ly.close_time)
        col = UP if ly.profit > 0 else DOWN if ly.profit < 0 else MUTED
        for a in (ax, lx):
            a.plot([ox, cx], [ly.open_price, ly.close_price], linestyle=":", color=col,
                   linewidth=1.1, zorder=5)
            a.scatter([ox], [ly.open_price], marker="^" if buy else "v", s=60, color=side,
                      edgecolor=WHITE, linewidth=0.6, zorder=6)
            a.scatter([cx], [ly.close_price], marker="X", s=55, color=col, edgecolor=WHITE,
                      linewidth=0.6, zorder=6)
        ax.annotate(f"{ly.name} {_money(ly.profit)}", (cx, ly.close_price),
                    xytext=(8, -14 - 13 * k if buy else 8 + 13 * k), textcoords="offset points",
                    color=col, fontsize=7.5, fontweight="bold", zorder=7,
                    bbox=_LABEL_BOX)

    lows = [float(np.nanmin(l_))] + [v for v in (t.sl, t.tp1, t.zone_bottom) if v is not None]
    highs = [float(np.nanmax(h_))] + [v for v in (t.sl, t.tp1, t.zone_top) if v is not None]
    lo, hi = min(lows), max(highs)
    pad = (hi - lo) * 0.06 or 1.0
    ax.set_ylim(lo - pad, hi + pad)
    # H1 50 EMA off the top / bottom of the chart: say where it is.
    if t.h1_ema is not None and not (lo - pad <= t.h1_ema <= hi + pad):
        above = t.h1_ema > hi + pad
        ax.text(0.99, 0.985 if above else 0.015,
                f"H1 50 EMA {t.h1_ema:.1f} {'above' if above else 'below'} the chart",
                color=BLUE, fontsize=8, ha="right", va="top" if above else "bottom",
                transform=ax.transAxes)
    # Level labels on the right, pushed apart when the lines are close.
    gap = (hi - lo + 2 * pad) * 0.035
    placed: list[float] = []
    for y, col, lab, _ in sorted(levels, key=lambda v: v[0]):
        ty = y if not placed else max(y, placed[-1] + gap)
        placed.append(ty)
        ax.annotate(f" {lab} {y:.2f}", (n - 0.5, y), xytext=(n - 0.5, ty), textcoords="data",
                    color=col, fontsize=8, va="center", fontweight="bold", annotation_clip=False)
    ax.set_xlim(-1, n + 0.5)
    ax.text(0.01, 0.985, f"{t.tf} candles", color=MUTED, fontsize=8, va="top",
            transform=ax.transAxes)
    ax.tick_params(labelbottom=False)

    # ---- line chart ---------------------------------------------------------
    lx.plot(xs, c_, color=GOLD, linewidth=1.3, zorder=3)
    llo = min([float(np.nanmin(c_))] + [v for v in (t.zone_bottom, t.tp1) if v is not None])
    lhi = max([float(np.nanmax(c_))] + [v for v in (t.zone_top, t.tp1) if v is not None])
    lpad = (lhi - llo) * 0.08 or 1.0
    lx.set_ylim(llo - lpad, lhi + lpad)
    lx.text(0.01, 0.96, "line chart (closes): the M / W shape", color=MUTED, fontsize=8,
            va="top", transform=lx.transAxes)
    ticks = np.linspace(0, n - 1, num=min(7, n)).round().astype(int)
    lx.set_xticks(ticks)
    lx.set_xticklabels([f"{_uk(idx[i]):%H:%M}" for i in ticks])

    # ---- layer table ----------------------------------------------------------
    rows = [["Layer", "Lots", "Open (UK)", "Close (UK)", "How it closed", "P&L"]]
    for ly in t.layers:
        rows.append([ly.name, f"{ly.volume:.2f}",
                     f"{_uk(ly.open_time):%H:%M:%S} @ {ly.open_price:.2f}",
                     f"{_uk(ly.close_time):%H:%M:%S} @ {ly.close_price:.2f}",
                     ly.how, _money(ly.profit)])
    rows.append(["Total", f"{sum(x.volume for x in t.layers):.2f}", "", "", "", _money(t.profit)])
    tb = tab.table(cellText=rows, loc="upper center", cellLoc="left",
                   colWidths=[0.1, 0.07, 0.25, 0.25, 0.27, 0.11], bbox=[0, 0.12, 1.19, 0.88])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.2)
    for (r, ci), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(PANEL if r else "#1d1d21")
        txt = cell.get_text()
        txt.set_color(GOLD if r == 0 else WHITE)
        if ci == 5 and r > 0:
            v = t.profit if r == len(rows) - 1 else t.layers[r - 1].profit
            txt.set_color(UP if v > 0 else DOWN if v < 0 else MUTED)
        if r == len(rows) - 1:
            txt.set_fontweight("bold")
    if t.notes:
        tab.text(0.0, 0.02, "  ·  ".join(t.notes), color=MUTED, fontsize=7.5,
                 transform=tab.transAxes)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return buf.getvalue()
