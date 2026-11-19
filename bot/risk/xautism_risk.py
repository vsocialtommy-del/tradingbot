"""XAUTISM risk gates (Nov 2026, step 7).

* Fixed money risk per trade (both layers together), sized off the
  broker's own loss-per-lot so it's right in the account currency.
* Daily loss limit: realised today + floating on open positions.
* Two losing trades and done for the day. On the M5 bot a trade where
  only the 0.01 entry layer filled (layer 2 never did) doesn't count
  toward that limit (16 Nov 2026); the daily £ limit still sees it.
* Spread filter on new entries.

(One trade at a time = ``BotLoopConfig.max_simultaneous_setups = 1``.)

The trading day is the broker day used everywhere else in the bot: it
rolls at 17:00 New York time (22:00 UK in winter, 21:00 in summer).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from loguru import logger

from bot.risk.daily_halt import ET_TIMEZONE, DAILY_RESET_HOUR, current_trading_date

# MT5 deal entry types (mt5.DEAL_ENTRY_*); numeric so tests don't need MT5.
DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_ENTRY_INOUT = 2
DEAL_ENTRY_OUT_BY = 3

# XAUUSD fallback when MT5 can't price a loss: $1 move = $100 per lot.
XAUUSD_LOSS_PER_POINT_PER_LOT = 100.0

_SETUP_TAG = re.compile(r"s=([0-9a-f]{8})")


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = True
    risk_per_trade: float = 100.0
    """Account currency lost if the SL is hit, both layers together."""
    max_daily_loss: float = 300.0
    """Realised today + floating. At or past this, no new trades today."""
    max_losses_per_day: int = 2
    count_first_layer_only_losses: bool = True
    """False = a losing trade where only the 0.01 entry layer filled
    (layer 2 never did) doesn't count toward ``max_losses_per_day``. Its
    money still counts toward the daily £ limit. M5 bot: False (16 Nov 2026)."""
    max_loss_streak: int | None = None
    """Stop for the day after this many losing trades IN A ROW (a win
    resets it, a breakeven doesn't). None = off. M1 bot: 4 (12 Nov 2026)."""
    max_spread: float = 0.50
    """Max ask - bid (price units, $0.50) to open a new trade."""
    max_lot: float = 2.0
    """Safety cap on any single layer."""
    lot_step: float = 0.01
    risk_pct: float | None = None
    """If set (e.g. 0.10), risk this share of the current balance per
    trade instead of the fixed ``risk_per_trade`` (compounding). Used by
    the M1 bot profile."""
    max_daily_loss_pct: float | None = None
    """If set (e.g. 0.30), the daily loss limit is this share of the
    current balance instead of the fixed ``max_daily_loss``."""
    room_check: bool = False
    """True = a hard daily limit: no new trade if the day's P&L minus that
    trade's risk would go past ``-max_daily_loss`` (at -£160 with £50 at
    risk and a £200 limit, no trade). Claude's plan (19 Nov 2026)."""


def effective_risk(config: "RiskConfig", balance: float | None) -> "RiskConfig":
    """Resolve % of balance settings into £ amounts for this moment.
    Fixed-£ configs (the M5 bot) come back unchanged."""
    if balance is None or (config.risk_pct is None and config.max_daily_loss_pct is None):
        return config
    from dataclasses import replace
    b = max(float(balance), 0.0)
    return replace(
        config,
        risk_per_trade=(b * config.risk_pct) if config.risk_pct is not None else config.risk_per_trade,
        max_daily_loss=(b * config.max_daily_loss_pct)
        if config.max_daily_loss_pct is not None else config.max_daily_loss,
    )


@dataclass(frozen=True)
class LayerLots:
    layer1: float
    layer2: float
    risk: float
    """What the two layers actually risk at the SL (account currency)."""


def split_lots(
    *,
    risk: float,
    layer1_entry: float,
    layer2_entry: float,
    sl: float,
    loss_per_lot: Callable[[float, float], float],
    first_touch_lots: float = 0.01,
    lot_step: float = 0.01,
    max_lot: float = 2.0,
) -> LayerLots:
    """0.01 for layer 1, then as much layer 2 as fits in ``risk``.

    ``loss_per_lot(entry, sl)`` is the loss for 1.0 lot. Layer 2 is
    floored to the lot step, so the real risk is at or just under
    ``risk``.
    """
    l1_loss = first_touch_lots * loss_per_lot(layer1_entry, sl)
    remaining = risk - l1_loss
    per_lot_2 = loss_per_lot(layer2_entry, sl)
    l2 = 0.0
    if remaining > 0 and per_lot_2 > 0:
        l2 = _floor(remaining / per_lot_2, lot_step)
        l2 = min(l2, max_lot)
    return LayerLots(
        layer1=first_touch_lots,
        layer2=l2,
        risk=round(l1_loss + l2 * per_lot_2, 2),
    )


@dataclass(frozen=True)
class DailyRisk:
    trading_day: str
    realised: float
    floating: float
    closed_trades: int
    losing_trades: int
    winning_trades: int = 0  # net above zero; breakevens count as neither
    loss_streak: int = 0     # losing trades in a row at the end of today
    first_layer_only_losses: int = 0
    """Losing trades (included in ``losing_trades``) where only the 0.01
    entry layer filled."""
    loss_streak_full: int = 0
    """Losing trades in a row, skipping 0.01-only losses (they neither add
    to the streak nor reset it). Used when those losses don't count."""

    @property
    def total(self) -> float:
        return self.realised + self.floating


@dataclass(frozen=True)
class RiskVerdict:
    blocked: bool
    reason: str | None = None
    daily: DailyRisk | None = None


def trading_day_start(now: datetime) -> datetime:
    day = current_trading_date(now)
    start_et = datetime(day.year, day.month, day.day, DAILY_RESET_HOUR, tzinfo=ET_TIMEZONE)
    return start_et.astimezone(timezone.utc)


def summarise_day(
    deals: list[dict[str, Any]],
    open_positions: list[dict[str, Any]],
    *,
    magic: int,
    day_start: datetime,
    trading_day: str,
    known: dict[int, str] | None = None,
    first_layer_lots: float = 0.01,
) -> DailyRisk:
    """Group the bot's deals into trades (one setup = layer 1 + layer 2)
    and total today's P/L.

    ``deals`` should reach back far enough to include the opening deals
    of anything closed today. A trade counts toward today once its last
    position closes today; it's a loss if its net is below zero.
    """
    start_ts = day_start.timestamp()

    # Positions the bot opened (IN deal with our magic) → trade key.
    trade_of_pos: dict[int, str] = {}
    for d in deals:
        if d.get("entry") == DEAL_ENTRY_IN and d.get("magic") == magic:
            pos = d.get("position_id")
            m = _SETUP_TAG.search(d.get("comment") or "")
            trade_of_pos[pos] = m.group(1) if m else f"pos{pos}"
    for p in open_positions:
        if p.get("magic") == magic:
            m = _SETUP_TAG.search(p.get("comment") or "")
            trade_of_pos.setdefault(p["ticket"], m.group(1) if m else f"pos{p['ticket']}")

    # Tickets the bot registered at placement win over comment parsing
    # (some brokers rewrite comments on triggered pending orders).
    for ticket, tag in (known or {}).items():
        trade_of_pos[ticket] = tag

    open_trades = {trade_of_pos[p["ticket"]] for p in open_positions if p["ticket"] in trade_of_pos}

    # Lots each trade actually opened. A trade whose fills add up to no
    # more than the 0.01 entry layer never got layer 2. Deals without a
    # volume leave the trade unknown, so it counts as a normal trade.
    in_lots: dict[str, float | None] = {}
    for d in deals:
        if d.get("entry") != DEAL_ENTRY_IN or d.get("position_id") not in trade_of_pos:
            continue
        key = trade_of_pos[d.get("position_id")]
        vol = d.get("volume")
        prev = in_lots.get(key, 0.0)
        in_lots[key] = None if vol is None or prev is None else prev + float(vol)

    realised = 0.0
    net: dict[str, float] = {}
    last_close: dict[str, float] = {}
    for d in deals:
        if d.get("entry") not in (DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT, DEAL_ENTRY_OUT_BY):
            continue
        pos = d.get("position_id")
        if pos not in trade_of_pos:
            continue
        if float(d.get("time", 0)) < start_ts:
            continue
        pnl = (
            float(d.get("profit", 0.0)) + float(d.get("commission", 0.0))
            + float(d.get("swap", 0.0)) + float(d.get("fee", 0.0))
        )
        realised += pnl
        key = trade_of_pos[pos]
        net[key] = net.get(key, 0.0) + pnl
        last_close[key] = max(last_close.get(key, 0.0), float(d.get("time", 0)))

    closed = {k: v for k, v in net.items() if k not in open_trades}
    floating = sum(
        float(p.get("profit", 0.0)) + float(p.get("swap", 0.0))
        for p in open_positions if p["ticket"] in trade_of_pos
    )
    def _first_layer_only(k: str) -> bool:
        return in_lots.get(k) is not None and in_lots[k] <= first_layer_lots + 1e-9

    streak = 0
    streak_full = 0
    for key in sorted(closed, key=lambda k: last_close.get(k, 0.0)):
        if closed[key] < 0:
            streak += 1
            if not _first_layer_only(key):
                streak_full += 1
        elif closed[key] > 0:
            streak = 0
            streak_full = 0
    return DailyRisk(
        trading_day=trading_day,
        realised=round(realised, 2),
        floating=round(floating, 2),
        closed_trades=len(closed),
        losing_trades=sum(1 for v in closed.values() if v < 0),
        winning_trades=sum(1 for v in closed.values() if v > 0),
        loss_streak=streak,
        first_layer_only_losses=sum(1 for k, v in closed.items() if v < 0 and _first_layer_only(k)),
        loss_streak_full=streak_full,
    )


def check_risk(daily: DailyRisk, config: RiskConfig) -> RiskVerdict:
    if daily.total <= -config.max_daily_loss:
        return RiskVerdict(True, (
            f"daily loss {daily.total:.2f} at or past -{config.max_daily_loss:.0f}"
        ), daily)
    if config.room_check and daily.total - config.risk_per_trade < -config.max_daily_loss:
        return RiskVerdict(True, (
            f"daily P&L {daily.total:.2f}: no room for a {config.risk_per_trade:.0f} trade "
            f"inside -{config.max_daily_loss:.0f}"
        ), daily)
    counted = daily.losing_trades
    if not config.count_first_layer_only_losses:
        counted -= daily.first_layer_only_losses
    if counted >= config.max_losses_per_day:
        return RiskVerdict(True, (
            f"{counted} losing trades today "
            f"(max {config.max_losses_per_day})"
        ), daily)
    streak = daily.loss_streak if config.count_first_layer_only_losses else daily.loss_streak_full
    if config.max_loss_streak and streak >= config.max_loss_streak:
        return RiskVerdict(True, (
            f"{streak} losing trades in a row (max {config.max_loss_streak})"
        ), daily)
    return RiskVerdict(False, None, daily)


class DailyRiskTracker:
    """Reads MT5 history (cached a few seconds) and answers "can we open
    a new trade today?"."""

    def __init__(self, mt5: Any, *, symbol: str = "XAUUSD",
                 cache_seconds: float = 10.0, lookback_days: int = 7) -> None:
        self._mt5 = mt5
        self._symbol = symbol
        self._cache_seconds = cache_seconds
        self._lookback = timedelta(days=lookback_days)
        self._cached: tuple[float, RiskVerdict] | None = None
        self.last: RiskVerdict | None = None
        """Latest verdict (for the dashboard status)."""
        self._known: dict[int, str] = {}
        self._offset = timedelta(0)
        """Broker server clock minus real UTC. MT5 stamps deals in server
        time (MetaQuotes-Demo / Vantage run 2-3 h ahead of UTC)."""

    def register(self, ticket: int | None, setup_id: Any) -> None:
        """Remember which setup a broker ticket belongs to, so layer 1
        and layer 2 count as ONE trade."""
        if ticket is not None:
            self._known[int(ticket)] = str(setup_id)[:8]

    def invalidate(self) -> None:
        self._cached = None

    def check(self, now: datetime, config: RiskConfig) -> RiskVerdict:
        if not config.enabled:
            return RiskVerdict(False)
        mono = time.monotonic()
        if self._cached is not None and mono - self._cached[0] < self._cache_seconds:
            return self._cached[1]
        start = trading_day_start(now)
        self._update_offset(now)
        try:
            # Deal times are server time, so "up to now" in UTC would miss
            # the last few hours of trades. Ask up to a day ahead instead
            # (there are no future deals, so nothing extra comes back).
            deals = self._mt5.get_deals(
                start - self._lookback + self._offset, now + timedelta(days=1),
            )
            positions = self._mt5.get_open_positions(symbol=self._symbol)
            magic = self._mt5.magic
        except Exception:
            logger.exception("risk: MT5 history read failed; blocking new trades this tick")
            return RiskVerdict(True, "MT5 history unavailable")
        daily = summarise_day(
            deals, positions, magic=magic, day_start=start + self._offset,
            trading_day=str(current_trading_date(now)),
            known=self._known,
        )
        verdict = check_risk(daily, config)
        self._cached = (mono, verdict)
        self.last = verdict
        return verdict


    @property
    def known(self) -> dict[int, str]:
        return dict(self._known)

    @property
    def offset(self) -> timedelta:
        """Broker server clock minus UTC (last good reading)."""
        return self._offset

    def _update_offset(self, now: datetime) -> None:
        """Server clock offset from the latest tick, to the nearest hour.

        Keeps the last good value if the tick is unreadable or looks
        stale / nonsense (weekend, terminal starting up)."""
        try:
            tick_time = self._mt5.get_current_price(self._symbol)["time"]
            diff = float((tick_time - now).total_seconds())
            hours = int(round(diff / 3600))
        except Exception:
            return
        if -12 <= hours <= 14 and abs(diff - hours * 3600) <= 20 * 60:
            new = timedelta(hours=hours)
            if new != self._offset:
                logger.info(f"risk: broker server clock is UTC{hours:+d}h")
            self._offset = new
            self._offset_known = True

    def refresh_offset(self, now: datetime) -> bool:
        """Read the broker clock offset now. True once a good reading has
        been taken (it can't be read from a stale tick, e.g. at the
        weekend)."""
        self._update_offset(now)
        return getattr(self, "_offset_known", False)


def _floor(value: float, step: float) -> float:
    if value <= 0:
        return 0.0
    return round(int(value / step + 1e-9) * step, 2)
