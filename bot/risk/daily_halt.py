"""Daily-loss halt — blocks new entries if drawdown reaches the configured cap.

Per spec Section 7:

* At day start (17:01 EST), record ``starting_balance``.
* After each closed trade, check if
  ``current_balance < starting_balance * (1 - daily_loss_limit_pct/100)``.
* If true: cancel pending orders, refuse new setups, allow open
  positions to run their course. Resume next trading day.

This module is **pure logic**. The orchestrator passes
``starting_balance`` (read from ``daily_pnl.starting_balance``) and
``current_balance`` (read from MT5); we return the verdict plus a
``resume_at`` timestamp so the dashboard can show "halted until …".

Design decisions
----------------

1. **Balance, not equity.**
   Daily P&L is computed from *realised* balance — the trade-exit
   amount. Unrealised drawdown on open positions might still recover.
   The halt should trigger on locked-in losses, not paper losses. This
   matches the spec ("after each closed trade") and keeps the runner
   trade alive even if its current floating P&L is in the red.

2. **17:00 ET (America/New_York), not UTC.**
   Spec says "17:00 EST" — that's the broker rollover. We use
   ``zoneinfo.ZoneInfo("America/New_York")`` so DST is handled
   automatically (17:00 ET shifts between 22:00 UTC in winter and
   21:00 UTC in summer, transparently). The orchestrator can pass
   times in any timezone; we convert internally.

3. **From starting, not from peak.**
   The daily halt's threshold is computed against
   ``starting_balance`` (the day's open). Spec Section 15.5 separately
   mentions a "10% drawdown from peak" review trigger — that's a
   different concept and lives in a different module.

4. **Soft halt.**
   ``is_halted=True`` is advisory. The orchestrator decides what to
   do — typically: block new setup activation, but keep
   ``tp1_manager`` / ``sl_manager`` running on already-open positions.
   This module doesn't enforce; it only computes.

5. **Boundary semantics: inclusive at the halt threshold.**
   ``drawdown_pct <= -daily_loss_limit_pct`` triggers the halt. So
   exactly -10% halts; -9.999% does not.

6. **Trading-day boundary at 17:00 ET, inclusive at start.**
   ``current_trading_date(2026-05-06 17:00 ET) == 2026-05-06``.
   ``current_trading_date(2026-05-06 16:59 ET) == 2026-05-05``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from loguru import logger

# Broker rollover timezone. ``America/New_York`` follows DST
# automatically — "17:00 ET" is 22:00 UTC in winter, 21:00 UTC in
# summer.
ET_TIMEZONE = ZoneInfo("America/New_York")

# 17:00 ET — spec Section 7's daily reset time.
DAILY_RESET_HOUR = 17


@dataclass(frozen=True)
class DailyHaltConfig:
    """Tunable; default matches the seeded ``bot_config`` value."""

    daily_loss_limit_pct: float = 10.0


@dataclass(frozen=True)
class DailyHaltResult:
    """Verdict from :func:`check_daily_halt`."""

    is_halted: bool
    current_drawdown_pct: float  # negative if losing, positive if winning
    threshold_pct: float  # the configured limit (positive number)
    starting_balance: float
    current_balance: float
    reason: str | None = None
    resume_at: datetime | None = None  # next 17:00 ET if halted


def check_daily_halt(
    *,
    starting_balance: float,
    current_balance: float,
    config: DailyHaltConfig | None = None,
    now: datetime | None = None,
) -> DailyHaltResult:
    """Compute the daily-halt verdict for a given balance pair.

    ``now`` is only used for ``resume_at`` and defaults to real wall
    time. Tests should pass an explicit ``now`` to avoid flakes.
    """
    cfg = config or DailyHaltConfig()

    if starting_balance <= 0:
        raise ValueError(
            f"starting_balance must be positive, got {starting_balance}"
        )

    drawdown_pct = (current_balance - starting_balance) / starting_balance * 100.0
    is_halted = drawdown_pct <= -cfg.daily_loss_limit_pct

    reason: str | None = None
    resume_at: datetime | None = None

    if is_halted:
        reason = "DAILY_LOSS_LIMIT_REACHED"
        effective_now = now if now is not None else datetime.now(tz=ET_TIMEZONE)
        resume_at = next_daily_reset(effective_now)
        logger.warning(
            f"DAILY HALT triggered: drawdown={drawdown_pct:.4f}% "
            f"(threshold -{cfg.daily_loss_limit_pct}%), "
            f"starting=${starting_balance:.2f}, current=${current_balance:.2f}, "
            f"resume_at={resume_at.isoformat()}"
        )

    return DailyHaltResult(
        is_halted=is_halted,
        current_drawdown_pct=drawdown_pct,
        threshold_pct=cfg.daily_loss_limit_pct,
        starting_balance=starting_balance,
        current_balance=current_balance,
        reason=reason,
        resume_at=resume_at,
    )


def current_trading_date(now: datetime) -> date:
    """Return the calendar date of the broker trading day containing ``now``.

    A trading day **starts** at 17:00 ET and is named for that start
    date. So:

    * 2026-05-06 16:59 ET → trading day 2026-05-05 (still in yesterday's).
    * 2026-05-06 17:00 ET → trading day 2026-05-06 (just rolled).
    * 2026-05-06 23:00 ET → trading day 2026-05-06.
    * 2026-05-07 12:00 ET → trading day 2026-05-06 (before today's reset).

    This matches ``daily_pnl.trading_date`` semantics from migration 001.

    Raises
    ------
    ValueError
        ``now`` is naive (no timezone). Always pass timezone-aware times.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be timezone-aware, got naive {now}")
    et = now.astimezone(ET_TIMEZONE)
    if et.hour < DAILY_RESET_HOUR:
        # Before today's 17:00 reset → still in yesterday's trading day.
        return (et - timedelta(days=1)).date()
    return et.date()


def next_daily_reset(now: datetime) -> datetime:
    """Return the next 17:00 ET timestamp strictly after ``now``.

    Used to populate ``resume_at`` when the halt fires. If ``now`` is
    exactly at 17:00 ET, that boundary is considered already crossed —
    we return the *next* day's 17:00.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be timezone-aware, got naive {now}")
    et = now.astimezone(ET_TIMEZONE)
    today_reset = et.replace(
        hour=DAILY_RESET_HOUR, minute=0, second=0, microsecond=0
    )
    if et < today_reset:
        return today_reset
    return today_reset + timedelta(days=1)


def is_halt_expired(halted_at: datetime, now: datetime) -> bool:
    """True iff ``now`` is in a later trading day than ``halted_at``.

    The orchestrator uses this to decide whether a previously-recorded
    halt should be lifted (i.e., the daily reset has happened since
    the halt fired).
    """
    return current_trading_date(now) > current_trading_date(halted_at)
