"""Initial SL placement and broker-side modification (spec Section 5).

Three responsibilities, kept as separate methods so each can be tested
and called independently:

1. :meth:`SLManager.calculate_initial_sl` — turn an OHLC window into an
   SL price using the lowest swing low (BUY) / highest swing high
   (SELL) within the configured lookback, then apply the anti-stop-hunt
   buffer.

2. :meth:`SLManager.validate_sl_distance` — sanity-check distance from
   entry. Independent of calculation so the caller can validate any SL
   (computed, manual override, BE level).

3. :meth:`SLManager.apply_sl_to_setup` — modify SL on every open
   position belonging to a setup. Used both at Layer-1 placement
   (initial SL) and by ``entry_trigger`` when each new layer fires
   (apply the same setup-wide SL). The TP1 → BE move lives in
   ``tp1_manager`` and is intentionally NOT routed through here, so
   the BE-failure semantics there don't leak in.

Lifecycle (spec 5.2)::

    Trade opens   → SL at structural-low - buffer (computed here)
    TP1 hits      → SL → BE on remaining 50%   (tp1_manager, not here)
    Runner phase  → manual; bot does not auto-trail (spec 5.2)

Design decisions called out in the PR
-------------------------------------

1. **Multi-position modify is fail-soft.** ``apply_sl_to_setup`` tries
   every open position and aggregates errors. Returns False if any
   modify failed (so the caller can retry / alert) but never aborts
   early. Mirrors ``tp1_manager._move_sl_to_be``. Roll-back was
   considered and rejected — reverting one modify_order's success
   needs the old SL plus another modify_order call that can also
   fail; we'd just compound the problem.

2. **No-swings-found falls back to bar low / high extreme**, not closes
   and not error. Closes ignore wicks (a wick below the close range
   could hit a too-close SL); erroring loses tradeable setups when
   structural detection's strength threshold doesn't trigger on a
   quiet section. Bar extremes are what actually hit SL, the buffer
   still applies, and the result flags ``fallback_used=True`` so the
   caller can log a warning.

3. **Validation is decoupled from calculation.** ``calculate_initial_sl``
   doesn't know the entry price (and shouldn't — entry comes from the
   layer fill). Callers compose: calc → validate → apply.

4. **Structural reference is the swing's price (close-based), not the
   bar's low/high.** ``detect_swings`` works on closes; that's the
   level the structure module agrees is "the swing." The buffer is
   what protects against wicks below it. Same model spec Section 5.1
   describes ("below the recent lower low + buffer").

5. **No state on the manager.** SL doesn't get computed once and
   cached — every call is fresh. The caller (main loop) is responsible
   for invocation cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import pandas as pd
from loguru import logger

from bot.execution.mt5_connector import MT5Connector
from bot.logging.supabase_logger import Setup, SupabaseLogger, Trade
from bot.strategy.imbalance import ImbalanceZone
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.structure import (
    Swing,
    detect_swings,
    get_swings_within_lookback,
)


# --------------------------------------------------------------------------- #
# Result + config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SLCalculation:
    """Output of :meth:`SLManager.calculate_initial_sl`.

    ``reference_swing_price`` is the structural level used (a swing
    price when one was found, or a bar low/high in the fallback
    branch — see ``fallback_used``).
    """

    sl_price: float
    reference_swing_price: float
    buffer_used: float
    lookback_used: int
    direction: str  # "BUY" or "SELL"
    fallback_used: bool = False
    """True iff no swings were found in the lookback window and the
    reference came from a bar low / high extreme instead."""


@dataclass(frozen=True)
class SLValidation:
    """Output of :meth:`SLManager.validate_sl_distance`."""

    is_valid: bool
    distance_points: float
    is_too_close: bool
    is_too_far: bool
    error: str | None = None


@dataclass(frozen=True)
class SLManagerConfig:
    symbol: str = "XAUUSD"
    swing_strength: int = 3
    """Bars on each side for ``detect_swings``. Mirrors
    :attr:`StructureConfig.swing_strength`."""
    recent_swing_lookback: int = 20
    """Number of recent candles to scan for the structural reference
    swing. Default per spec Section 5.1; ``bot_config`` value
    overrides via the orchestrator."""
    sl_buffer_points: float = 17.5
    """Anti-stop-hunt buffer in price units. Default $17.5 = midpoint
    of spec's 15-20 point band."""
    min_sl_distance_points: float = 5.0
    """SL closer than this is rejected — slippage / spread would
    routinely stop us out before the trade breathes."""
    max_sl_distance_points: float = 200.0
    """SL farther than this is flagged as suspicious; the caller's
    policy (skip / clip / warn) decides what to do."""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class SLManager:
    """Compute, validate, and apply SLs across a setup's positions."""

    def __init__(
        self,
        mt5: MT5Connector,
        supabase: SupabaseLogger,
        config: SLManagerConfig | None = None,
    ) -> None:
        self._mt5 = mt5
        self._supabase = supabase
        self._config = config or SLManagerConfig()

    # ----------------------------------------------------------------- #
    # Calculate
    # ----------------------------------------------------------------- #

    def calculate_initial_sl(
        self,
        zone: ValidatedZone | ImbalanceZone,
        ohlc_df: pd.DataFrame,
    ) -> SLCalculation:
        """Compute the initial SL for a setup on ``zone``.

        BUY: ``min(swing_low.price for swing_low in last N candles) - buffer``
        SELL: ``max(swing_high.price for swing_high in last N candles) + buffer``

        When no matching-kind swings exist in the lookback window the
        reference falls back to ``min(low)`` (BUY) or ``max(high)``
        (SELL) over the same window — see design decision #2 in the
        module docstring. ``fallback_used=True`` flags this in the
        result so the caller can log appropriately.
        """
        for col in ("low", "high", "close"):
            if col not in ohlc_df.columns:
                raise ValueError(f"ohlc_df must have a '{col}' column")

        cfg = self._config
        n = len(ohlc_df)
        lookback = min(cfg.recent_swing_lookback, n)
        if lookback < 1:
            raise ValueError(
                "ohlc_df is empty; cannot compute SL without bars"
            )
        window = ohlc_df.iloc[-lookback:]

        # Swing-based primary path.
        swings = detect_swings(ohlc_df, cfg.swing_strength)
        last_index = n - 1
        recent_swings = get_swings_within_lookback(
            swings, from_bar=last_index, lookback=lookback,
        )

        if zone.direction == "BUY":
            ref_price, fallback = _resolve_buy_reference(recent_swings, window)
            sl_price = ref_price - cfg.sl_buffer_points
        else:
            ref_price, fallback = _resolve_sell_reference(recent_swings, window)
            sl_price = ref_price + cfg.sl_buffer_points

        if fallback:
            logger.warning(
                f"sl_manager: no {zone.direction} swing in last "
                f"{lookback} bars; falling back to bar extreme "
                f"({ref_price})"
            )

        return SLCalculation(
            sl_price=sl_price,
            reference_swing_price=ref_price,
            buffer_used=cfg.sl_buffer_points,
            lookback_used=lookback,
            direction=zone.direction,
            fallback_used=fallback,
        )

    # ----------------------------------------------------------------- #
    # Validate
    # ----------------------------------------------------------------- #

    def validate_sl_distance(
        self,
        entry_price: float,
        sl_price: float,
        direction: str,
    ) -> SLValidation:
        """Sanity-check SL distance and side. Pure function on inputs."""
        if direction not in ("BUY", "SELL"):
            return SLValidation(
                is_valid=False,
                distance_points=0.0,
                is_too_close=False,
                is_too_far=False,
                error=f"unknown direction: {direction!r}",
            )

        # Side check first — a wrong-side SL has no meaningful "distance".
        if direction == "BUY" and sl_price >= entry_price:
            return SLValidation(
                is_valid=False,
                distance_points=abs(entry_price - sl_price),
                is_too_close=False,
                is_too_far=False,
                error=(
                    f"BUY SL ({sl_price}) must be below entry "
                    f"({entry_price})"
                ),
            )
        if direction == "SELL" and sl_price <= entry_price:
            return SLValidation(
                is_valid=False,
                distance_points=abs(sl_price - entry_price),
                is_too_close=False,
                is_too_far=False,
                error=(
                    f"SELL SL ({sl_price}) must be above entry "
                    f"({entry_price})"
                ),
            )

        distance = abs(entry_price - sl_price)
        cfg = self._config
        # Boundary inclusive on both ends — spec Section 5.1 doesn't
        # specify, and inclusive matches the rest of the codebase
        # (entry triggers, TP1 trigger).
        is_too_close = distance < cfg.min_sl_distance_points
        is_too_far = distance > cfg.max_sl_distance_points
        is_valid = not (is_too_close or is_too_far)

        error: str | None = None
        if is_too_close:
            error = (
                f"SL distance {distance} below minimum "
                f"{cfg.min_sl_distance_points}"
            )
        elif is_too_far:
            error = (
                f"SL distance {distance} above maximum "
                f"{cfg.max_sl_distance_points}"
            )

        return SLValidation(
            is_valid=is_valid,
            distance_points=distance,
            is_too_close=is_too_close,
            is_too_far=is_too_far,
            error=error,
        )

    # ----------------------------------------------------------------- #
    # Apply
    # ----------------------------------------------------------------- #

    def apply_sl_to_setup(
        self, setup: Setup, sl_price: float,
    ) -> bool:
        """Modify SL on every open position belonging to ``setup``.

        Returns True if all open positions had their SL updated, False
        if any modify failed. WAITING / terminal trades are skipped:
        WAITING has no broker position yet, terminal is gone.

        Setups with zero open positions return True (no-op): the apply
        is satisfied trivially.
        """
        trades = self._supabase.get_trades_for_setup(setup.id)
        open_trades = [
            t for t in trades
            if t.status in ("FILLED", "PARTIALLY_CLOSED")
            and t.mt5_ticket is not None
        ]
        if not open_trades:
            logger.debug(
                f"sl_manager: setup {setup.id} has no open positions; "
                f"apply_sl_to_setup is a no-op"
            )
            return True

        all_ok = True
        for trade in open_trades:
            assert trade.mt5_ticket is not None  # filtered above
            ticket = trade.mt5_ticket
            try:
                self._mt5.modify_order(ticket, sl=sl_price)
            except Exception as e:
                msg = (
                    f"modify_order SL failed for setup={setup.id} "
                    f"layer={trade.layer_number} ticket={ticket}: {e}"
                )
                logger.exception(f"sl_manager: CRITICAL {msg}")
                self._safe_log_event(
                    "ERROR",
                    f"CRITICAL: SL modify failed (sl_manager.apply_sl_to_setup)",
                    context={
                        "setup_id": str(setup.id),
                        "trade_id": str(trade.id),
                        "ticket": ticket,
                        "attempted_sl": sl_price,
                        "exception": str(e),
                    },
                    setup_id=setup.id,
                    trade_id=trade.id,
                )
                all_ok = False
                continue

            # Sync the trade row's sl_price field. Best-effort: if it
            # fails, the broker is the source of truth and reconciliation
            # will catch up.
            try:
                self._supabase.update_trade(
                    trade.id, sl_price=sl_price,
                )
            except Exception as e:
                logger.exception(
                    f"sl_manager: trade-row sl_price sync failed for "
                    f"trade {trade.id}: {e}"
                )

            self._safe_log_event(
                "INFO",
                f"SL applied to layer {trade.layer_number}",
                context={
                    "setup_id": str(setup.id),
                    "trade_id": str(trade.id),
                    "ticket": ticket,
                    "sl_price": sl_price,
                },
                setup_id=setup.id,
                trade_id=trade.id,
            )

        return all_ok

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    def _safe_log_event(
        self,
        level: str,
        message: str,
        *,
        context: dict[str, Any],
        setup_id: UUID | None = None,
        trade_id: UUID | None = None,
    ) -> None:
        """Best-effort bot_logs write — never raises."""
        try:
            self._supabase.log_event(
                level=level,  # type: ignore[arg-type]
                message=message,
                context=context,
                setup_id=setup_id,
                trade_id=trade_id,
            )
        except Exception:
            logger.exception("sl_manager: log_event failed (non-fatal)")


# --------------------------------------------------------------------------- #
# Module-level helpers (testable without an SLManager instance)
# --------------------------------------------------------------------------- #


def _resolve_buy_reference(
    recent_swings: list[Swing],
    window: pd.DataFrame,
) -> tuple[float, bool]:
    """Pick the BUY-side structural reference price.

    Returns ``(price, fallback_used)``. Primary path: the lowest
    swing-low in the window (multiple swings → take the minimum;
    equal-priced swings collapse trivially). Fallback: the lowest bar
    low across the same window.
    """
    lows = [s.price for s in recent_swings if s.kind == "LOW"]
    if lows:
        return min(lows), False
    return float(window["low"].min()), True


def _resolve_sell_reference(
    recent_swings: list[Swing],
    window: pd.DataFrame,
) -> tuple[float, bool]:
    """Pick the SELL-side structural reference price.

    Returns ``(price, fallback_used)``. Primary path: the highest
    swing-high in the window. Fallback: the highest bar high in the
    same window.
    """
    highs = [s.price for s in recent_swings if s.kind == "HIGH"]
    if highs:
        return max(highs), False
    return float(window["high"].max()), True
