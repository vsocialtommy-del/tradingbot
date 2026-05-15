"""Next-signal writer — pushes the bot's "next zone" view to Supabase.

PR #51: the operator wants a phone-friendly web page showing the
closest BUY and closest SELL zone the bot is currently watching. The
Next.js dashboard (``dashboard/``) reads the ``signals`` table; this
module is the writer that keeps that table fresh on every M5 close.

Algorithm
---------

For each direction (BUY / SELL):

1. Pull all currently-renderable zones from Supabase (CONFIRMED /
   ACTIVE / FLIPPED). FLIPPED zones with ``flipped_direction`` set
   to the target direction count too — they're tradeable in that
   flipped direction (PR #38).
2. Filter:
   * Recent: ``formed_at`` within the last ``max_age_days`` (default
     7) — matches the zone-snapshot CSV writer's age cutoff.
   * Reachable: zone's nearest edge is within ``max_distance_points``
     of current price (default $50). Matches zone-snapshot too.
   * Visible: skip "dead" zones whose status is CONSUMED or
     VIOLATED. The Supabase status filter already excludes these,
     but FLIPPED rows with old prior state can still survive — the
     status check is defensive.
   * Direction-correct: price is still on the right side of the
     zone to retest into it. For BUY (demand), bid > zone.top
     (price above, waiting to drop to entry). For SELL (supply),
     ask < zone.bottom (price below, waiting to rise to entry).
     If price is already inside or past the zone, the retest moment
     has passed — that zone isn't a "next signal".
3. Pick the closest qualifying zone by distance from current price
   to the L1 entry edge.
4. Compute SL (= ``zone.bottom - sl_buffer`` for BUY / ``zone.top +
   sl_buffer`` for SELL — same formula as ``compute_sl_price``) and
   the TP1 / TP2 / TP3 chain via ``find_nearest_local_peak``. The
   TP chain mirrors what ``_try_place_setup`` would do at order
   placement; the result is the trade levels the bot WOULD place
   if the retest fired right now.
5. Upsert the resulting :class:`SignalInput` via
   :meth:`SupabaseLogger.upsert_signal_for_direction`. If no zone
   qualifies, deactivate any currently-active signal in that
   direction so the dashboard renders "No active signal" instead
   of stale data.

Failure handling
----------------

This module is operator diagnostics, not part of the trading
critical path. The writer's :meth:`write` method MUST NEVER raise to
the caller — every Supabase / strategy call is wrapped, errors are
logged, and the M5 close continues. The caller in ``main.py`` adds a
second try/except as a defensive belt-and-braces guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from loguru import logger

from bot.logging.supabase_logger import SignalInput, SupabaseLogger, Zone
from bot.strategy.tp_target import find_nearest_local_peak

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NextSignalConfig:
    """Tunables for :class:`NextSignalWriter`."""

    max_distance_points: float = 50.0
    """A zone qualifies only if its nearest edge is within this many
    price points of the current bid/ask. Matches the zone-snapshot
    CSV writer (PR #49)."""

    max_age_days: int = 7
    """Zone must have ``formed_at`` within this many days. Matches
    zone-snapshot."""

    sl_buffer_points: float = 17.5
    """Buffer past the zone bound for the SL. Matches the strong-point
    module's default (``StrongPointConfig.sl_buffer_points``)."""

    tp_lookback_bars: int = 50
    """Lookback for the ``find_nearest_local_peak`` chain. Matches the
    pipeline's default (``StrategyPipelineConfig.tp1_local_peak_lookback_bars``).
    Kept as a NextSignal-local field so this module can be tuned
    independently for the dashboard view if needed."""


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


class NextSignalWriter:
    """Compute + persist the next-signal snapshot each M5 close.

    The writer is stateless: every call recomputes the BUY + SELL
    snapshots from scratch. There's no cache to invalidate, no
    membership set to keep coherent — Supabase is the source of
    truth and the M5 cadence is slow enough that the few hundred
    µs of redundant work is irrelevant.
    """

    def __init__(
        self,
        supabase: SupabaseLogger,
        *,
        config: NextSignalConfig | None = None,
    ) -> None:
        self._supabase = supabase
        self._cfg = config or NextSignalConfig()

    def write(
        self,
        df: "pd.DataFrame",
        *,
        bid: float,
        ask: float,
        now: datetime | None = None,
    ) -> dict[str, str | None]:
        """Compute and upsert the BUY + SELL signals.

        Returns a small dict describing what happened per direction,
        for logging / observability:

            {"BUY": "wrote", "SELL": "deactivated"}
            {"BUY": "error", "SELL": "wrote"}

        Values: ``"wrote"`` (new signal upserted), ``"deactivated"``
        (no qualifying zone, any active signal marked inactive),
        ``"error"`` (Supabase / strategy call raised — logged, the
        other direction is still processed).
        """
        if now is None:
            now = datetime.now(tz=timezone.utc)
        if len(df) == 0:
            logger.debug("next-signal: empty df, skipping")
            return {"BUY": "skipped", "SELL": "skipped"}

        try:
            zones = self._supabase.get_zones_by_status(
                ["CONFIRMED", "ACTIVE", "FLIPPED"],
            )
        except Exception:
            logger.exception(
                "next-signal: get_zones_by_status failed; dashboard "
                "will be stale until the next M5 close"
            )
            return {"BUY": "error", "SELL": "error"}

        return {
            "BUY": self._process_direction("BUY", zones, df, bid, ask, now),
            "SELL": self._process_direction("SELL", zones, df, bid, ask, now),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _process_direction(
        self,
        direction: str,
        zones: list[Zone],
        df: "pd.DataFrame",
        bid: float,
        ask: float,
        now: datetime,
    ) -> str:
        try:
            picked = _pick_closest_zone(
                direction=direction,
                zones=zones,
                bid=bid,
                ask=ask,
                now=now,
                max_distance_points=self._cfg.max_distance_points,
                max_age_days=self._cfg.max_age_days,
            )
        except Exception:
            logger.exception(
                f"next-signal: _pick_closest_zone raised for {direction}"
            )
            return "error"

        if picked is None:
            try:
                self._supabase.deactivate_signals_for_direction(
                    direction,  # type: ignore[arg-type]
                )
            except Exception:
                logger.exception(
                    f"next-signal: deactivate_signals_for_direction "
                    f"failed for {direction}"
                )
                return "error"
            return "deactivated"

        try:
            signal_input = self._build_signal(
                direction=direction, zone=picked,
                df=df, bid=bid, ask=ask,
            )
        except Exception:
            logger.exception(
                f"next-signal: _build_signal raised for {direction} "
                f"zone {picked.id}"
            )
            return "error"

        try:
            self._supabase.upsert_signal_for_direction(signal_input)
        except Exception:
            logger.exception(
                f"next-signal: upsert_signal_for_direction failed "
                f"for {direction} zone {picked.id}"
            )
            return "error"
        return "wrote"

    def _build_signal(
        self,
        *,
        direction: str,
        zone: Zone,
        df: "pd.DataFrame",
        bid: float,
        ask: float,
    ) -> SignalInput:
        """Compute trade levels for a zone + assemble the insert payload."""
        zone_top = float(zone.top)
        zone_bottom = float(zone.bottom)
        # L1 entry is the near edge of the zone — top for BUY (price
        # drops INTO the zone from above and touches the top first),
        # bottom for SELL (mirror).
        if direction == "BUY":
            entry = zone_top
            sl = zone_bottom - self._cfg.sl_buffer_points
            current_price = bid
        else:
            entry = zone_bottom
            sl = zone_top + self._cfg.sl_buffer_points
            current_price = ask

        tp1, tp2, tp3 = _compute_tp_chain(
            df=df, entry=entry, direction=direction,
            lookback=self._cfg.tp_lookback_bars,
        )

        # Distance is signed from current_price to entry, in the
        # direction the trade will travel. For a pending BUY retest
        # (price above the zone, waiting to drop to entry):
        # current_price > entry → distance = current_price - entry > 0.
        # For SELL (price below, waiting to rise to entry):
        # current_price < entry → distance = entry - current_price > 0.
        distance = (
            current_price - entry if direction == "BUY"
            else entry - current_price
        )

        return SignalInput(
            direction=direction,  # type: ignore[arg-type]
            zone_id=zone.id,
            zone_top=Decimal(f"{zone_top:.2f}"),
            zone_bottom=Decimal(f"{zone_bottom:.2f}"),
            entry_price=Decimal(f"{entry:.2f}"),
            sl_price=Decimal(f"{sl:.2f}"),
            tp1_price=Decimal(f"{tp1:.2f}") if tp1 is not None else None,
            tp2_price=Decimal(f"{tp2:.2f}") if tp2 is not None else None,
            tp3_price=Decimal(f"{tp3:.2f}") if tp3 is not None else None,
            pattern_type=zone.pattern_type,
            zone_status=zone.status,
            current_price=Decimal(f"{current_price:.2f}"),
            distance_dollars=Decimal(f"{distance:.2f}"),
            is_active=True,
        )


# --------------------------------------------------------------------------- #
# Module-level helpers (testable without a NextSignalWriter instance)
# --------------------------------------------------------------------------- #


def _zone_matches_direction(zone: Zone, direction: str) -> bool:
    """True iff ``zone`` would be traded in ``direction``.

    For CONFIRMED / ACTIVE: ``zone.direction == direction``.
    For FLIPPED: ``zone.flipped_direction == direction`` (the original
    direction was the opposite — the violation flipped it).
    """
    if zone.status == "FLIPPED":
        return zone.flipped_direction == direction
    return zone.direction == direction


def _zone_is_pending_retest(zone: Zone, direction: str, bid: float, ask: float) -> bool:
    """True iff price is still on the right side of the zone to retest.

    The user wants the "next signal the bot is waiting to fire" —
    that means the retest hasn't happened yet. For BUY (demand)
    that's bid above zone.top (price has to drop to enter). For
    SELL (supply) that's ask below zone.bottom (price has to rise
    to enter).

    Once price is inside or past the zone, the retest moment has
    arrived or gone — those zones aren't "next signals" any more.
    """
    if direction == "BUY":
        return bid > float(zone.top)
    return ask < float(zone.bottom)


def _pick_closest_zone(
    *,
    direction: str,
    zones: list[Zone],
    bid: float,
    ask: float,
    now: datetime,
    max_distance_points: float,
    max_age_days: int,
) -> Zone | None:
    """Filter + pick the nearest qualifying zone in one pass.

    Pure function. The only branchy logic is the BUY-vs-SELL
    asymmetry (which price to compare to which zone edge).
    """
    current_price = bid if direction == "BUY" else ask
    cutoff = now - timedelta(days=max_age_days)
    candidates: list[tuple[float, Zone]] = []

    for z in zones:
        # Status filter — defensive; supabase query already excludes
        # CONSUMED / VIOLATED but be paranoid in case of races.
        if z.status not in ("CONFIRMED", "ACTIVE", "FLIPPED"):
            continue
        # Direction match (handles FLIPPED via flipped_direction).
        if not _zone_matches_direction(z, direction):
            continue
        # Pending-retest gate — price still on the right side.
        if not _zone_is_pending_retest(z, direction, bid, ask):
            continue
        # Age filter.
        if z.formed_at < cutoff:
            continue
        # Distance filter — nearest edge within range.
        top = float(z.top)
        bottom = float(z.bottom)
        nearest_edge = (
            bottom if current_price > top
            else top if current_price < bottom
            else current_price
        )
        distance = abs(nearest_edge - current_price)
        if distance > max_distance_points:
            continue
        # Distance from current price to L1 entry (= the zone edge
        # price will retest first). This is what we sort on.
        entry_edge = top if direction == "BUY" else bottom
        sort_distance = abs(current_price - entry_edge)
        candidates.append((sort_distance, z))

    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[0][1]


def _compute_tp_chain(
    *,
    df: "pd.DataFrame",
    entry: float,
    direction: str,
    lookback: int,
) -> tuple[float | None, float | None, float | None]:
    """Run the TP1 → TP2 → TP3 chain off the entry price.

    Identical to what ``_try_place_setup`` does at order placement
    (PR #41). TP2 anchors on TP1, TP3 on TP2; if any link in the
    chain returns ``None`` the rest of the chain is also ``None``.
    """
    tp1 = find_nearest_local_peak(
        df, entry_price=entry,
        direction=direction,  # type: ignore[arg-type]
        lookback_bars=lookback,
    )
    if tp1 is None:
        return None, None, None
    tp2 = find_nearest_local_peak(
        df, entry_price=tp1,
        direction=direction,  # type: ignore[arg-type]
        lookback_bars=lookback,
    )
    if tp2 is None:
        return tp1, None, None
    tp3 = find_nearest_local_peak(
        df, entry_price=tp2,
        direction=direction,  # type: ignore[arg-type]
        lookback_bars=lookback,
    )
    return tp1, tp2, tp3
