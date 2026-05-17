"""Next-signal writer — pushes the bot's "next zone" view to Supabase.

PR #51: the operator wants a phone-friendly web page showing the
closest BUY and closest SELL zone the bot is currently watching. The
Next.js dashboard (``dashboard/``) reads the ``signals`` table; this
module is the writer that keeps that table fresh on every M5 close.

Algorithm
---------

For each direction (BUY / SELL):

1. Pull currently-alive, untraded zones from Supabase: status in
   ``("CONFIRMED", "FLIPPED")``.

   * ``ACTIVE`` is deliberately excluded — those zones already have
     a setup with at least one filled layer on them. Showing them
     as "next signal" would be misleading: the bot can't (and
     won't) place another setup on a zone it's already trading
     (dedup rule).
   * ``CONSUMED`` and ``VIOLATED`` are dead by their lifecycle
     status — excluded automatically by the status filter.

2. Filter further:
   * **Recent**: ``formed_at`` within the last ``max_age_days``
     (default 7) — matches the zone-snapshot CSV writer.
   * **Reachable**: zone's nearest edge is within
     ``max_distance_points`` of current price (default $100).
   * **Direction-correct**: price still on the right side of the
     zone to retest into it. For BUY (demand), bid > zone.top.
     For SELL (supply), ask < zone.bottom. Zones price is
     already inside or past aren't "next signals".
   * **Flip premise intact (FLIPPED zones only)**: skip zones
     whose flip has been body-broken by a post-flip bar. Mirrors
     the bot's own ``_load_flipped_candidates`` rule — without
     this check, the dashboard would advertise FLIPPED zones the
     bot has correctly decided are dead. Uses the current df, so
     the dashboard always reflects the *live* tradeability state,
     not whatever was true at flip time.
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
from bot.strategy.zone_lifecycle import flipped_zone_body_broken_since_flip

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NextSignalConfig:
    """Tunables for :class:`NextSignalWriter`."""

    max_distance_points: float = 100.0
    """A zone qualifies only if its nearest edge is within this many
    price points of the current bid/ask.

    Bumped from 50 to 100 in PR #54 — at 50 the dashboard regularly
    showed "No active signal" because no qualifying zone existed
    within that tight window (especially in directional markets where
    fresh zones lag $50+ behind the move). 100 surfaces the next
    reasonable opportunity without cluttering the view with truly
    distant zones. Doesn't affect any trading behaviour; this filter
    is dashboard-only."""

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
            # ACTIVE is excluded: those zones already have a filled
            # setup on them (dedup blocks placement on the same zone
            # again), so showing them as a "next signal" is misleading.
            # CONSUMED / VIOLATED are also out — dead by status.
            zones = self._supabase.get_zones_by_status(
                ["CONFIRMED", "FLIPPED"],
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
                df=df,
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
    df: "pd.DataFrame",
    max_distance_points: float,
    max_age_days: int,
) -> Zone | None:
    """Filter + pick the nearest qualifying ALIVE zone.

    "Alive" means:
    * Status is ``CONFIRMED`` or ``FLIPPED`` (``ACTIVE`` already has a
      filled setup on it — bot can't trade it again, so showing it as a
      "next signal" is misleading).
    * For FLIPPED zones, the flip premise hasn't been body-broken by
      any post-flip bar in the current df. Mirrors the same check the
      bot uses in ``_load_flipped_candidates`` so the dashboard never
      advertises a zone the bot won't actually trade.

    Pure function. The only branchy logic is the BUY-vs-SELL
    asymmetry (which price to compare to which zone edge).
    """
    import pandas as pd  # noqa: PLC0415 — keep TYPE_CHECKING import light

    current_price = bid if direction == "BUY" else ask
    cutoff = now - timedelta(days=max_age_days)
    candidates: list[tuple[float, Zone]] = []

    for z in zones:
        # Status filter — defensive; supabase query already excludes
        # CONSUMED / VIOLATED / ACTIVE but be paranoid in case of races.
        if z.status not in ("CONFIRMED", "FLIPPED"):
            continue
        # Direction match (handles FLIPPED via flipped_direction).
        if not _zone_matches_direction(z, direction):
            continue
        # FLIPPED flip-premise check: skip zones whose flip has been
        # invalidated by a body-close past the wrong side since the
        # flip. The bot's placement queue applies the same check, so
        # mirroring it here keeps the dashboard truthful.
        if z.status == "FLIPPED":
            if z.flipped_at is None or z.flipped_direction is None:
                # DB CHECK should prevent this; defensive skip.
                continue
            if flipped_zone_body_broken_since_flip(
                zone_top=float(z.top),
                zone_bottom=float(z.bottom),
                flipped_direction=z.flipped_direction,
                flipped_at=pd.Timestamp(z.flipped_at),
                df=df,
            ):
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
