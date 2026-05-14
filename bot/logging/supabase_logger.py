"""Supabase write layer.

Persists trades, setups, zones, daily PnL, and structured bot logs to
the Supabase tables defined in spec Section 9.2 (and migrations/
001_initial_schema.sql). The dashboard (Phase E) reads these via
Supabase realtime subscriptions, so every write here also drives the UI.

Design notes
------------
* Uses the **service_role** key (bypasses RLS). Phase A migration enables
  RLS with no policies — service_role is the only role that can read or
  write until Phase E adds anon policies.
* Pydantic models validate inputs at the boundary so callers can't slip
  malformed data into Postgres. Each model mirrors the table's CHECK
  constraints (e.g. ``direction`` is ``Literal["BUY", "SELL"]``).
* Prices use ``Decimal`` for exactness; ``model_dump(mode="json")``
  serialises Decimal to string, which Postgres NUMERIC accepts.
* The class is a thin wrapper — no caching, no retry, no async. If the
  network blips, calls raise; the caller (bot main loop) decides how to
  recover.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ``supabase-py`` is a runtime dep of the live bot but is NOT needed by
# the backtest framework (which writes nothing to Supabase). Guarding
# the import keeps ``bot.backtest`` usable on Linux/Colab/CI hosts that
# don't have it installed. ``Client`` is annotation-only thanks to the
# ``from __future__ import annotations`` above; ``create_client`` is
# called lazily by :meth:`SupabaseLogger.__init__` only when the live
# bot actually instantiates a logger.
if TYPE_CHECKING:
    from supabase import Client


def _get_create_client() -> Any:
    """Lazy ``from supabase import create_client``.

    Same pattern as :class:`bot.execution.mt5_connector._LazyMT5`: any
    transitive importer (sl_manager, tp1_manager, news_filter, …) can
    pull this module in without supabase-py installed; only
    constructing a real :class:`SupabaseLogger` triggers the import.
    """
    try:
        from supabase import create_client  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "supabase-py is required for SupabaseLogger but is not "
            "installed. If you only need the backtest framework, use "
            "``bot.backtest`` — it does not depend on supabase-py."
        ) from e
    return create_client


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

Direction = Literal["BUY", "SELL"]
ZoneType = Literal["STRONG_POINT", "IMBALANCE"]
ZoneStatus = Literal[
    "CONFIRMED", "ACTIVE", "CONSUMED", "VIOLATED", "FLIPPED",
]
"""See ``migrations/007_zone_lifecycle.sql`` for the state machine. The
in-memory pre-states (FRESH / TRADEABLE) never reach the DB and are
deliberately absent from this literal."""
PatternType = Literal[
    # Legacy W/M era (preserved for back-compat on existing rows).
    "W", "M", "N",
    # S&D codes (PR #31 onward; see migrations/006_pattern_type_snd_codes.sql).
    "RBR", "DBD", "DBR", "RBD",
]
EntryMode = Literal["STRONG_POINT_FIRST_TOUCH", "IMBALANCE_FIRST_TOUCH"]
SetupStatus = Literal[
    "PENDING", "ACTIVE", "TP1_HIT", "CLOSED", "SKIPPED", "STOPPED_OUT"
]
TradeStatus = Literal[
    "WAITING", "FILLED", "PARTIALLY_CLOSED", "CLOSED", "CANCELLED"
]
OrderType = Literal["MARKET", "LIMIT"]
CloseReason = Literal[
    # Per-layer TPs (PR #41; see migration 009).
    "TP1", "TP2", "TP3",
    # Stop / external closes.
    "SL_HIT", "BE_HIT", "MANUAL_CLOSE", "NEWS_CLOSE",
    # PR #43 / migration 011: a WAITING layer cancelled because a
    # previous layer's TP fired (cascade close on the parent setup).
    # Distinguishes the cascade path from MANUAL_CLOSE for analytics.
    "CASCADE_CANCELLED",
    # PR #47 / migration 012: body-close-out-of-zone confirmation on
    # an M5 bar after entry. ZONE_EXIT = the shallowest FILLED layer
    # was closed at the trigger. ZONE_EXIT_CANCELLED = a WAITING
    # layer was cancelled because the zone-exit confirmation makes
    # further retest fills unwanted.
    "ZONE_EXIT",
    "ZONE_EXIT_CANCELLED",
]
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
ImpactLevel = Literal["HIGH", "MEDIUM", "LOW"]


class _ModelBase(BaseModel):
    """Base config — strict types, no silent coercion of unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ZoneInput(_ModelBase):
    """Insert payload for ``zones``.

    ``status`` defaults to ``"CONFIRMED"`` because the bot only persists
    zones at the Strong Point confirmation boundary; pre-confirmation
    states (FRESH / TRADEABLE) live only in pipeline output. Downstream
    transitions are made via :meth:`SupabaseLogger.update_zone_status`,
    not by re-inserting with a different status.
    """

    symbol: str = "XAUUSD"
    direction: Direction
    zone_type: ZoneType
    pattern_type: PatternType
    top: Decimal
    bottom: Decimal
    approach_count: int = 0
    qualified_imbalance_at: datetime | None = None
    formed_at: datetime
    last_evaluation_result: dict[str, Any] | None = None
    status: ZoneStatus = "CONFIRMED"


class SetupInput(_ModelBase):
    """Insert payload for ``setups``.

    ``planned_tp2_price`` / ``planned_tp3_price`` (PR #41 / migration
    009) are best-effort at creation: TP1 is required, TP2 / TP3 are
    populated when the lookback yields peaks above the previous TP
    and stay NULL otherwise. NULL slots are recomputed by
    ``tp_manager`` when the previous layer's TP hits, against the
    then-current bars.
    """

    zone_id: UUID
    direction: Direction
    entry_mode: EntryMode
    planned_layer1_price: Decimal
    planned_layer2_price: Decimal
    planned_layer3_price: Decimal
    planned_sl_price: Decimal
    planned_tp1_price: Decimal
    planned_tp2_price: Decimal | None = None
    planned_tp3_price: Decimal | None = None
    status: SetupStatus
    skip_reason: str | None = None


class TradeInput(_ModelBase):
    """Insert payload for ``trades`` (one row per layer, 1..3 per setup)."""

    setup_id: UUID
    layer_number: int = Field(ge=1, le=3)
    direction: Direction
    order_type: OrderType
    mt5_ticket: int | None = None
    entry_price: Decimal | None = None
    exit_price: Decimal | None = None
    lot_size: Decimal
    sl_price: Decimal
    tp_price: Decimal | None = None
    status: TradeStatus
    pnl: Decimal | None = None
    commission: Decimal = Decimal("0")
    swap: Decimal = Decimal("0")
    close_reason: CloseReason | None = None
    filled_at: datetime | None = None
    closed_at: datetime | None = None


class DailyPnlUpdate(_ModelBase):
    """Upsert payload for ``daily_pnl``.

    All fields except ``trading_date`` are optional. ``starting_balance``
    is required by the table schema (NOT NULL) so the first call of a
    new trading day must include it; subsequent calls during the day can
    omit it and only fields with non-None values will be sent (so we
    never overwrite an existing starting_balance with NULL).
    """

    trading_date: date
    starting_balance: Decimal | None = None
    ending_balance: Decimal | None = None
    realized_pnl: Decimal | None = None
    trade_count: int | None = None
    winning_trades: int | None = None
    losing_trades: int | None = None
    halted_at: datetime | None = None


class LogEvent(_ModelBase):
    """Insert payload for ``bot_logs``."""

    level: LogLevel
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    setup_id: UUID | None = None
    trade_id: UUID | None = None


# ---------------------------------------------------------------------------
# Pydantic READ models — typed views of rows pulled back from Supabase.
# Used by position_tracker for type-safe state-machine logic.
# ---------------------------------------------------------------------------


class _ReadModelBase(BaseModel):
    """Read models are lenient: extra fields ignored (forward-compat)."""

    model_config = ConfigDict(extra="ignore")


class Zone(_ReadModelBase):
    """A row from the ``zones`` table.

    Lifecycle columns (``status`` / ``consumed_at`` / ``violated_at`` /
    ``flipped_at`` / ``flipped_direction``) were added in migration
    007. ``invalidated_at`` is the legacy column kept for back-compat
    with rows predating migration 007 — new code reads / writes the
    explicit lifecycle columns above instead.
    """

    id: UUID
    symbol: str
    direction: Direction
    zone_type: ZoneType
    pattern_type: PatternType
    top: Decimal
    bottom: Decimal
    approach_count: int = 0
    qualified_imbalance_at: datetime | None = None
    formed_at: datetime
    invalidated_at: datetime | None = None  # legacy; see docstring
    last_evaluation_result: dict[str, Any] | None = None
    status: ZoneStatus = "CONFIRMED"
    consumed_at: datetime | None = None
    violated_at: datetime | None = None
    flipped_at: datetime | None = None
    flipped_direction: Direction | None = None
    created_at: datetime
    updated_at: datetime


class Setup(_ReadModelBase):
    """A row from the ``setups`` table."""

    id: UUID
    zone_id: UUID
    direction: Direction
    entry_mode: EntryMode
    planned_layer1_price: Decimal
    planned_layer2_price: Decimal
    planned_layer3_price: Decimal
    planned_sl_price: Decimal
    planned_tp1_price: Decimal
    planned_tp2_price: Decimal | None = None
    planned_tp3_price: Decimal | None = None
    status: SetupStatus
    skip_reason: str | None = None
    activated_at: datetime | None = None
    closed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class Trade(_ReadModelBase):
    """A row from the ``trades`` table."""

    id: UUID
    setup_id: UUID
    layer_number: int
    direction: Direction
    order_type: OrderType
    mt5_ticket: int | None = None
    entry_price: Decimal | None = None
    exit_price: Decimal | None = None
    lot_size: Decimal
    sl_price: Decimal
    tp_price: Decimal | None = None
    status: TradeStatus
    pnl: Decimal | None = None
    commission: Decimal = Decimal("0")
    swap: Decimal = Decimal("0")
    close_reason: CloseReason | None = None
    filled_at: datetime | None = None
    closed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class NewsEvent(_ReadModelBase):
    """A row from the ``news_events`` table.

    Populated by a Vercel cron from Finnhub (spec Section 8). The bot
    only reads this table; it never writes to it. ``forecast`` and
    ``actual`` are deliberately ``str`` rather than numeric — Finnhub
    embeds units (``%``, ``$``, ``M``) and qualitative values
    (``Hawkish``, ``Dovish``) that don't round-trip through ``float``.
    The trading decision in :mod:`bot.filters.news_filter` only needs
    ``event_time`` + ``impact_level`` + ``currency``; the value strings
    are informational for the dashboard.
    """

    id: UUID
    event_time: datetime
    currency: str
    title: str
    impact_level: ImpactLevel
    forecast: str | None = None
    actual: str | None = None
    fetched_at: datetime
    created_at: datetime


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class SupabaseLogger:
    """Thin typed wrapper over the supabase-py client for the bot's tables."""

    def __init__(self, url: str, service_role_key: str) -> None:
        self._client: Client = _get_create_client()(url, service_role_key)

    @classmethod
    def from_env(cls) -> SupabaseLogger:
        """Build from ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY`` env vars."""
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        return cls(url, key)

    # ---- zones ----
    def log_zone(self, zone: ZoneInput) -> dict[str, Any]:
        """Insert a new zone row. Returns the inserted row."""
        payload = zone.model_dump(mode="json", exclude_none=True)
        result = self._client.table("zones").insert(payload).execute()
        return result.data[0]

    def get_zones_by_status(
        self, statuses: list[ZoneStatus],
    ) -> list[Zone]:
        """Return every zone whose ``status`` is in ``statuses``.

        The per-bar lifecycle scanner uses this with the non-terminal
        set (CONFIRMED / ACTIVE / CONSUMED / VIOLATED). FLIPPED is
        terminal so callers usually exclude it.
        """
        result = (
            self._client.table("zones")
            .select("*")
            .in_("status", list(statuses))
            .execute()
        )
        return [Zone.model_validate(row) for row in (result.data or [])]

    def get_zone_by_id(self, zone_id: UUID | str) -> Zone | None:
        result = (
            self._client.table("zones")
            .select("*")
            .eq("id", str(zone_id))
            .execute()
        )
        rows = result.data or []
        if not rows:
            return None
        return Zone.model_validate(rows[0])

    def update_zone_status(
        self,
        zone_id: UUID | str,
        status: ZoneStatus,
        *,
        flipped_direction: Direction | None = None,
    ) -> Zone:
        """Patch a zone to ``status`` and stamp the matching timestamps.

        The DB CHECK enforces (post-migration-010):
          * ``status='CONSUMED'`` / ``'VIOLATED'`` / ``'FLIPPED'`` each
            require their headline timestamp NOT NULL.
          * ``status='FLIPPED'`` additionally requires
            ``flipped_direction`` NOT NULL (separate narrower CHECK).
          * ``status='ACTIVE'`` has **no** timestamp restrictions —
            preserves the full audit trail for zones that came via
            the SnD Flip path (FLIPPED → ACTIVE keeps ``violated_at``,
            ``flipped_at``, ``flipped_direction`` populated).
          * ``status='CONFIRMED'`` requires all three timestamps NULL
            (fresh-state invariant).

        Pre-migration-010 behaviour: this method emitted explicit
        NULLs for ``violated_at`` and ``flipped_at`` on the ACTIVE
        transition to satisfy the strict CHECK. That destroyed the
        flip history on the row. After migration 010 the CHECK
        permits the timestamps to persist, so we simply leave them
        alone — only ``status`` gets patched on ACTIVE.
        """
        now = datetime.now(tz=timezone.utc)
        fields: dict[str, Any] = {"status": status}
        if status == "CONSUMED":
            fields["consumed_at"] = now
        elif status == "VIOLATED":
            fields["violated_at"] = now
        elif status == "FLIPPED":
            if flipped_direction is None:
                raise ValueError(
                    "update_zone_status: flipped_direction is required "
                    "when status='FLIPPED'"
                )
            fields["flipped_at"] = now
            fields["flipped_direction"] = flipped_direction
        # status == "ACTIVE": nothing to add. The existing
        # violated_at / flipped_at / flipped_direction values on the
        # row stay as historical markers — migration 010 makes that
        # legal under the CHECK.
        payload = _serialize_update_payload(fields)
        result = (
            self._client.table("zones")
            .update(payload)
            .eq("id", str(zone_id))
            .execute()
        )
        rows = result.data or []
        if not rows:
            raise ValueError(f"zone {zone_id} not found for update")
        return Zone.model_validate(rows[0])

    # ---- setups ----
    def log_setup(self, setup: SetupInput) -> dict[str, Any]:
        """Insert a new setup row. Returns the inserted row."""
        payload = setup.model_dump(mode="json", exclude_none=True)
        result = self._client.table("setups").insert(payload).execute()
        return result.data[0]

    # ---- trades ----
    def log_trade(self, trade: TradeInput) -> dict[str, Any]:
        """Insert a new trade (layer) row. Returns the inserted row."""
        payload = trade.model_dump(mode="json", exclude_none=True)
        result = self._client.table("trades").insert(payload).execute()
        return result.data[0]

    # ---- daily_pnl ----
    def update_daily_pnl(self, update: DailyPnlUpdate) -> dict[str, Any]:
        """Upsert today's daily_pnl row keyed on ``trading_date``.

        Only fields explicitly set on ``update`` are sent — None-valued
        fields are stripped so a partial update never blanks an
        existing column (notably ``starting_balance``).
        """
        payload = update.model_dump(mode="json", exclude_none=True)
        result = (
            self._client.table("daily_pnl")
            .upsert(payload, on_conflict="trading_date")
            .execute()
        )
        return result.data[0]

    # ---- bot_config ----
    def check_bot_config(self, key: str) -> Any:
        """Return the JSONB value for ``key`` (already parsed by supabase-py)."""
        result = (
            self._client.table("bot_config")
            .select("value")
            .eq("key", key)
            .single()
            .execute()
        )
        return result.data["value"]

    # ---- bot_logs ----
    def log_event(
        self,
        level: LogLevel,
        message: str,
        context: dict[str, Any] | None = None,
        *,
        setup_id: UUID | str | None = None,
        trade_id: UUID | str | None = None,
    ) -> dict[str, Any]:
        """Insert a structured log row."""
        event = LogEvent(
            level=level,
            message=message,
            context=context or {},
            setup_id=UUID(str(setup_id)) if setup_id else None,
            trade_id=UUID(str(trade_id)) if trade_id else None,
        )
        payload = event.model_dump(mode="json", exclude_none=True)
        result = self._client.table("bot_logs").insert(payload).execute()
        return result.data[0]

    # ---- read methods (used by position_tracker) ----

    def get_setups_by_status(
        self, statuses: list[str]
    ) -> list[Setup]:
        """Return all setup rows whose status is in ``statuses``."""
        result = (
            self._client.table("setups")
            .select("*")
            .in_("status", list(statuses))
            .execute()
        )
        return [Setup.model_validate(row) for row in (result.data or [])]

    def get_setup_by_id(self, setup_id: UUID | str) -> Setup | None:
        """Return one setup by id, or None if not found."""
        result = (
            self._client.table("setups")
            .select("*")
            .eq("id", str(setup_id))
            .execute()
        )
        rows = result.data or []
        if not rows:
            return None
        return Setup.model_validate(rows[0])

    def get_trade_by_id(self, trade_id: UUID | str) -> Trade | None:
        result = (
            self._client.table("trades")
            .select("*")
            .eq("id", str(trade_id))
            .execute()
        )
        rows = result.data or []
        if not rows:
            return None
        return Trade.model_validate(rows[0])

    def get_trades_for_setup(self, setup_id: UUID | str) -> list[Trade]:
        """Return all trade (layer) rows for a setup, ordered by layer."""
        result = (
            self._client.table("trades")
            .select("*")
            .eq("setup_id", str(setup_id))
            .order("layer_number")
            .execute()
        )
        return [Trade.model_validate(row) for row in (result.data or [])]

    # ---- news_events (read-only — Vercel cron writes) ----

    def get_news_events_in_window(
        self,
        start: datetime,
        end: datetime,
        *,
        currencies: list[str] | None = None,
        min_impact: ImpactLevel = "HIGH",
    ) -> list[NewsEvent]:
        """Return events in ``[start, end]`` matching the impact / currency filter.

        ``min_impact`` follows the standard severity ordering
        ``HIGH > MEDIUM > LOW``; the query selects every level at or
        above the threshold. ``currencies`` is None ⇒ no currency
        filter (return everything in window). Boundary inclusive on
        both ends so the caller's blackout window logic is exact.
        """
        # Severity ordering used to expand min_impact to a set.
        order: list[ImpactLevel] = ["LOW", "MEDIUM", "HIGH"]
        if min_impact not in order:
            raise ValueError(f"unknown min_impact: {min_impact!r}")
        levels = order[order.index(min_impact):]

        q = (
            self._client.table("news_events")
            .select("*")
            .gte("event_time", start.isoformat())
            .lte("event_time", end.isoformat())
            .in_("impact_level", levels)
        )
        if currencies:
            q = q.in_("currency", currencies)
        result = q.order("event_time").execute()
        return [NewsEvent.model_validate(row) for row in (result.data or [])]

    # ---- update methods ----

    def update_setup(
        self, setup_id: UUID | str, **fields: Any
    ) -> Setup:
        """Patch fields on a setup row. Returns the updated row.

        ``None`` values are stripped so partial updates can't blank
        existing columns. Pass an explicit empty string or sentinel if
        you really need to clear something.
        """
        payload = _serialize_update_payload(fields)
        if not payload:
            raise ValueError("update_setup called with no fields to update")
        result = (
            self._client.table("setups")
            .update(payload)
            .eq("id", str(setup_id))
            .execute()
        )
        rows = result.data or []
        if not rows:
            raise ValueError(f"setup {setup_id} not found for update")
        return Setup.model_validate(rows[0])

    def update_trade(
        self, trade_id: UUID | str, **fields: Any
    ) -> Trade:
        """Patch fields on a trade row. Returns the updated row."""
        payload = _serialize_update_payload(fields)
        if not payload:
            raise ValueError("update_trade called with no fields to update")
        result = (
            self._client.table("trades")
            .update(payload)
            .eq("id", str(trade_id))
            .execute()
        )
        rows = result.data or []
        if not rows:
            raise ValueError(f"trade {trade_id} not found for update")
        return Trade.model_validate(rows[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_update_payload(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop None values; convert Decimals/UUIDs/datetimes to JSON-friendly forms."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, Decimal):
            out[k] = str(v)
        elif isinstance(v, UUID):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
