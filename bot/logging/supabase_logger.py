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
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from supabase import Client, create_client


# ---------------------------------------------------------------------------
# Pydantic input models
# ---------------------------------------------------------------------------

Direction = Literal["BUY", "SELL"]
ZoneType = Literal["STRONG_POINT", "IMBALANCE"]
PatternType = Literal["W", "M", "N"]
EntryMode = Literal["STRONG_POINT_FIRST_TOUCH", "IMBALANCE_FIRST_TOUCH"]
SetupStatus = Literal[
    "PENDING", "ACTIVE", "TP1_HIT", "CLOSED", "SKIPPED", "STOPPED_OUT"
]
TradeStatus = Literal[
    "PENDING", "FILLED", "PARTIALLY_CLOSED", "CLOSED", "CANCELLED"
]
OrderType = Literal["MARKET", "LIMIT"]
CloseReason = Literal["TP1", "SL_HIT", "BE_HIT", "MANUAL_CLOSE", "NEWS_CLOSE"]
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]
ImpactLevel = Literal["HIGH", "MEDIUM", "LOW"]


class _ModelBase(BaseModel):
    """Base config — strict types, no silent coercion of unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ZoneInput(_ModelBase):
    """Insert payload for ``zones``."""

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


class SetupInput(_ModelBase):
    """Insert payload for ``setups``."""

    zone_id: UUID
    direction: Direction
    entry_mode: EntryMode
    planned_layer1_price: Decimal
    planned_layer2_price: Decimal
    planned_layer3_price: Decimal
    planned_sl_price: Decimal
    planned_tp1_price: Decimal
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
# Wrapper
# ---------------------------------------------------------------------------


class SupabaseLogger:
    """Thin typed wrapper over the supabase-py client for the bot's tables."""

    def __init__(self, url: str, service_role_key: str) -> None:
        self._client: Client = create_client(url, service_role_key)

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
