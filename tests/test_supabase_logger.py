"""Tests for ``bot.logging.supabase_logger``.

Pydantic-only coverage of the new lifecycle types (ZoneStatus, ZoneInput
default, Zone read model) plus a thin wrapper test for
``update_zone_status`` driven by a mocked supabase-py client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from bot.logging.supabase_logger import (
    SupabaseLogger,
    Zone,
    ZoneInput,
)


NOW = datetime(2026, 5, 8, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# ZoneInput — default status is CONFIRMED
# --------------------------------------------------------------------------- #


class TestZoneInputDefaults:
    def test_status_defaults_to_confirmed(self) -> None:
        zi = ZoneInput(
            direction="BUY", zone_type="STRONG_POINT", pattern_type="RBR",
            top=Decimal("105"), bottom=Decimal("100"),
            formed_at=NOW,
        )
        assert zi.status == "CONFIRMED"

    def test_status_can_be_overridden(self) -> None:
        # Useful only for tests / backfills; production code never
        # inserts with a non-CONFIRMED status.
        zi = ZoneInput(
            direction="BUY", zone_type="STRONG_POINT", pattern_type="RBR",
            top=Decimal("105"), bottom=Decimal("100"),
            formed_at=NOW, status="ACTIVE",
        )
        assert zi.status == "ACTIVE"

    def test_status_rejects_illegal_value(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ZoneInput(
                direction="BUY", zone_type="STRONG_POINT", pattern_type="RBR",
                top=Decimal("105"), bottom=Decimal("100"),
                formed_at=NOW,
                status="BOGUS",  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------------- #
# Zone read model — accepts the new lifecycle columns
# --------------------------------------------------------------------------- #


class TestZoneReadModel:
    def _base_payload(self, **overrides: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": str(uuid4()),
            "symbol": "XAUUSD",
            "direction": "BUY",
            "zone_type": "STRONG_POINT",
            "pattern_type": "RBR",
            "top": "105.00",
            "bottom": "100.00",
            "approach_count": 0,
            "formed_at": NOW.isoformat(),
            "status": "CONFIRMED",
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        }
        payload.update(overrides)
        return payload

    def test_validates_confirmed_row(self) -> None:
        z = Zone.model_validate(self._base_payload())
        assert z.status == "CONFIRMED"
        assert z.consumed_at is None
        assert z.violated_at is None
        assert z.flipped_at is None

    def test_validates_consumed_row(self) -> None:
        z = Zone.model_validate(self._base_payload(
            status="CONSUMED", consumed_at=NOW.isoformat(),
        ))
        assert z.status == "CONSUMED"
        assert z.consumed_at == NOW

    def test_validates_flipped_row(self) -> None:
        z = Zone.model_validate(self._base_payload(
            status="FLIPPED",
            violated_at=NOW.isoformat(),
            flipped_at=NOW.isoformat(),
            flipped_direction="SELL",
        ))
        assert z.status == "FLIPPED"
        assert z.flipped_direction == "SELL"

    def test_unknown_columns_are_ignored(self) -> None:
        # ``_ReadModelBase`` is configured ``extra="ignore"`` so a
        # future column added in a later migration doesn't break
        # existing client code. Validates that intent.
        payload = self._base_payload()
        payload["future_column"] = "anything"
        z = Zone.model_validate(payload)
        assert z.status == "CONFIRMED"


# --------------------------------------------------------------------------- #
# update_zone_status — stamps the right timestamp + flipped_direction
# --------------------------------------------------------------------------- #


def _make_logger_with_mock_client() -> tuple[SupabaseLogger, MagicMock]:
    """Build a SupabaseLogger with the supabase-py client mocked.

    Constructing :class:`SupabaseLogger` directly requires URL +
    service_role_key + the supabase-py ``create_client`` call to
    succeed. We patch the lazy importer to return a factory that
    yields a MagicMock client, then build the logger.
    """
    # We can't easily monkeypatch _get_create_client here without
    # pytest-mock, so we build the bare object and swap the client.
    logger = SupabaseLogger.__new__(SupabaseLogger)
    client = MagicMock()
    logger._client = client  # type: ignore[attr-defined]
    return logger, client


def _stub_table_update(client: MagicMock, returned_row: dict[str, Any]) -> MagicMock:
    """Wire ``client.table(...).update(...).eq(...).execute()`` chain.

    Returns the ``table.update`` mock (the *method*, not its return
    value) so assertions can inspect what payload was sent.
    """
    table = MagicMock()
    table.update.return_value.eq.return_value.execute.return_value.data = (
        [returned_row]
    )
    client.table.return_value = table
    return table.update


class TestUpdateZoneStatus:
    def test_to_consumed_stamps_consumed_at(self) -> None:
        logger, client = _make_logger_with_mock_client()
        zone_id = uuid4()
        returned = {
            "id": str(zone_id),
            "symbol": "XAUUSD", "direction": "BUY",
            "zone_type": "STRONG_POINT", "pattern_type": "RBR",
            "top": "105.00", "bottom": "100.00", "approach_count": 0,
            "formed_at": NOW.isoformat(),
            "status": "CONSUMED",
            "consumed_at": NOW.isoformat(),
            "created_at": NOW.isoformat(), "updated_at": NOW.isoformat(),
        }
        update = _stub_table_update(client, returned)

        result = logger.update_zone_status(zone_id, "CONSUMED")

        assert result.status == "CONSUMED"
        payload = update.call_args.args[0]
        assert payload["status"] == "CONSUMED"
        assert "consumed_at" in payload
        assert "violated_at" not in payload
        assert "flipped_at" not in payload
        assert "flipped_direction" not in payload

    def test_to_violated_stamps_violated_at(self) -> None:
        logger, client = _make_logger_with_mock_client()
        zone_id = uuid4()
        returned = {
            "id": str(zone_id),
            "symbol": "XAUUSD", "direction": "BUY",
            "zone_type": "STRONG_POINT", "pattern_type": "RBR",
            "top": "105.00", "bottom": "100.00", "approach_count": 0,
            "formed_at": NOW.isoformat(),
            "status": "VIOLATED",
            "violated_at": NOW.isoformat(),
            "created_at": NOW.isoformat(), "updated_at": NOW.isoformat(),
        }
        update = _stub_table_update(client, returned)

        logger.update_zone_status(zone_id, "VIOLATED")

        payload = update.call_args.args[0]
        assert payload["status"] == "VIOLATED"
        assert "violated_at" in payload
        assert "consumed_at" not in payload

    def test_to_flipped_requires_direction_and_stamps_all(self) -> None:
        logger, client = _make_logger_with_mock_client()
        zone_id = uuid4()
        returned = {
            "id": str(zone_id),
            "symbol": "XAUUSD", "direction": "BUY",
            "zone_type": "STRONG_POINT", "pattern_type": "RBR",
            "top": "105.00", "bottom": "100.00", "approach_count": 0,
            "formed_at": NOW.isoformat(),
            "status": "FLIPPED",
            "violated_at": NOW.isoformat(),
            "flipped_at": NOW.isoformat(),
            "flipped_direction": "SELL",
            "created_at": NOW.isoformat(), "updated_at": NOW.isoformat(),
        }
        update = _stub_table_update(client, returned)

        logger.update_zone_status(
            zone_id, "FLIPPED", flipped_direction="SELL",
        )

        payload = update.call_args.args[0]
        assert payload["status"] == "FLIPPED"
        assert payload["flipped_direction"] == "SELL"
        assert "flipped_at" in payload

    def test_to_flipped_without_direction_raises(self) -> None:
        logger, _ = _make_logger_with_mock_client()
        with pytest.raises(ValueError, match="flipped_direction is required"):
            logger.update_zone_status(uuid4(), "FLIPPED")

    def test_to_active_stamps_no_timestamp(self) -> None:
        # CONFIRMED → ACTIVE doesn't have a dedicated timestamp (the
        # zone's ``updated_at`` trigger captures the moment).
        logger, client = _make_logger_with_mock_client()
        zone_id = uuid4()
        returned = {
            "id": str(zone_id),
            "symbol": "XAUUSD", "direction": "BUY",
            "zone_type": "STRONG_POINT", "pattern_type": "RBR",
            "top": "105.00", "bottom": "100.00", "approach_count": 0,
            "formed_at": NOW.isoformat(),
            "status": "ACTIVE",
            "created_at": NOW.isoformat(), "updated_at": NOW.isoformat(),
        }
        update = _stub_table_update(client, returned)

        logger.update_zone_status(zone_id, "ACTIVE")

        payload = update.call_args.args[0]
        assert payload == {"status": "ACTIVE"}

    def test_zone_not_found_raises(self) -> None:
        logger, client = _make_logger_with_mock_client()
        table = MagicMock()
        table.update.return_value.eq.return_value.execute.return_value.data = []
        client.table.return_value = table

        with pytest.raises(ValueError, match="not found for update"):
            logger.update_zone_status(uuid4(), "CONSUMED")
