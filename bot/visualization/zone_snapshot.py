"""Zone snapshot writer — PR #49.

Writes a CSV snapshot of the zones the bot currently sees to a file
inside MT5's ``MQL5/Files/`` sandbox. A companion MQL5 EA
(``mql5/ZoneOverlay.mq5``) polls that file every few seconds and
draws / updates / deletes ``OBJ_RECTANGLE`` objects on the chart.

Why a file (not a network socket / RPC):

* MQL5 EAs run inside the MT5 terminal — easiest IPC is file I/O.
* MQL5 has built-in CSV parsing (``FILE_CSV``); no JSON library
  dependency.
* The MT5 sandbox restricts where EAs can read from — using the
  EA's own ``MQL5/Files/`` is the path of least resistance.
* Atomic write (temp file + rename) gives us reader-side safety
  for free.

Lifecycle
---------

The writer is initialised at bot startup with a target path (from the
``MT5_FILES_DIR`` env var). On each M5 close, ``_detect_new_zones``
calls :meth:`ZoneSnapshotWriter.write` with the latest zones list and
the current price. The writer filters the zones, formats them as CSV,
and writes them atomically.

Failure handling
----------------

Visualization is operator diagnostics. Any failure (missing dir,
permission error, disk full, etc.) MUST NOT crash the trading loop —
the caller wraps the write call in try/except, and this module logs
warnings instead of raising.

If ``MT5_FILES_DIR`` is unset, the writer is constructed in a no-op
mode: ``write()`` returns silently. This lets operators on a
non-Windows dev box (where MT5 isn't running) run the bot without the
visualization noise.

CSV format
----------

Header + one row per zone (UTF-8, no BOM, LF line endings)::

    zone_id,direction,status,flipped_direction,top,bottom,formed_at_unix
    7c97d9ca-7993-4c0c-9f66-4cc2c9281988,BUY,CONFIRMED,,4691.00,4685.00,1747251000
    3f24b6e1-...,SELL,FLIPPED,BUY,4715.50,4711.00,1747252800

``formed_at_unix`` is integer Unix epoch seconds (UTC) so the EA can
``ObjectCreate(..., OBJ_RECTANGLE, 0, formed_at_unix, top, ...)``
directly — MQL5 ``datetime`` is the same Unix epoch representation.

Filters
-------

Configurable via :class:`ZoneSnapshotConfig` (constants at module top
for easy tuning):

* **Distance**: zone bounds must be within ``draw_distance_range``
  points of ``current_price``. Default $50.
* **Age**: zone formed within the last ``draw_age_days`` days.
  Default 7 days. Reduces clutter from week-old zones that are
  almost certainly broken.
* **Status**: only ``CONFIRMED`` / ``ACTIVE`` / ``FLIPPED``.
  ``CONSUMED`` / ``VIOLATED`` are dead by definition and would
  clutter the chart.
* **Visibility (PR #48)** *(optional)*: skip zones with ≥1
  post-formation candles bodying through (obscured). Caller passes
  ``df``; if absent, this filter is skipped.

The "FLIPPED with no flipped_direction" row is impossible per the
zones-table CHECK constraint — we still defensively drop those rows
rather than emit a malformed CSV line.
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Iterable

from loguru import logger

if TYPE_CHECKING:
    import pandas as pd

    from bot.logging.supabase_logger import Zone


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ZoneSnapshotConfig:
    """Tunables for the snapshot writer."""

    output_filename: str = "tradingbot_zones.csv"
    """Filename inside the MT5 ``MQL5/Files/`` directory. The EA reads
    by this name. Don't change unless you also update the EA."""

    draw_distance_range: float = 50.0
    """Only emit zones whose bounds are within this many price points
    of ``current_price``. XAUUSD: ±$50. Reduces visual clutter when
    the DB has zones far from current market."""

    draw_age_days: int = 7
    """Only emit zones with ``formed_at`` within the last N days."""

    apply_visibility_filter: bool = True
    """When ``True`` AND ``df`` is provided to :meth:`write`, exclude
    zones with ≥1 post-formation candles bodying through (PR #48's
    visibility rule). Matches what the bot will actually trade."""


# Statuses to render. CONFIRMED is a brand-new zone the bot has
# detected but not yet traded; ACTIVE is currently being traded;
# FLIPPED is a zone tradeable in its flipped_direction (PR #38).
# CONSUMED / VIOLATED are dead → omitted entirely.
RENDERED_STATUSES: frozenset[str] = frozenset({"CONFIRMED", "ACTIVE", "FLIPPED"})


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


class ZoneSnapshotWriter:
    """Atomically writes the current-zone CSV for the MQL5 EA to read.

    Operator setup: set the ``MT5_FILES_DIR`` env var to MT5's
    ``MQL5/Files/`` path. If unset, the writer constructs in
    no-op mode and ``write()`` is silent.
    """

    def __init__(
        self,
        output_dir: str | None,
        *,
        config: ZoneSnapshotConfig | None = None,
    ) -> None:
        self._cfg = config or ZoneSnapshotConfig()
        # Resolve the output path or fall through to no-op mode.
        if output_dir and output_dir.strip():
            self._output_dir: str | None = output_dir
            self._output_path: str | None = os.path.join(
                output_dir, self._cfg.output_filename,
            )
            # Sanity-check the directory exists at startup. Log a
            # warning and downgrade to no-op if not — better than
            # crashing every M5 close.
            if not os.path.isdir(output_dir):
                logger.warning(
                    "zone snapshot: MT5_FILES_DIR='{}' is not a "
                    "directory — visualization disabled. Create the "
                    "directory or correct the env var to enable.",
                    output_dir,
                )
                self._enabled = False
            else:
                self._enabled = True
                logger.info(
                    "zone snapshot: enabled, writing to {}",
                    self._output_path,
                )
        else:
            self._output_dir = None
            self._output_path = None
            self._enabled = False
            logger.debug(
                "zone snapshot: MT5_FILES_DIR not set — visualization "
                "disabled (no-op mode)"
            )

    @property
    def enabled(self) -> bool:
        """``True`` iff a valid output path was supplied at construction."""
        return self._enabled

    def write(
        self,
        zones: Iterable["Zone"],
        *,
        current_price: float,
        df: "pd.DataFrame | None" = None,
        now: datetime | None = None,
    ) -> int:
        """Filter ``zones`` and write the CSV snapshot atomically.

        Returns the number of zones written (after filtering). Returns
        0 silently if the writer is disabled.

        ``df`` (optional): the latest OHLC df. Used for the visibility
        filter (PR #48). If absent, that filter is skipped.

        ``now`` (optional): override for the age cutoff. Default = real
        UTC now. Used by tests for deterministic outputs.
        """
        if not self._enabled:
            return 0
        if now is None:
            now = datetime.now(tz=timezone.utc)
        zones_list = list(zones)
        try:
            filtered = self._filter(
                zones_list, current_price=current_price, df=df, now=now,
            )
            self._atomic_write(filtered)
        except Exception:
            logger.exception(
                "zone snapshot: write failed; visualization will be "
                "stale until the next M5 close"
            )
            return 0
        logger.debug(
            "zone snapshot: wrote {} zone(s) (filtered from {})",
            len(filtered), len(zones_list),
        )
        return len(filtered)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _filter(
        self,
        zones: list["Zone"],
        *,
        current_price: float,
        df: "pd.DataFrame | None",
        now: datetime,
    ) -> list["Zone"]:
        """Apply the four filters in order. Pure function."""
        cutoff = now - timedelta(days=self._cfg.draw_age_days)
        result: list[Zone] = []
        for z in zones:
            # Status filter.
            if z.status not in RENDERED_STATUSES:
                continue
            # FLIPPED rows MUST have flipped_direction populated per
            # migration 007 CHECK; defensive guard for old / malformed
            # rows.
            if z.status == "FLIPPED" and z.flipped_direction is None:
                continue
            # Age filter (formed_at is tz-aware in the read model).
            if z.formed_at < cutoff:
                continue
            # Distance filter — zone's nearest edge within range.
            top = float(z.top)
            bottom = float(z.bottom)
            nearest_edge = (
                bottom if current_price > top
                else top if current_price < bottom
                else current_price  # inside the zone — definitely close
            )
            if abs(nearest_edge - current_price) > self._cfg.draw_distance_range:
                continue
            # Visibility filter — skip obscured zones if df provided.
            if self._cfg.apply_visibility_filter and df is not None:
                if self._is_obscured(z, df):
                    continue
            result.append(z)
        return result

    @staticmethod
    def _is_obscured(zone: "Zone", df: "pd.DataFrame") -> bool:
        """True iff ≥2 post-formation candles have bodies through the zone.

        Matches PR #48's visibility rule. We import lazily to avoid a
        startup-time circular import (zone_visibility imports nothing
        from this module, but the bot's startup graph is sensitive).
        """
        from bot.strategy.zone_visibility import (  # noqa: PLC0415
            count_bodies_through_zone,
        )
        import pandas as pd  # noqa: PLC0415
        count = count_bodies_through_zone(
            df,
            zone_top=float(zone.top),
            zone_bottom=float(zone.bottom),
            since_time=pd.Timestamp(zone.formed_at),
        )
        # 0 = visible (fresh), 1 = flipped (still tradeable in flip
        # direction), 2+ = obscured / dead. Drop 2+.
        return count >= 2

    def _atomic_write(self, zones: list["Zone"]) -> None:
        """Write CSV via temp-file + rename. Reader (EA) never sees a
        half-written file because ``os.replace`` is atomic on the same
        filesystem (POSIX rename / Windows MoveFileEx with REPLACE).
        """
        assert self._output_path is not None  # _enabled guarantees this
        assert self._output_dir is not None
        # ``mkstemp`` in the same dir as the target so ``os.replace``
        # is on the same filesystem (true atomic rename).
        fd, tmp_path = tempfile.mkstemp(
            prefix=".zones_", suffix=".csv.tmp",
            dir=self._output_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, lineterminator="\n")
                writer.writerow([
                    "zone_id", "direction", "status",
                    "flipped_direction", "top", "bottom",
                    "formed_at_unix",
                ])
                for z in zones:
                    writer.writerow([
                        str(z.id),
                        z.direction,
                        z.status,
                        z.flipped_direction or "",
                        f"{float(z.top):.5f}",
                        f"{float(z.bottom):.5f}",
                        int(z.formed_at.timestamp()),
                    ])
            os.replace(tmp_path, self._output_path)
        except Exception:
            # Best-effort cleanup; if the temp file is left over, the
            # next write's mkstemp will produce a fresh one anyway.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
