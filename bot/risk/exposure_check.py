"""Exposure cap — enforce ``max_simultaneous_setups`` from spec Section 7.

Stops the bot from over-trading by blocking new setup activation once
the configured concurrent-setup count is reached. Pure logic; the
orchestrator passes the active count and the cap; we return the
verdict.

What counts as "active"
-----------------------
A setup consumes exposure capacity while it's in any of:

    PENDING    pending limit orders waiting to fill
    ACTIVE     at least one layer filled, trade in progress
    TP1_HIT    TP1 partial-take done, runner is still open

Everything else (``CLOSED``, ``SKIPPED``, ``STOPPED_OUT``) is a
terminal state and does *not* count.

**Why TP1_HIT counts.** After TP1, 50% of the position is closed but
the runner remains. The runner is still managed risk on the account —
SL is at break-even but the position still consumes margin and demands
attention. Counting it toward exposure prevents the bot from stacking
new setups while three runners are sitting open.

Math
----
The question is always "**can ONE more setup be opened?**":

    can_open_new := active_count < max_simultaneous

(Equivalently ``active_count + 1 <= max_simultaneous``.) The
``with_candidate`` flag in :func:`check_exposure` is **purely
documentation** — it does not affect the arithmetic. It exists so the
result's ``reason`` text can disambiguate "checked because we have a
specific candidate" from "checked at top of loop, no candidate yet"
when surfaced in logs.

Edge cases
----------
* ``max_simultaneous == 0`` → all calls return False. Effectively a
  kill-switch via config; the dashboard's main kill switch is
  ``bot_config.kill_switch``, but setting max to 0 also works.
* ``max_simultaneous < 0`` → :class:`ValueError`; configuration error.
* ``active_count > max_simultaneous`` → False with a warning logged.
  Shouldn't normally happen but defended against — usually means a
  bug elsewhere or a manual SQL change.
* ``active_count < 0`` → :class:`ValueError`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from loguru import logger

# Statuses (also defined in supabase_logger / migration 001) that
# count toward concurrent-setup exposure.
ACTIVE_STATUSES: frozenset[str] = frozenset({"PENDING", "ACTIVE", "TP1_HIT"})

# Terminal statuses — useful for callers that want to filter the
# "done" pile.
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"CLOSED", "SKIPPED", "STOPPED_OUT"}
)


@dataclass(frozen=True)
class ExposureCheckResult:
    """Verdict from :func:`check_exposure`."""

    can_open_new: bool
    current_count: int
    max_allowed: int
    reason: str | None = None


def check_exposure(
    active_count: int,
    max_simultaneous: int,
    *,
    with_candidate: bool = False,
) -> ExposureCheckResult:
    """Return whether the bot has capacity to open one more setup.

    Parameters
    ----------
    active_count
        Number of currently-active setups (i.e. in ``PENDING`` /
        ``ACTIVE`` / ``TP1_HIT``). NOT including any candidate being
        evaluated.
    max_simultaneous
        Inclusive cap on concurrent active setups. Pulled from
        ``bot_config.max_simultaneous_setups`` (default 3).
    with_candidate
        Cosmetic flag. Does not change the arithmetic — the question
        is always "can one more be opened?". When True, the result's
        ``reason`` reflects that the check was made for a specific
        candidate (used in setup-skipped log lines).

    Raises
    ------
    ValueError
        ``max_simultaneous < 0`` or ``active_count < 0``.
    """
    if max_simultaneous < 0:
        raise ValueError(
            f"max_simultaneous must be >= 0, got {max_simultaneous}"
        )
    if active_count < 0:
        raise ValueError(f"active_count must be >= 0, got {active_count}")

    if active_count > max_simultaneous:
        logger.warning(
            f"over-exposure detected: active_count={active_count} > "
            f"max_simultaneous={max_simultaneous}"
        )

    can_open_new = active_count < max_simultaneous
    reason: str | None = None if can_open_new else "MAX_EXPOSURE_REACHED"

    logger.debug(
        f"exposure: active={active_count} max={max_simultaneous} "
        f"can_open_new={can_open_new} candidate={with_candidate}"
    )
    return ExposureCheckResult(
        can_open_new=can_open_new,
        current_count=active_count,
        max_allowed=max_simultaneous,
        reason=reason,
    )


def count_active_setups(setups: Iterable[Any]) -> int:
    """Count setups whose status is in :data:`ACTIVE_STATUSES`.

    Each item may be either an object with a ``status`` attribute or a
    :class:`Mapping` with a ``status`` key. Supporting both shapes lets
    this work with pydantic models (object access), raw Supabase rows
    (dicts), and ad-hoc test fixtures (either) without callers having
    to massage their data.
    """
    return sum(1 for s in setups if _extract_status(s) in ACTIVE_STATUSES)


def _extract_status(setup: Any) -> str | None:
    """Pull the ``status`` field from a setup-shaped object or dict."""
    if isinstance(setup, Mapping):
        return setup.get("status")
    return getattr(setup, "status", None)
