"""Position sizing — fixed for v1, infrastructure for v1.1's risk-based sizing.

v1 (per spec Section 4.4): every layer uses fixed 0.01 lots regardless
of account balance. The fixed mode is the default and what the bot
actually uses today.

v1.1 (per spec Section 7): 0.33% risk per layer (1% per setup across
3 layers). Activated by flipping ``bot_config.sizing_mode`` from
``"FIXED"`` to ``"RISK_BASED"`` — no code change. The infrastructure
is ready now so v1.1 is a config flip, not another build cycle.

Math (XAUUSD specifics)
-----------------------
For XAUUSD on Vantage MT5 (2-decimal price quotes):

* 1 point  = $0.01 in price (MT5's smallest price increment)
* 1 lot    = 100 oz
* 1 point movement on 1 lot = 100 × $0.01 = **$1.00**

Risk formula::

    risk_amount     = balance × risk_per_layer_pct / 100
    sl_distance_pts = |entry - sl| / point_size
    per_lot_risk    = sl_distance_pts × point_value_per_lot
    raw_lot         = risk_amount / per_lot_risk
    final_lot       = floor_to_step(raw_lot), clamped to [min_lot, max_lot]

So `$10,000 × 0.33% / 30pt = $33 / $30 = 1.10 lots` raw, then capped
to ``max_lot`` (default 1.0) with a warning.

Rounding direction
------------------
**Floor** (always round down to ``lot_step``). Conservative — never
exceeds the configured risk budget. ``Decimal`` is used internally so
``floor(1.10 / 0.01)`` doesn't get bitten by float precision (which
would otherwise return 109 not 110 due to ``1.10 / 0.01 == 109.999...``).

Instrument-specific knowledge
-----------------------------
Encoded in :class:`InstrumentSpec`: ``point_size``, ``point_value_per_lot``,
``lot_step``, ``min_lot``, ``max_lot``. ``XAUUSD_SPEC`` is the v1
default. Adding a new instrument later is a new spec record — no
changes to the calculator.

In production, ``point_value_per_lot`` can be confirmed against
``mt5.symbol_info(symbol).trade_tick_value`` at startup; that's a
sanity check, not a runtime input — the sizing math runs without a
live MT5 connection so it stays unit-testable.

Caps
----
* ``min_lot`` is the broker minimum (0.01 for XAUUSD on Vantage). Any
  calculated value below this is bumped up with a warning — a sub-min
  result usually means the SL is too wide for the account size, and
  the trader should probably skip the trade rather than over-risk.
* ``max_lot`` is the bot's safety cap (default 1.0). Any calculated
  value above this is capped with a warning. This prevents runaway
  positions on accidentally-tiny SL distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from loguru import logger


class SizingMode(str, Enum):
    """Which sizing formula to use."""

    FIXED = "FIXED"
    RISK_BASED = "RISK_BASED"


@dataclass(frozen=True)
class InstrumentSpec:
    """Per-instrument knowledge needed for risk-based sizing."""

    symbol: str
    point_size: float           # smallest price increment
    point_value_per_lot: float  # USD per 1 point per 1 lot
    lot_step: float             # smallest lot increment
    min_lot: float              # broker minimum
    max_lot: float              # bot safety cap


# v1 default. ``max_lot=1.0`` matches the spec's "reasonable maximum,
# default 1.0". Bump it in ``bot_config`` once accounts grow large
# enough that 0.33% risk needs >1 lot to express.
XAUUSD_SPEC = InstrumentSpec(
    symbol="XAUUSD",
    point_size=0.01,
    point_value_per_lot=1.0,
    lot_step=0.01,
    min_lot=0.01,
    max_lot=1.0,
)


@dataclass(frozen=True)
class SizingConfig:
    """Tunables; pulled from ``bot_config`` by the orchestrator."""

    mode: SizingMode = SizingMode.FIXED
    fixed_lot_size: float = 0.01
    risk_per_layer_pct: float = 0.33  # used when mode == RISK_BASED


@dataclass(frozen=True)
class LotSizeResult:
    """Calculator output. ``warning`` is non-None iff the result was clamped."""

    lot_size: float
    calculated_risk_amount: float  # USD; what the lot ACTUALLY risks at SL
    sizing_mode_used: SizingMode
    reason: str
    warning: str | None = None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def calculate_lot_size(
    *,
    balance: float,
    entry_price: float,
    sl_price: float,
    instrument: InstrumentSpec = XAUUSD_SPEC,
    config: SizingConfig | None = None,
) -> LotSizeResult:
    """Compute the lot size for one layer, given account state and trade params.

    Raises
    ------
    ValueError
        If ``balance`` is not positive, ``sl_price == entry_price`` in
        risk-based mode, or ``config.mode`` is not a known
        :class:`SizingMode`.
    """
    cfg = config or SizingConfig()

    if balance <= 0:
        raise ValueError(f"balance must be positive, got {balance}")

    if cfg.mode == SizingMode.FIXED:
        return _fixed_size(cfg, entry_price, sl_price, instrument)
    if cfg.mode == SizingMode.RISK_BASED:
        return _risk_based_size(cfg, balance, entry_price, sl_price, instrument)
    raise ValueError(f"unknown sizing mode: {cfg.mode!r}")


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _fixed_size(
    cfg: SizingConfig,
    entry_price: float,
    sl_price: float,
    instrument: InstrumentSpec,
) -> LotSizeResult:
    """v1 path: ignore balance, return the configured fixed lot."""
    sl_dist_pts = abs(entry_price - sl_price) / instrument.point_size
    risk_amount = (
        cfg.fixed_lot_size * sl_dist_pts * instrument.point_value_per_lot
    )
    return LotSizeResult(
        lot_size=cfg.fixed_lot_size,
        calculated_risk_amount=risk_amount,
        sizing_mode_used=SizingMode.FIXED,
        reason=(
            f"FIXED mode: returning configured lot_size={cfg.fixed_lot_size} "
            f"(implied risk at this SL: ${risk_amount:.2f})"
        ),
    )


def _risk_based_size(
    cfg: SizingConfig,
    balance: float,
    entry_price: float,
    sl_price: float,
    instrument: InstrumentSpec,
) -> LotSizeResult:
    """v1.1 path: lot = (balance × risk%) / (SL dist × per-lot value)."""
    sl_distance_dollars = abs(entry_price - sl_price)
    if sl_distance_dollars <= 0:
        raise ValueError(
            f"sl_price ({sl_price}) must differ from entry_price "
            f"({entry_price}) in RISK_BASED mode"
        )

    sl_dist_pts = sl_distance_dollars / instrument.point_size
    per_lot_risk = sl_dist_pts * instrument.point_value_per_lot
    risk_amount = balance * cfg.risk_per_layer_pct / 100.0
    raw_lot = risk_amount / per_lot_risk

    rounded = _floor_to_step(raw_lot, instrument.lot_step)

    warning: str | None = None
    if rounded < instrument.min_lot:
        warning = (
            f"calculated lot {raw_lot:.4f} below broker min "
            f"{instrument.min_lot}; capping to min"
        )
        logger.warning(warning)
        rounded = instrument.min_lot
    elif rounded > instrument.max_lot:
        warning = (
            f"calculated lot {raw_lot:.4f} above safety cap "
            f"{instrument.max_lot}; capping to max"
        )
        logger.warning(warning)
        rounded = instrument.max_lot

    actual_risk = rounded * sl_dist_pts * instrument.point_value_per_lot
    return LotSizeResult(
        lot_size=rounded,
        calculated_risk_amount=actual_risk,
        sizing_mode_used=SizingMode.RISK_BASED,
        reason=(
            f"RISK_BASED: balance=${balance:.2f} × {cfg.risk_per_layer_pct}% "
            f"= ${risk_amount:.2f} budget; SL={sl_dist_pts:.0f}pts × "
            f"${instrument.point_value_per_lot:.2f}/pt/lot = "
            f"${per_lot_risk:.2f}/lot; raw={raw_lot:.4f}, rounded={rounded}"
        ),
        warning=warning,
    )


def _floor_to_step(value: float, step: float) -> float:
    """Floor ``value`` to the nearest multiple of ``step``.

    Uses :class:`Decimal` to avoid the classic float-precision trap
    where ``int(1.10 / 0.01)`` returns ``109`` instead of ``110``.
    """
    v = Decimal(str(value))
    s = Decimal(str(step))
    n = int(v / s)  # truncate toward zero (floor for positive values)
    return float(Decimal(n) * s)
