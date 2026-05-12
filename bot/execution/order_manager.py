"""Order manager — places Layer 1 (market) + writes WAITING rows for L2/L3.

**Strategy change (May 2026):** Layers 2 and 3 are no longer placed at
the broker as pending limit orders. They're tracked in Supabase as
``WAITING`` trade rows; the :mod:`bot.execution.entry_trigger` module
fires them as market orders when the live tick reaches each layer's
trigger price.

**TP1 refinement (May 2026, loosened-rules PR):** TP1 is now computed
in the strategy/orchestrator layer (see
:mod:`bot.strategy.tp1_target`) and passed in as the ``tp1_price``
argument. The previous BOS_LEVEL / FIXED_DISTANCE machinery is gone
along with the break-and-close Strong Point gate that produced
``broken_swing``. The caller is expected to have already decided
whether a tradeable TP1 exists; ``place_layered_orders`` just
records it.

Why bot-managed instead of broker-pending:

* Real-time control of each entry decision — easier to skip if the
  setup invalidates between layers.
* Cleaner backtest path (same trigger logic on historical data as on
  live ticks).
* The bot can apply gating rules (news filter, exposure cap) right
  before each layer fires, not just at setup creation.

Pipeline order
--------------
::

    1. Compute layer prices (always doable, no I/O)
    2. Pre-checks (input validation, zone tradeable, SL sane)
    3. log_setup → Supabase (status=PENDING) ............ side effect #1
    4. place_market_order → MT5 (Layer 1) ............... side effect #2
    5. Resolve filled price; gap-through check
    6. log_trade for Layer 1 (FILLED) ................... side effect #3
    7. log_trade for Layers 2 + 3 (WAITING, no ticket)... side effects #4, #5

Failures degrade ``OrderPlacementResult.status``:

    PLACED   Layer 1 placed + all 3 trade rows written
    SKIPPED  Layer 1 placed but gap-through detected; Layer 1 closed
    FAILED   pre-checks failed OR Layer 1 itself failed

Design decisions
----------------

1. **Supabase before MT5.** Setup tracked even on MT5 failure;
   ``setup_id`` available to tag MT5 orders via ``comment``.

2. **No TP on the MT5 order.** Spec Section 6.1 needs a 50% close at
   TP1, which MT5's TP closes 100%. TP1 is bot-managed by
   ``tp1_manager``. Only SL on the broker side as a backstop.

3. **L2/L3 trade rows have ``mt5_ticket=None`` and ``entry_price=None``.**
   They get populated when ``entry_trigger`` fires the layers.

4. **TP1 is supplied, not computed.** Hoisting TP1 out of
   ``order_manager`` lets the orchestrator reject zones with no
   tradeable peak before any DB write or broker call — same shape
   as the SL-distance gate.

5. **Comment format**: ``bot:L1:s={first 8 chars of setup_id}`` — fits
   MT5's 31-char limit. Only Layer 1 gets a broker comment;
   ``entry_trigger`` writes its own comments when it fires L2/L3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal
from uuid import UUID

from loguru import logger

from bot.execution.mt5_connector import MT5Connector
from bot.logging.supabase_logger import (
    SetupInput,
    SupabaseLogger,
    TradeInput,
)
from bot.strategy.strong_point import ValidatedZone

PlacementStatus = Literal["PLACED", "FAILED", "SKIPPED"]


@dataclass(frozen=True)
class OrderManagerConfig:
    symbol: str = "XAUUSD"

    # Gap-through tolerance in *price units*. Layer 1's filled price
    # must not be more than this much past the far zone edge. Default
    # 5 points = $0.05 (XAUUSD's smallest price increment).
    gap_tolerance_dollars: float = 0.05


@dataclass(frozen=True)
class OrderPlacementResult:
    setup_id: UUID | None
    layer_1_ticket: int | None  # broker ticket for Layer 1
    layer_2_trade_id: UUID | None  # Supabase row UUID for the WAITING L2 row
    layer_3_trade_id: UUID | None  # Supabase row UUID for the WAITING L3 row
    layer_1_filled_price: float | None
    sl_price: float
    tp1_price: float
    status: PlacementStatus
    error_messages: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def place_layered_orders(
    zone: ValidatedZone,
    zone_id: UUID,
    lot_size: float,
    sl_price: float,
    tp1_price: float,
    *,
    mt5: MT5Connector,
    supabase: SupabaseLogger,
    config: OrderManagerConfig | None = None,
) -> OrderPlacementResult:
    """Place Layer 1 + write Supabase rows for Layers 2/3 (status=WAITING).

    Layers 2 and 3 are NOT sent to the broker. ``entry_trigger`` fires
    them as market orders when the live tick reaches their trigger price.

    ``tp1_price`` is supplied by the caller (see
    :mod:`bot.strategy.tp1_target`). Pre-checks reject obviously
    insane values (zero, wrong side of entry) but do not enforce a
    minimum distance or R:R ratio.
    """
    cfg = config or OrderManagerConfig()
    errors: list[str] = []

    # 1. Compute layer prices (TP1 already supplied).
    layer_1_price, layer_2_price, layer_3_price = (
        _compute_layer_prices(zone)
    )

    # 2. Pre-checks.
    pre_check_error = _validate_inputs(zone, lot_size, sl_price, tp1_price)
    if pre_check_error is not None:
        errors.append(pre_check_error)
        logger.warning(f"order_manager pre-check failed: {pre_check_error}")
        return _failed_result(sl_price, tp1_price, errors)

    # 3. Create setup record.
    #
    # v1 only handles Strong Point setups → entry_mode hardcoded.
    # When Imbalance (setup #4) lands, we'll add a discriminator field
    # on the validated zone (e.g. an ``is_imbalance`` flag emitted by
    # a future ImbalanceZone validator) and dispatch here.
    entry_mode = "STRONG_POINT_FIRST_TOUCH"
    try:
        setup_row = supabase.log_setup(SetupInput(
            zone_id=zone_id,
            direction=zone.direction,
            entry_mode=entry_mode,
            planned_layer1_price=Decimal(str(layer_1_price)),
            planned_layer2_price=Decimal(str(layer_2_price)),
            planned_layer3_price=Decimal(str(layer_3_price)),
            planned_sl_price=Decimal(str(sl_price)),
            planned_tp1_price=Decimal(str(tp1_price)),
            status="PENDING",
        ))
        setup_id = UUID(str(setup_row["id"]))
    except Exception as e:
        errors.append(f"Supabase log_setup failed: {e}")
        logger.exception("order_manager: setup creation failed; aborting")
        return _failed_result(sl_price, tp1_price, errors)

    setup_id_short = str(setup_id)[:8]

    # 4. Place Layer 1 (market).
    try:
        layer_1_ticket = mt5.place_market_order(
            symbol=cfg.symbol,
            direction=zone.direction,
            lot_size=lot_size,
            sl=sl_price,
            tp=None,  # bot-managed; see module docstring.
            comment=f"bot:L1:s={setup_id_short}",
        )
    except Exception as e:
        errors.append(f"Layer 1 market order failed: {e}")
        logger.exception("order_manager: Layer 1 placement failed")
        _try_log_event(
            supabase, "ERROR",
            "Layer 1 placement failed",
            context={"error": str(e)},
            setup_id=setup_id,
        )
        return _failed_result(sl_price, tp1_price, errors, setup_id=setup_id)

    # 5. Resolve filled price + gap-through check.
    layer_1_filled_price = _resolve_filled_price(
        mt5, cfg.symbol, layer_1_ticket
    )
    if layer_1_filled_price is not None:
        gap = _detect_gap_through(
            zone, layer_1_filled_price, cfg.gap_tolerance_dollars
        )
        if gap is not None:
            try:
                mt5.close_position(layer_1_ticket)
            except Exception as e:
                errors.append(f"failed to close Layer 1 after gap: {e}")
                logger.exception("order_manager: close-on-gap failed")
            errors.append(gap)
            _try_log_event(
                supabase, "WARN", "Setup skipped: gap through zone",
                context={
                    "filled_price": layer_1_filled_price,
                    "zone_top": zone.top,
                    "zone_bottom": zone.bottom,
                    "tolerance": cfg.gap_tolerance_dollars,
                },
                setup_id=setup_id,
            )
            return OrderPlacementResult(
                setup_id=setup_id,
                layer_1_ticket=layer_1_ticket,
                layer_2_trade_id=None,
                layer_3_trade_id=None,
                layer_1_filled_price=layer_1_filled_price,
                sl_price=sl_price,
                tp1_price=tp1_price,
                status="SKIPPED",
                error_messages=errors,
            )

    # 6. Write trade records: Layer 1 FILLED, Layers 2/3 WAITING.
    layer_2_trade_id, layer_3_trade_id = _write_trade_rows(
        supabase=supabase,
        setup_id=setup_id,
        zone=zone,
        lot_size=lot_size,
        sl_price=sl_price,
        layer_1_ticket=layer_1_ticket,
        layer_1_filled_price=layer_1_filled_price,
        errors=errors,
    )

    return OrderPlacementResult(
        setup_id=setup_id,
        layer_1_ticket=layer_1_ticket,
        layer_2_trade_id=layer_2_trade_id,
        layer_3_trade_id=layer_3_trade_id,
        layer_1_filled_price=layer_1_filled_price,
        sl_price=sl_price,
        tp1_price=tp1_price,
        status="PLACED",
        error_messages=errors,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _compute_layer_prices(
    zone: ValidatedZone,
) -> tuple[float, float, float]:
    """Return (layer_1_price, layer_2_price, layer_3_price).

    BUY: L1=zone.top, L2=midpoint, L3=zone.bottom. SELL: mirror.
    """
    midpoint = (zone.top + zone.bottom) / 2.0
    if zone.direction == "BUY":
        return zone.top, midpoint, zone.bottom
    return zone.bottom, midpoint, zone.top


def _validate_inputs(
    zone: ValidatedZone, lot_size: float, sl_price: float,
    tp1_price: float,
) -> str | None:
    if not zone.refined_zone.is_tradeable:
        return f"zone not tradeable: {zone.refined_zone.rejection_reason}"
    if not zone.is_strong_point:
        return "zone is not a Strong Point"
    if lot_size <= 0:
        return f"invalid lot_size: {lot_size}"
    if zone.direction == "BUY" and sl_price >= zone.top:
        return (
            f"SL ({sl_price}) must be below zone.top ({zone.top}) for BUY"
        )
    if zone.direction == "SELL" and sl_price <= zone.bottom:
        return (
            f"SL ({sl_price}) must be above zone.bottom ({zone.bottom}) for SELL"
        )
    # TP1 must be on the favourable side of Layer 1's entry.
    if zone.direction == "BUY" and tp1_price <= zone.top:
        return (
            f"TP1 ({tp1_price}) must be above zone.top ({zone.top}) for BUY"
        )
    if zone.direction == "SELL" and tp1_price >= zone.bottom:
        return (
            f"TP1 ({tp1_price}) must be below zone.bottom "
            f"({zone.bottom}) for SELL"
        )
    return None


def _resolve_filled_price(
    mt5: MT5Connector, symbol: str, ticket: int
) -> float | None:
    try:
        positions = mt5.get_open_positions(symbol=symbol)
    except Exception:
        logger.exception("order_manager: get_open_positions failed")
        return None
    for p in positions:
        if p.get("ticket") == ticket:
            price = p.get("price_open")
            if price is not None:
                return float(price)
    return None


def _detect_gap_through(
    zone: ValidatedZone, filled_price: float, tolerance: float
) -> str | None:
    """Error message if fill is past the far zone edge by > tolerance."""
    if zone.direction == "BUY":
        threshold = zone.bottom - tolerance
        if filled_price < threshold:
            return (
                f"GAP_THROUGH_ZONE: filled at {filled_price} below "
                f"zone.bottom={zone.bottom} - tolerance={tolerance}"
            )
    else:  # SELL
        threshold = zone.top + tolerance
        if filled_price > threshold:
            return (
                f"GAP_THROUGH_ZONE: filled at {filled_price} above "
                f"zone.top={zone.top} + tolerance={tolerance}"
            )
    return None


def _write_trade_rows(
    *,
    supabase: SupabaseLogger,
    setup_id: UUID,
    zone: ValidatedZone,
    lot_size: float,
    sl_price: float,
    layer_1_ticket: int,
    layer_1_filled_price: float | None,
    errors: list[str],
) -> tuple[UUID | None, UUID | None]:
    """Write all three trade rows. Best-effort.

    Returns (layer_2_trade_id, layer_3_trade_id). Either may be None
    if its insert failed — Layer 1 errors are logged but don't fail
    the whole call (Layer 1 is already on the broker; its row is
    repairable).
    """
    layer_2_trade_id: UUID | None = None
    layer_3_trade_id: UUID | None = None

    # Layer 1: FILLED with broker ticket and entry price.
    try:
        supabase.log_trade(TradeInput(
            setup_id=setup_id,
            layer_number=1,
            direction=zone.direction,
            order_type="MARKET",
            mt5_ticket=layer_1_ticket,
            entry_price=(
                Decimal(str(layer_1_filled_price))
                if layer_1_filled_price is not None else None
            ),
            lot_size=Decimal(str(lot_size)),
            sl_price=Decimal(str(sl_price)),
            tp_price=None,
            status="FILLED",
        ))
    except Exception as e:
        errors.append(f"Layer 1 trade row write failed: {e}")
        logger.exception(
            "order_manager: Layer 1 trade row write failed — "
            "broker has the order but Supabase doesn't; reconciliation needed"
        )

    # Layer 2: WAITING — entry_trigger will fire it.
    for layer_num in (2, 3):
        try:
            row = supabase.log_trade(TradeInput(
                setup_id=setup_id,
                layer_number=layer_num,
                direction=zone.direction,
                order_type="MARKET",  # all bot-fired layers are market orders
                mt5_ticket=None,
                entry_price=None,
                lot_size=Decimal(str(lot_size)),
                sl_price=Decimal(str(sl_price)),
                tp_price=None,
                status="WAITING",
            ))
            trade_id = UUID(str(row["id"]))
            if layer_num == 2:
                layer_2_trade_id = trade_id
            else:
                layer_3_trade_id = trade_id
        except Exception as e:
            errors.append(f"Layer {layer_num} WAITING trade row failed: {e}")
            logger.exception(
                f"order_manager: Layer {layer_num} WAITING row write failed"
            )

    return layer_2_trade_id, layer_3_trade_id


def _try_log_event(
    supabase: SupabaseLogger,
    level: str,
    message: str,
    *,
    context: dict,
    setup_id: UUID,
) -> None:
    """Best-effort bot_logs write — never raises."""
    try:
        supabase.log_event(
            level=level,  # type: ignore[arg-type]
            message=message,
            context=context,
            setup_id=setup_id,
        )
    except Exception:
        logger.exception("order_manager: log_event failed (non-fatal)")


def _failed_result(
    sl_price: float,
    tp1_price: float,
    errors: list[str],
    *,
    setup_id: UUID | None = None,
) -> OrderPlacementResult:
    return OrderPlacementResult(
        setup_id=setup_id,
        layer_1_ticket=None,
        layer_2_trade_id=None,
        layer_3_trade_id=None,
        layer_1_filled_price=None,
        sl_price=sl_price,
        tp1_price=tp1_price,
        status="FAILED",
        error_messages=list(errors),
    )
