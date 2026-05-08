"""Order manager — places the three layered orders for a setup.

Per spec Section 4:

* **Layer 1**: market order at the entry tick (fires immediately when
  this function is called — the *orchestrator* is responsible for
  watching ticks and only calling us when entry conditions are met).
* **Layer 2**: pending limit at zone midpoint.
* **Layer 3**: pending limit at the far zone edge (zone.bottom for BUY,
  zone.top for SELL).

All three share one SL. After this module returns, ``tp1_manager`` and
``sl_manager`` take over for the rest of the lifecycle.

Pipeline order
--------------
::

    1. Pre-checks (input validation, zone tradeable, SL sane)
    2. Compute layer prices + TP1
    3. log_setup → Supabase (status=PENDING) ............ side effect #1
    4. place_market_order → MT5 (Layer 1) ............... side effect #2
    5. Look up filled price; gap-through check
    6. place_limit_order × 2 → MT5 (Layers 2/3) ......... side effects #3, #4
    7. log_trade × {1..3} → Supabase ..................... side effects #5..

Failures at any step degrade the result status:

    PLACED   all 3 layers placed
    PARTIAL  Layer 1 placed, but at least one of 2/3 failed
    FAILED   pre-checks failed, OR Layer 1 itself failed
    SKIPPED  Layer 1 placed but gap-through detected; Layer 1 closed

Design decisions called out in PR #13
-------------------------------------

1. **Supabase before MT5.** Even if MT5 placement fails, the setup is
   tracked in the database for reconciliation. The setup_id is also
   needed to tag MT5 orders via the comment field.

2. **No TP on the MT5 orders.** Spec Section 6.1 requires a 50% close at
   TP1 plus break-even SL move on the runner. MT5's built-in TP closes
   100%, which would defeat the partial-take design. TP1 is therefore
   a bot-managed event monitored by ``tp1_manager``. Only SL is set on
   the broker side as a catastrophic backstop.

3. **Idempotency is the caller's responsibility.** The orchestrator
   tracks which zones have been traded; we don't query Supabase to
   check. This keeps the function fast and pure-call. (A defensive
   check could be added later if buggy callers become a real risk.)

4. **Order comment format**: ``bot:L{1,2,3}:s={first 8 chars of setup_id}``
   — fits MT5's 31-char comment limit and gives operator-side
   traceability when looking at the broker's trade history.

5. **Partial fills are accepted (Section 4.5).** If Layer 2 or Layer 3
   fails to place, we log the failure and continue with the layers we
   got. Layer 1 must succeed (it's the entry).

6. **Setup record stays at PENDING in v1** if anything goes sideways
   downstream. The supabase_logger doesn't yet have an
   ``update_setup_status`` method; we log to ``bot_logs`` for
   reconciliation. Adding update_setup is a follow-up.
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
from bot.strategy.imbalance import ImbalanceZone

PlacementStatus = Literal["PLACED", "PARTIAL", "FAILED", "SKIPPED"]


@dataclass(frozen=True)
class OrderManagerConfig:
    symbol: str = "XAUUSD"
    tp1_distance_dollars: float = 4.0
    # Gap-through tolerance in *price units*. Layer 1's filled price
    # must not be more than this much past the far zone edge. Default
    # 5 points = $0.05 (XAUUSD's smallest price increment).
    gap_tolerance_dollars: float = 0.05


@dataclass(frozen=True)
class OrderPlacementResult:
    setup_id: UUID | None
    layer_1_ticket: int | None
    layer_2_ticket: int | None
    layer_3_ticket: int | None
    layer_1_filled_price: float | None
    sl_price: float
    tp1_price: float
    status: PlacementStatus
    error_messages: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def place_layered_orders(
    zone: ImbalanceZone,
    zone_id: UUID,
    lot_size: float,
    sl_price: float,
    *,
    mt5: MT5Connector,
    supabase: SupabaseLogger,
    config: OrderManagerConfig | None = None,
) -> OrderPlacementResult:
    """Place Layer 1 (market) + Layers 2-3 (limits) for a tradeable zone.

    Parameters
    ----------
    zone
        :class:`ImbalanceZone` from the Phase B pipeline. Must be
        ``is_tradeable=True`` and either ``is_strong_point=True`` or
        ``is_imbalance=True``.
    zone_id
        UUID of the already-persisted zone row in Supabase.
    lot_size
        From :func:`bot.risk.position_sizing.calculate_lot_size`.
    sl_price
        Stop-loss price for *all three* layers. Must be on the correct
        side of the zone (below zone.top for BUY; above zone.bottom for
        SELL).
    mt5
        :class:`MT5Connector` instance.
    supabase
        :class:`SupabaseLogger` instance.
    config
        Optional :class:`OrderManagerConfig` overrides.
    """
    cfg = config or OrderManagerConfig()
    errors: list[str] = []

    # 1. Compute layer prices + TP1 (always doable; no I/O).
    layer_prices = _compute_layer_prices(zone, cfg)
    layer_1_price, layer_2_price, layer_3_price, tp1_price = layer_prices

    # 2. Pre-checks. Fail fast before any side effects.
    pre_check_error = _validate_inputs(zone, lot_size, sl_price)
    if pre_check_error is not None:
        errors.append(pre_check_error)
        logger.warning(f"order_manager pre-check failed: {pre_check_error}")
        return _failed_result(sl_price, tp1_price, errors)

    # 3. Create setup record in Supabase. If this fails, no MT5 calls.
    entry_mode = (
        "IMBALANCE_FIRST_TOUCH" if zone.is_imbalance
        else "STRONG_POINT_FIRST_TOUCH"
    )
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
            tp=None,  # TP1 is bot-managed; see module docstring.
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

    # 5. Look up filled price and check for gap-through.
    layer_1_filled_price = _resolve_filled_price(mt5, cfg.symbol, layer_1_ticket)
    if layer_1_filled_price is not None:
        gap = _detect_gap_through(zone, layer_1_filled_price, cfg.gap_tolerance_dollars)
        if gap is not None:
            # Close Layer 1 — we shouldn't be in this trade.
            try:
                mt5.close_position(layer_1_ticket)
            except Exception as e:
                errors.append(f"failed to close Layer 1 after gap detected: {e}")
                logger.exception("order_manager: failed to close Layer 1 on gap")
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
                layer_2_ticket=None,
                layer_3_ticket=None,
                layer_1_filled_price=layer_1_filled_price,
                sl_price=sl_price,
                tp1_price=tp1_price,
                status="SKIPPED",
                error_messages=errors,
            )

    # 6. Place Layers 2 & 3 (pending limits). Failures here are
    # accepted as partial fills (spec Section 4.5).
    layer_2_ticket = _try_place_limit(
        mt5, cfg.symbol, zone.direction, lot_size, layer_2_price,
        sl_price, f"bot:L2:s={setup_id_short}", errors, layer=2,
    )
    layer_3_ticket = _try_place_limit(
        mt5, cfg.symbol, zone.direction, lot_size, layer_3_price,
        sl_price, f"bot:L3:s={setup_id_short}", errors, layer=3,
    )

    # 7. Determine final status.
    placed = sum(
        1 for t in (layer_1_ticket, layer_2_ticket, layer_3_ticket)
        if t is not None
    )
    status: PlacementStatus = "PLACED" if placed == 3 else "PARTIAL"

    # 8. Write trade records (best-effort — Layer 1 is already on the
    # broker, so a Supabase failure here is a bookkeeping problem, not
    # a trading problem).
    _try_log_trades(
        supabase=supabase,
        setup_id=setup_id,
        zone=zone,
        lot_size=lot_size,
        sl_price=sl_price,
        tp1_price=tp1_price,
        layer_1_ticket=layer_1_ticket,
        layer_1_filled_price=layer_1_filled_price,
        layer_2_ticket=layer_2_ticket,
        layer_2_price=layer_2_price,
        layer_3_ticket=layer_3_ticket,
        layer_3_price=layer_3_price,
        errors=errors,
    )

    return OrderPlacementResult(
        setup_id=setup_id,
        layer_1_ticket=layer_1_ticket,
        layer_2_ticket=layer_2_ticket,
        layer_3_ticket=layer_3_ticket,
        layer_1_filled_price=layer_1_filled_price,
        sl_price=sl_price,
        tp1_price=tp1_price,
        status=status,
        error_messages=errors,
    )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _compute_layer_prices(
    zone: ImbalanceZone, cfg: OrderManagerConfig
) -> tuple[float, float, float, float]:
    """Return (layer_1_price, layer_2_price, layer_3_price, tp1_price)."""
    midpoint = (zone.top + zone.bottom) / 2.0
    if zone.direction == "BUY":
        return zone.top, midpoint, zone.bottom, zone.top + cfg.tp1_distance_dollars
    # SELL
    return zone.bottom, midpoint, zone.top, zone.bottom - cfg.tp1_distance_dollars


def _validate_inputs(
    zone: ImbalanceZone, lot_size: float, sl_price: float
) -> str | None:
    """Return an error message if pre-checks fail, else None."""
    if not zone.is_tradeable:
        return f"zone not tradeable: {zone.rejection_reason}"
    if not (zone.is_strong_point or zone.is_imbalance):
        return "zone is neither a Strong Point nor an Imbalance"
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
    return None


def _resolve_filled_price(
    mt5: MT5Connector, symbol: str, ticket: int
) -> float | None:
    """Look up the open position's fill price by ticket. None if not found."""
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
    zone: ImbalanceZone, filled_price: float, tolerance: float
) -> str | None:
    """Return an error message if the fill is past the far edge by > tolerance.

    For BUY: filled_price < zone.bottom - tolerance is a gap-through.
    For SELL: filled_price > zone.top + tolerance is a gap-through.
    """
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


def _try_place_limit(
    mt5: MT5Connector,
    symbol: str,
    direction: str,
    lot_size: float,
    price: float,
    sl_price: float,
    comment: str,
    errors: list[str],
    *,
    layer: int,
) -> int | None:
    try:
        return mt5.place_limit_order(
            symbol=symbol,
            direction=direction,  # type: ignore[arg-type]
            lot_size=lot_size,
            price=price,
            sl=sl_price,
            tp=None,
            comment=comment,
        )
    except Exception as e:
        errors.append(f"Layer {layer} pending limit failed: {e}")
        logger.exception(f"order_manager: Layer {layer} placement failed")
        return None


def _try_log_trades(
    *,
    supabase: SupabaseLogger,
    setup_id: UUID,
    zone: ImbalanceZone,
    lot_size: float,
    sl_price: float,
    tp1_price: float,
    layer_1_ticket: int | None,
    layer_1_filled_price: float | None,
    layer_2_ticket: int | None,
    layer_2_price: float,
    layer_3_ticket: int | None,
    layer_3_price: float,
    errors: list[str],
) -> None:
    """Best-effort trade-row writes. Failures don't change the result status."""
    try:
        if layer_1_ticket is not None:
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
                tp_price=None,  # bot-managed, not on broker order
                status="FILLED",
            ))
        if layer_2_ticket is not None:
            supabase.log_trade(TradeInput(
                setup_id=setup_id,
                layer_number=2,
                direction=zone.direction,
                order_type="LIMIT",
                mt5_ticket=layer_2_ticket,
                entry_price=None,  # not yet filled
                lot_size=Decimal(str(lot_size)),
                sl_price=Decimal(str(sl_price)),
                tp_price=None,
                status="PENDING",
            ))
        if layer_3_ticket is not None:
            supabase.log_trade(TradeInput(
                setup_id=setup_id,
                layer_number=3,
                direction=zone.direction,
                order_type="LIMIT",
                mt5_ticket=layer_3_ticket,
                entry_price=None,
                lot_size=Decimal(str(lot_size)),
                sl_price=Decimal(str(sl_price)),
                tp_price=None,
                status="PENDING",
            ))
    except Exception as e:
        errors.append(f"trade record write failed: {e}")
        logger.exception(
            "order_manager: trade-record writes failed — orders are "
            "placed on broker but unrecorded; will need reconciliation"
        )


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
        layer_2_ticket=None,
        layer_3_ticket=None,
        layer_1_filled_price=None,
        sl_price=sl_price,
        tp1_price=tp1_price,
        status="FAILED",
        error_messages=list(errors),
    )
