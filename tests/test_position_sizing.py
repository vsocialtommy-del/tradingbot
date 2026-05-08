"""Tests for ``bot.risk.position_sizing``.

Test data uses realistic Vantage Gold prices (around $1900). SL
distances are stated in MT5 "points" — for XAUUSD with 2-decimal
quotes, 1 point = $0.01.
"""

from __future__ import annotations

import pytest

from bot.risk.position_sizing import (
    XAUUSD_SPEC,
    InstrumentSpec,
    LotSizeResult,
    SizingConfig,
    SizingMode,
    _floor_to_step,
    calculate_lot_size,
)


# --------------------------------------------------------------------------- #
# FIXED mode — v1 path
# --------------------------------------------------------------------------- #


class TestFixedMode:
    def test_default_returns_001(self) -> None:
        result = calculate_lot_size(
            balance=1000,
            entry_price=1900.00,
            sl_price=1899.70,
        )
        assert result.lot_size == 0.01
        assert result.sizing_mode_used == SizingMode.FIXED
        assert result.warning is None

    def test_ignores_balance(self) -> None:
        # Same fixed lot whether balance is $100 or $100k.
        small = calculate_lot_size(
            balance=100, entry_price=1900.00, sl_price=1899.70
        )
        big = calculate_lot_size(
            balance=100_000, entry_price=1900.00, sl_price=1899.70
        )
        assert small.lot_size == big.lot_size == 0.01

    def test_ignores_sl_distance(self) -> None:
        tight = calculate_lot_size(
            balance=1000, entry_price=1900.00, sl_price=1899.95
        )
        wide = calculate_lot_size(
            balance=1000, entry_price=1900.00, sl_price=1850.00
        )
        assert tight.lot_size == wide.lot_size == 0.01

    def test_implied_risk_calculated_for_audit(self) -> None:
        # Even in FIXED mode we report what the trade actually risks
        # at the given SL — useful for the dashboard's risk-per-trade
        # display.
        result = calculate_lot_size(
            balance=1000,
            entry_price=1900.00,
            sl_price=1899.70,  # 30pt SL
        )
        # 0.01 lots × 30 pts × $1/pt/lot = $0.30
        assert result.calculated_risk_amount == pytest.approx(0.30)
        assert "implied risk" in result.reason.lower()

    def test_custom_fixed_lot(self) -> None:
        result = calculate_lot_size(
            balance=1000,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.FIXED, fixed_lot_size=0.05),
        )
        assert result.lot_size == 0.05


# --------------------------------------------------------------------------- #
# RISK_BASED mode — v1.1 path
# --------------------------------------------------------------------------- #


class TestRiskBasedMode:
    def test_typical_scenario_1k_account(self) -> None:
        # $1,000 × 0.33% = $3.30 budget
        # 30pt SL × $1/pt/lot = $30 per lot
        # raw = $3.30 / $30 = 0.11 lots → fits within bounds
        result = calculate_lot_size(
            balance=1000,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        assert result.lot_size == pytest.approx(0.11)
        assert result.sizing_mode_used == SizingMode.RISK_BASED
        assert result.warning is None
        # Actual risk is `lot × per_lot_risk` = 0.11 × $30 = $3.30
        assert result.calculated_risk_amount == pytest.approx(3.30)

    def test_typical_scenario_10k_account_caps_at_max(self) -> None:
        # $10,000 × 0.33% = $33 budget
        # 30pt SL × $1/pt/lot = $30 per lot
        # raw = $33 / $30 = 1.10 lots → ABOVE default max_lot=1.0
        # → capped to 1.0 with warning
        result = calculate_lot_size(
            balance=10_000,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        assert result.lot_size == 1.0
        assert result.warning is not None
        assert "max" in result.warning.lower()
        # Actual risk at 1.0 lot × $30/lot = $30 (under $33 budget — capped).
        assert result.calculated_risk_amount == pytest.approx(30.0)

    def test_with_higher_max_lot_returns_full_calculation(self) -> None:
        # Same $10k scenario but with max_lot raised — returns 1.10.
        permissive = InstrumentSpec(
            symbol="XAUUSD",
            point_size=0.01,
            point_value_per_lot=1.0,
            lot_step=0.01,
            min_lot=0.01,
            max_lot=5.0,
        )
        result = calculate_lot_size(
            balance=10_000,
            entry_price=1900.00,
            sl_price=1899.70,
            instrument=permissive,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        assert result.lot_size == pytest.approx(1.10)
        assert result.warning is None

    def test_zero_sl_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="differ"):
            calculate_lot_size(
                balance=1000,
                entry_price=1900.00,
                sl_price=1900.00,
                config=SizingConfig(mode=SizingMode.RISK_BASED),
            )

    def test_tiny_balance_caps_to_min_with_warning(self) -> None:
        # $5 × 0.33% = $0.0165
        # raw = $0.0165 / $30 = 0.00055 lots → below min 0.01 → cap up
        result = calculate_lot_size(
            balance=5,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        assert result.lot_size == 0.01
        assert result.warning is not None
        assert "min" in result.warning.lower()
        # At capped 0.01 the actual risk is 0.01 × 30 = $0.30, FAR more
        # than the $0.0165 budget — but that's the cost of having a
        # broker minimum.
        assert result.calculated_risk_amount == pytest.approx(0.30)

    def test_huge_balance_caps_to_max_with_warning(self) -> None:
        # $1M × 0.33% = $3,300
        # raw = $3300 / $30 = 110 lots → way above max 1.0 → cap
        result = calculate_lot_size(
            balance=1_000_000,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        assert result.lot_size == 1.0
        assert result.warning is not None
        assert "max" in result.warning.lower()


# --------------------------------------------------------------------------- #
# Mode switching
# --------------------------------------------------------------------------- #


class TestModeSwitching:
    def test_same_inputs_different_mode_different_outputs(self) -> None:
        common = dict(balance=1000, entry_price=1900.00, sl_price=1899.70)
        fixed = calculate_lot_size(
            **common, config=SizingConfig(mode=SizingMode.FIXED)
        )
        risk_based = calculate_lot_size(
            **common, config=SizingConfig(mode=SizingMode.RISK_BASED)
        )
        # FIXED → 0.01; RISK_BASED → 0.11.
        assert fixed.lot_size == 0.01
        assert risk_based.lot_size == pytest.approx(0.11)
        assert fixed.lot_size != risk_based.lot_size

    def test_unknown_mode_raises(self) -> None:
        # Build a SizingConfig with a mode that's not in the enum.
        # Using dataclass replace trick to bypass the enum type.
        cfg = SizingConfig()
        # Use object.__setattr__ to bypass frozen.
        object.__setattr__(cfg, "mode", "BOGUS")
        with pytest.raises(ValueError, match="unknown sizing mode"):
            calculate_lot_size(
                balance=1000,
                entry_price=1900.00,
                sl_price=1899.70,
                config=cfg,
            )


# --------------------------------------------------------------------------- #
# Lot rounding (floor) behaviour
# --------------------------------------------------------------------------- #


class TestRounding:
    def test_floor_to_lot_step(self) -> None:
        # raw lot 0.055 → floor to 0.05 (lot_step=0.01).
        # $500 × 0.33% = $1.65; $1.65 / $30 = 0.055.
        result = calculate_lot_size(
            balance=500,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        assert result.lot_size == pytest.approx(0.05)

    def test_exact_step_value_unchanged(self) -> None:
        # Inputs that produce an exact 0.10 lot.
        # raw = 0.10 → balance × 0.33% = 0.10 × $30 → balance = $909.09
        result = calculate_lot_size(
            balance=909.09,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        # raw ≈ 0.0999999 → floor to 0.09
        # (boundary case: floor of "essentially 0.10" goes to 0.09)
        # This documents the floor semantics — there's no off-by-one
        # forgiveness on the boundary; pick balance=$910 to get clean 0.10.
        assert result.lot_size == pytest.approx(0.09)

    def test_just_above_step_rounds_down_to_step(self) -> None:
        # raw ≈ 0.1001 → floor to 0.10
        result = calculate_lot_size(
            balance=910,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        # $910 × 0.33% = $3.003; $3.003 / $30 = 0.10010 → floor 0.10
        assert result.lot_size == pytest.approx(0.10)

    def test_floor_to_step_helper_directly(self) -> None:
        # Sanity-check the helper's float-precision robustness.
        assert _floor_to_step(1.10, 0.01) == 1.10
        assert _floor_to_step(1.105, 0.01) == 1.10
        assert _floor_to_step(0.0123, 0.01) == 0.01
        assert _floor_to_step(0.99, 0.01) == 0.99
        # The classic float-trap case: 1.10 / 0.01 == 109.999... in IEEE 754
        # but Decimal gives us 110.
        assert _floor_to_step(1.10, 0.01) != 1.09


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


class TestInputValidation:
    def test_zero_balance_rejected(self) -> None:
        with pytest.raises(ValueError, match="balance"):
            calculate_lot_size(
                balance=0, entry_price=1900.00, sl_price=1899.70
            )

    def test_negative_balance_rejected(self) -> None:
        with pytest.raises(ValueError, match="balance"):
            calculate_lot_size(
                balance=-100, entry_price=1900.00, sl_price=1899.70
            )

    def test_buy_and_sell_sl_orientation_both_work(self) -> None:
        # Distance is computed as abs(), so sl above entry (SELL) and
        # sl below entry (BUY) yield the same lot size.
        buy = calculate_lot_size(
            balance=1000,
            entry_price=1900.00,
            sl_price=1899.70,  # SL below
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        sell = calculate_lot_size(
            balance=1000,
            entry_price=1900.00,
            sl_price=1900.30,  # SL above
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        assert buy.lot_size == sell.lot_size


# --------------------------------------------------------------------------- #
# Reason / audit trail
# --------------------------------------------------------------------------- #


class TestReason:
    def test_fixed_mode_reason_mentions_fixed(self) -> None:
        result = calculate_lot_size(
            balance=1000, entry_price=1900.00, sl_price=1899.70
        )
        assert "FIXED" in result.reason

    def test_risk_based_reason_includes_calculation_breadcrumbs(self) -> None:
        result = calculate_lot_size(
            balance=1000,
            entry_price=1900.00,
            sl_price=1899.70,
            config=SizingConfig(mode=SizingMode.RISK_BASED),
        )
        # Reason should mention balance, risk%, SL distance, raw, rounded.
        assert "$1000.00" in result.reason
        assert "0.33%" in result.reason
        assert "30pts" in result.reason
        assert "raw" in result.reason.lower()
