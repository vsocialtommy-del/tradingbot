"""Claude's plan (19 Nov 2026): the M1 bot also trades M5 zones."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd
import pytest

from bot.plan import DrawdownBreaker, PlanConfig, drawdown_multiplier
from bot.risk.xautism_risk import DailyRisk, RiskConfig, check_risk


# --------------------------------------------------------------------------- #
# Drawdown breaker
# --------------------------------------------------------------------------- #

STEPS = PlanConfig().drawdown_steps


@pytest.mark.parametrize("balance,mult", [
    (10_000, 1.0), (9_700, 1.0), (9_601, 1.0), (9_600, 0.5),
    (9_450, 0.5), (9_400, 0.25), (8_000, 0.25),
])
def test_multiplier_steps(balance, mult) -> None:
    assert drawdown_multiplier(10_000, balance, STEPS) == mult


def test_breaker_starts_from_the_balance_when_the_plan_starts(tmp_path) -> None:
    # The M1 account peaked at ~£10,236 before the plan; the plan starts at
    # £9,446, so it must start at full risk, not a quarter.
    b = DrawdownBreaker(tmp_path / "plan.json", STEPS)
    assert b.update(10013557311, 9_446.0) == 1.0
    assert b.peak == 9_446.0
    assert b.update(10013557311, 9_900.0) == 1.0     # new peak
    assert b.update(10013557311, 9_480.0) == 0.5     # 420 below 9,900
    assert b.update(10013557311, 9_250.0) == 0.25    # 650 below
    assert b.update(10013557311, 9_600.0) == 1.0     # back within 400


def test_breaker_survives_a_restart_and_resets_on_a_new_account(tmp_path) -> None:
    path = tmp_path / "plan.json"
    DrawdownBreaker(path, STEPS).update(111, 10_000.0)
    DrawdownBreaker(path, STEPS).update(111, 10_500.0)
    again = DrawdownBreaker(path, STEPS)
    assert again.update(111, 10_050.0) == 0.5            # peak 10,500 kept
    assert json.loads(path.read_text())["peak"] == 10_500.0
    fresh = DrawdownBreaker(path, STEPS)
    assert fresh.update(222, 10_000.0) == 1.0            # new demo account
    assert fresh.peak == 10_000.0


def test_breaker_keeps_last_value_when_mt5_fails(tmp_path) -> None:
    b = DrawdownBreaker(tmp_path / "plan.json", STEPS, refresh_seconds=0)
    mt5 = MagicMock()
    mt5.get_account_info.return_value = {"login": 1, "balance": 10_000.0}
    assert b.multiplier(mt5) == 1.0
    mt5.get_account_info.return_value = {"login": 1, "balance": 9_500.0}
    assert b.multiplier(mt5) == 0.5
    mt5.get_account_info.side_effect = RuntimeError("no connection")
    assert b.multiplier(mt5) == 0.5


# --------------------------------------------------------------------------- #
# -£200 dead stop with the room check
# --------------------------------------------------------------------------- #

def _daily(total: float) -> DailyRisk:
    return DailyRisk(trading_day="2026-11-19", realised=total, floating=0.0,
                     closed_trades=0, losing_trades=0)


@pytest.mark.parametrize("total,blocked", [
    (0.0, False), (-100.0, False), (-150.0, False), (-150.01, True),
    (-160.0, True), (-199.0, True), (120.0, False),
])
def test_room_check(total, blocked) -> None:
    cfg = RiskConfig(risk_per_trade=50.0, max_daily_loss=200.0, room_check=True,
                     max_losses_per_day=50)
    assert check_risk(_daily(total), cfg).blocked is blocked


def test_room_check_off_is_the_old_rule() -> None:
    cfg = RiskConfig(risk_per_trade=50.0, max_daily_loss=200.0, max_losses_per_day=50)
    assert check_risk(_daily(-199.0), cfg).blocked is False
    assert check_risk(_daily(-200.0), cfg).blocked is True
