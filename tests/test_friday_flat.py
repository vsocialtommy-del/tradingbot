"""Flat before the weekend (FTMO Standard, Tommy 27 Nov 2026): no new
trades on Friday from 20:00 UK, close everything at 21:45 UK."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from bot.profiles import build_profile
from tests.test_main import (  # noqa: F401
    _make_trade_for_death_pass,
    _xautism,
    bot,
    make_setup,
    mock_mt5,
    mock_supabase,
)

FRI = lambda h, m=0: datetime(2026, 11, 27, h, m, tzinfo=timezone.utc)


def _on(b):
    _xautism(b)
    b.config = replace(b.config, friday_no_new_after="20:00", friday_close_at="21:45")


@pytest.mark.parametrize("when, phase", [
    (FRI(19, 59), 0), (FRI(20, 0), 1), (FRI(21, 44), 1), (FRI(21, 45), 2), (FRI(23, 30), 2),
    (datetime(2026, 11, 26, 21, 50, tzinfo=timezone.utc), 0),        # Thursday
    (datetime(2026, 11, 29, 23, 30, tzinfo=timezone.utc), 0),        # Sunday reopen
    (datetime(2026, 6, 26, 19, 30, tzinfo=timezone.utc), 1),         # summer: 20:30 UK (BST)
])
def test_phase_uses_uk_time(bot, when, phase) -> None:  # noqa: F811
    b, _ = bot
    _on(b)
    assert b._friday_phase(when) == phase


def test_off_by_default(bot) -> None:  # noqa: F811
    b, _ = bot
    assert b._friday_phase(FRI(21, 50)) == 0


def test_no_new_trades_friday_evening(bot, mocker) -> None:  # noqa: F811
    b, mgrs = bot
    _on(b)
    b.state.last_config_refresh = FRI(20, 30)
    mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
    pending = mocker.patch.object(b, "_try_place_pending")
    b.run_iteration(FRI(20, 30))
    pending.assert_not_called()


def test_closes_everything_at_2145_once(bot, mock_mt5, mocker) -> None:  # noqa: F811
    b, mgrs = bot
    _on(b)
    b.state.last_config_refresh = FRI(21, 45)
    s = make_setup(status="ACTIVE")
    mgrs["position_tracker"].get_active_setups.return_value = [s]
    filled = _make_trade_for_death_pass(setup_id=s.id, layer_number=1, status="FILLED", mt5_ticket=111)
    waiting = _make_trade_for_death_pass(setup_id=s.id, layer_number=2, status="WAITING", mt5_ticket=222)
    mgrs["position_tracker"].get_trades_for_setup.return_value = [filled, waiting]
    cancel = mocker.patch("bot.main.cancel_layer2_at_broker")
    mocker.patch("bot.main.run_strategy_pipeline", return_value=[])
    b.run_iteration(FRI(21, 45))
    mock_mt5.close_position.assert_called_once_with(111)
    cancel.assert_called_once()
    calls = mgrs["position_tracker"].update_trade_status.call_args_list
    assert any(c.args[:2] == (filled.id, "CLOSED") and c.kwargs["close_reason"] == "MANUAL_CLOSE" for c in calls)
    mock_mt5.close_position.reset_mock()
    b.run_iteration(FRI(21, 46))
    mock_mt5.close_position.assert_not_called()


def test_both_live_bots_have_it() -> None:
    for cfg in (build_profile("m5")[0], build_profile("m1", plan=True)[0]):
        assert (cfg.friday_no_new_after, cfg.friday_close_at) == ("20:00", "21:45")
