"""Phone dashboard support: closed trades, closest zones, status, live risk."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from bot.dashboard.bridge import (
    DashboardBridge,
    ZoneView,
    build_closed_trades,
    pick_closest_zones,
)
from bot.profiles import build_profile, live_risk, risk_label

MAGIC = 777
OFFSET = timedelta(hours=2)  # MetaQuotes-Demo server clock = UTC+2


def _ts(utc: datetime) -> float:
    """Deal time as MT5 stamps it (server clock as epoch seconds)."""
    return (utc + OFFSET).timestamp()


def _deal(pos, entry, *, t, price, volume, type_=0, profit=0.0, magic=MAGIC,
          comment="", commission=0.0):
    return {"position_id": pos, "entry": entry, "time": _ts(t), "price": price,
            "volume": volume, "type": type_, "profit": profit, "magic": magic,
            "comment": comment, "commission": commission, "swap": 0.0, "fee": 0.0}


T0 = datetime(2026, 11, 11, 9, 2, tzinfo=timezone.utc)


class TestClosedTrades:
    def test_layers_of_one_setup_share_a_key(self) -> None:
        deals = [
            _deal(1, 0, t=T0, price=4057.04, volume=0.01, comment="bot:L1 s=abcdef12"),
            _deal(2, 0, t=T0, price=4056.93, volume=0.29, comment="bot:L2stop s=abcdef12"),
            _deal(1, 1, t=T0 + timedelta(minutes=12), price=4054.62, volume=0.01,
                  type_=1, profit=-1.82),
            _deal(2, 1, t=T0 + timedelta(minutes=12), price=4054.62, volume=0.29,
                  type_=1, profit=-50.30),
        ]
        rows = build_closed_trades(deals, bot="M1", magic=MAGIC, server_offset=OFFSET)
        assert [r["setup_key"] for r in rows] == ["abcdef12", "abcdef12"]
        assert sum(r["profit"] for r in rows) == pytest.approx(-52.12)
        r = rows[0]
        assert r["direction"] == "BUY" and r["bot"] == "M1"
        # server time turned back into real UTC
        assert r["open_time"] == T0.isoformat()
        assert r["trading_day"] == "2026-11-11"

    def test_open_or_part_closed_positions_left_out(self) -> None:
        deals = [
            _deal(5, 0, t=T0, price=4000.0, volume=0.30),
            _deal(5, 1, t=T0 + timedelta(minutes=1), price=4001.0, volume=0.10, type_=1, profit=10),
            _deal(6, 0, t=T0, price=4000.0, volume=0.10),
        ]
        assert build_closed_trades(deals, bot="M5", magic=MAGIC, server_offset=OFFSET) == []

    def test_other_magic_ignored_and_known_ticket_wins(self) -> None:
        deals = [
            _deal(7, 0, t=T0, price=4000.0, volume=0.01, magic=1),
            _deal(7, 1, t=T0, price=4001.0, volume=0.01, type_=1, profit=1, magic=1),
            _deal(8, 0, t=T0, price=4000.0, volume=0.01, type_=1, comment="rewritten"),
            _deal(8, 1, t=T0, price=3999.0, volume=0.01, type_=0, profit=1),
        ]
        rows = build_closed_trades(deals, bot="M5", magic=MAGIC, server_offset=OFFSET,
                                   known={8: "1234abcd"})
        assert len(rows) == 1
        assert rows[0]["setup_key"] == "1234abcd" and rows[0]["direction"] == "SELL"

    def test_trading_day_rolls_at_5pm_new_york(self) -> None:
        # 22:30 UK in November = after the 17:00 New York reset -> next day.
        late = datetime(2026, 11, 11, 22, 30, tzinfo=timezone.utc)
        deals = [
            _deal(9, 0, t=late, price=4000.0, volume=0.01),
            _deal(9, 1, t=late + timedelta(minutes=5), price=4001.0, volume=0.01,
                  type_=1, profit=1.0),
        ]
        rows = build_closed_trades(deals, bot="M5", magic=MAGIC, server_offset=OFFSET)
        assert rows[0]["trading_day"] == "2026-11-12"

    def test_commission_counts(self) -> None:
        deals = [
            _deal(3, 0, t=T0, price=4000.0, volume=0.10, commission=-0.70),
            _deal(3, 1, t=T0, price=4001.0, volume=0.10, type_=1, profit=10.0, commission=-0.70),
        ]
        rows = build_closed_trades(deals, bot="M5", magic=MAGIC, server_offset=OFFSET)
        assert rows[0]["profit"] == pytest.approx(8.60)


class TestClosestZones:
    def test_picks_nearest_each_side(self) -> None:
        zs = [
            ZoneView("SELL", 4119.34, 4114.58, 4114.58, 4120.8, 4110.0, 77.0),
            ZoneView("SELL", 4177.01, 4172.61, 4172.61, None, None, 135.0),
            ZoneView("BUY", 4020.11, 4016.90, 4020.11, 4015.9, 4021.3, 17.4),
        ]
        buy, sell = pick_closest_zones(zs)
        assert buy.top == 4020.11 and sell.bottom == 4114.58
        assert sell.as_dict()["distance"] == 77.0

    def test_none_when_missing(self) -> None:
        assert pick_closest_zones([]) == (None, None)


class TestBridge:
    def test_missing_tables_warn_once_and_back_off(self, mocker: MockerFixture) -> None:
        sb = MagicMock()
        sb.upsert_rows.side_effect = Exception(
            "{'code': 'PGRST205', 'message': \"Could not find the table 'public.bot_status'\"}"
        )
        warn = mocker.patch("bot.dashboard.bridge.logger.warning")
        br = DashboardBridge(sb, "M5")
        now = datetime.now(timezone.utc)
        assert br.write_status({"a": 1}, now) is False
        assert br.write_status({"a": 1}, now) is False  # backing off, no call
        assert sb.upsert_rows.call_count == 1
        assert warn.call_count == 1
        assert br.sync_trades([{"position_id": 1}]) == 0

    def test_status_row_shape(self) -> None:
        sb = MagicMock()
        br = DashboardBridge(sb, "M1")
        now = datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
        assert br.write_status({"x": 1}, now)
        table, rows = sb.upsert_rows.call_args.args
        assert table == "bot_status"
        assert rows == [{"bot": "M1", "updated_at": now.isoformat(), "data": {"x": 1}}]
        assert sb.upsert_rows.call_args.kwargs == {"on_conflict": "bot"}

    def test_trades_sent_once(self) -> None:
        sb = MagicMock()
        br = DashboardBridge(sb, "M5")
        rows = [{"position_id": 1}, {"position_id": 2}]
        assert br.sync_trades(rows) == 2
        assert br.sync_trades(rows) == 0
        assert br.sync_trades(rows + [{"position_id": 3}]) == 1


class TestLiveRisk:
    def test_nothing_set_keeps_bot_settings(self) -> None:
        base = build_profile("m5")[0].risk
        assert live_risk(base) is base
        assert live_risk(base, "", "") is base

    def test_m5_can_switch_to_percent(self) -> None:
        base = build_profile("m5")[0].risk
        r = live_risk(base, "0.5%", 4)
        assert r.risk_pct == pytest.approx(0.005)
        assert r.max_losses_per_day == 4
        assert r.max_daily_loss_pct == pytest.approx(0.025)

    def test_losses_only_keeps_risk_and_moves_cap(self) -> None:
        base = build_profile("m5")[0].risk          # £100, cap £300
        r = live_risk(base, None, 3)
        assert r.risk_per_trade == 100.0 and r.risk_pct is None
        assert r.max_daily_loss == 400.0 and r.max_losses_per_day == 3
        m1 = build_profile("m1", "1%")[0].risk
        r = live_risk(m1, None, 4)
        assert r.risk_pct == pytest.approx(0.01) and r.max_daily_loss_pct == pytest.approx(0.05)

    def test_fixed_amount(self) -> None:
        base = build_profile("m1", "1%")[0].risk
        r = live_risk(base, "150", None)
        assert r.risk_pct is None and r.risk_per_trade == 150.0 and r.max_daily_loss == 450.0

    @pytest.mark.parametrize("risk,losses", [("lots", None), ("50%", None), (None, "0")])
    def test_bad_values_raise(self, risk, losses) -> None:
        with pytest.raises(ValueError):
            live_risk(build_profile("m5")[0].risk, risk, losses)

    def test_labels(self) -> None:
        assert risk_label(build_profile("m5")[0].risk) == "100"
        assert risk_label(build_profile("m1", "0.5%")[0].risk) == "0.5%"
