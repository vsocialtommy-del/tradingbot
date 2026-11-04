"""Step 7 risk gates: fixed risk sizing, daily loss, two losses and done."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from bot.risk.xautism_risk import (
    DEAL_ENTRY_IN,
    DEAL_ENTRY_OUT,
    DailyRisk,
    DailyRiskTracker,
    RiskConfig,
    check_risk,
    split_lots,
    summarise_day,
    trading_day_start,
)

MAGIC = 123456
USD = lambda e, sl: abs(e - sl) * 100.0  # noqa: E731  $100 per point per lot


class TestSplitLots:
    def test_normal_trade_fills_to_100(self) -> None:
        # Zone 3996-4000 buy, 30% SL = 3994.8 → 5.2 pts → 520 per lot.
        lots = split_lots(risk=100, layer1_entry=4000.0, layer2_entry=4000.0,
                          sl=3994.8, loss_per_lot=USD)
        assert lots.layer1 == 0.01
        assert lots.layer2 == 0.18          # (100 - 5.2) / 520 = 0.182
        assert lots.risk == pytest.approx(98.8)
        assert lots.risk <= 100

    def test_stacked_layer2_priced_from_outer_zone(self) -> None:
        # Sell 4018.0, outer limit 4030.4, SL 4034.69.
        lots = split_lots(risk=100, layer1_entry=4018.0, layer2_entry=4030.4,
                          sl=4034.69, loss_per_lot=USD)
        assert lots.layer2 == 0.19          # (100 - 16.69) / 429
        assert lots.risk <= 100

    def test_cap(self) -> None:
        lots = split_lots(risk=100, layer1_entry=4000.0, layer2_entry=4000.0,
                          sl=3999.9, loss_per_lot=USD, max_lot=2.0)
        assert lots.layer2 == 2.0

    def test_first_touch_alone_over_budget(self) -> None:
        lots = split_lots(risk=1, layer1_entry=4000.0, layer2_entry=4000.0,
                          sl=3990.0, loss_per_lot=USD)
        assert lots.layer2 == 0.0


DAY_START = datetime(2026, 11, 3, 22, 0, tzinfo=timezone.utc)
T_TODAY = DAY_START.timestamp() + 3600
T_YDAY = DAY_START.timestamp() - 3600


def deal(pos, entry, *, profit=0.0, t=T_TODAY, magic=MAGIC, comment=""):
    return {"position_id": pos, "entry": entry, "profit": profit, "time": t,
            "magic": magic, "comment": comment, "commission": 0.0, "swap": 0.0, "fee": 0.0}


class TestSummariseDay:
    def test_two_layers_are_one_trade(self) -> None:
        deals = [
            deal(1, DEAL_ENTRY_IN, comment="bot:L1:s=aaaaaaaa", t=T_YDAY),
            deal(2, DEAL_ENTRY_IN, comment="bot:L2stop:s=aaaaaaaa"),
            deal(1, DEAL_ENTRY_OUT, profit=-6.0, magic=0, comment="[sl]"),
            deal(2, DEAL_ENTRY_OUT, profit=-94.0, magic=0, comment="[sl]"),
            deal(3, DEAL_ENTRY_IN, comment="bot:L1:s=bbbbbbbb"),
            deal(3, DEAL_ENTRY_OUT, profit=40.0, magic=0),
        ]
        d = summarise_day(deals, [], magic=MAGIC, day_start=DAY_START, trading_day="x")
        assert d.realised == -60.0
        assert d.closed_trades == 2
        assert d.losing_trades == 1

    def test_known_tickets_group_when_comments_are_lost(self) -> None:
        deals = [
            deal(1, DEAL_ENTRY_IN, comment=""), deal(2, DEAL_ENTRY_IN, comment=""),
            deal(1, DEAL_ENTRY_OUT, profit=-6.0), deal(2, DEAL_ENTRY_OUT, profit=-94.0),
        ]
        d = summarise_day(deals, [], magic=MAGIC, day_start=DAY_START,
                          trading_day="x", known={1: "cccccccc", 2: "cccccccc"})
        assert d.losing_trades == 1

    def test_open_trade_counts_floating_not_as_closed(self) -> None:
        deals = [deal(1, DEAL_ENTRY_IN, comment="bot:L1:s=aaaaaaaa"),
                 deal(2, DEAL_ENTRY_IN, comment="bot:L2stop:s=aaaaaaaa"),
                 deal(1, DEAL_ENTRY_OUT, profit=-6.0)]
        pos = [{"ticket": 2, "magic": MAGIC, "profit": -50.0, "swap": 0.0, "comment": ""}]
        d = summarise_day(deals, pos, magic=MAGIC, day_start=DAY_START, trading_day="x")
        assert d.floating == -50.0 and d.total == -56.0
        assert d.closed_trades == 0 and d.losing_trades == 0

    def test_yesterdays_closes_and_other_magic_ignored(self) -> None:
        deals = [deal(1, DEAL_ENTRY_IN, t=T_YDAY - 10, comment="s=aaaaaaaa"),
                 deal(1, DEAL_ENTRY_OUT, profit=-100.0, t=T_YDAY),
                 deal(9, DEAL_ENTRY_IN, magic=999), deal(9, DEAL_ENTRY_OUT, profit=-500.0)]
        d = summarise_day(deals, [], magic=MAGIC, day_start=DAY_START, trading_day="x")
        assert d.realised == 0.0 and d.losing_trades == 0


class TestCheckRisk:
    cfg = RiskConfig()

    def test_daily_loss(self) -> None:
        v = check_risk(DailyRisk("x", -250.0, -50.0, 1, 1), self.cfg)
        assert v.blocked and "daily loss" in v.reason

    def test_two_losses(self) -> None:
        v = check_risk(DailyRisk("x", -200.0, 0.0, 2, 2), self.cfg)
        assert v.blocked and "2 losing" in v.reason

    def test_one_loss_is_fine(self) -> None:
        assert not check_risk(DailyRisk("x", -100.0, 0.0, 1, 1), self.cfg).blocked


def test_trading_day_starts_17_00_new_york() -> None:
    now = datetime(2026, 11, 4, 5, 42, tzinfo=timezone.utc)
    assert trading_day_start(now) == datetime(2026, 11, 3, 22, 0, tzinfo=timezone.utc)


class TestTracker:
    def test_blocks_when_history_unreadable(self) -> None:
        mt5 = MagicMock()
        mt5.get_deals.side_effect = RuntimeError("no terminal")
        v = DailyRiskTracker(mt5).check(datetime.now(timezone.utc), RiskConfig())
        assert v.blocked

    def test_caches_between_ticks(self) -> None:
        mt5 = MagicMock()
        mt5.get_deals.return_value = []
        mt5.get_open_positions.return_value = []
        mt5.magic = MAGIC
        t = DailyRiskTracker(mt5, cache_seconds=60)
        now = datetime.now(timezone.utc)
        t.check(now, RiskConfig()); t.check(now, RiskConfig())
        assert mt5.get_deals.call_count == 1
        t.invalidate(); t.check(now, RiskConfig())
        assert mt5.get_deals.call_count == 2

    def test_disabled(self) -> None:
        mt5 = MagicMock()
        assert not DailyRiskTracker(mt5).check(datetime.now(timezone.utc), RiskConfig(enabled=False)).blocked
        mt5.get_deals.assert_not_called()


class TestServerClock:
    """MT5 stamps deals in broker server time (MetaQuotes-Demo is UTC+2).
    A loss closed 30 min ago in real time carries a timestamp 1.5 h in
    the "future" as far as UTC is concerned; it must still count."""

    def _mt5(self, now, offset_h, deals):
        from datetime import timedelta
        mt5 = MagicMock()
        mt5.magic = MAGIC
        mt5.get_open_positions.return_value = []
        mt5.get_current_price.return_value = {
            "bid": 1.0, "ask": 1.0, "time": now + timedelta(hours=offset_h, seconds=5),
        }

        def get_deals(since, until):
            return [d for d in deals if since.timestamp() <= d["time"] <= until.timestamp()]

        mt5.get_deals.side_effect = get_deals
        return mt5

    def _two_losses(self, now, offset_h):
        from datetime import timedelta
        out = []
        for pos, mins in ((1, 90), (2, 30)):
            server_t = (now - timedelta(minutes=mins) + timedelta(hours=offset_h)).timestamp()
            out.append(deal(pos, DEAL_ENTRY_IN, t=server_t - 600))
            out.append(deal(pos, DEAL_ENTRY_OUT, profit=-100.0, t=server_t, magic=0))
        return out

    def test_recent_losses_counted_with_utc_plus_2_server(self) -> None:
        now = datetime(2026, 11, 4, 19, 30, tzinfo=timezone.utc)
        mt5 = self._mt5(now, 2, self._two_losses(now, 2))
        v = DailyRiskTracker(mt5).check(now, RiskConfig())
        assert v.daily.losing_trades == 2
        assert v.blocked

    def test_utc_server_still_works(self) -> None:
        now = datetime(2026, 11, 4, 19, 30, tzinfo=timezone.utc)
        mt5 = self._mt5(now, 0, self._two_losses(now, 0))
        v = DailyRiskTracker(mt5).check(now, RiskConfig())
        assert v.daily.losing_trades == 2

    def test_day_boundary_in_server_time(self) -> None:
        # Loss closed at 16:30 New York (before the 17:00 reset) belongs
        # to yesterday even though its server timestamp reads 23:30.
        from datetime import timedelta
        now = datetime(2026, 11, 4, 23, 0, tzinfo=timezone.utc)   # 18:00 NY
        start = trading_day_start(now)
        before_reset = (start - timedelta(minutes=30) + timedelta(hours=2)).timestamp()
        deals = [deal(1, DEAL_ENTRY_IN, t=before_reset - 600),
                 deal(1, DEAL_ENTRY_OUT, profit=-100.0, t=before_reset, magic=0)]
        mt5 = self._mt5(now, 2, deals)
        v = DailyRiskTracker(mt5).check(now, RiskConfig())
        assert v.daily.losing_trades == 0

    def test_stale_tick_keeps_last_offset(self) -> None:
        from datetime import timedelta
        now = datetime(2026, 11, 4, 19, 30, tzinfo=timezone.utc)
        mt5 = self._mt5(now, 2, [])
        t = DailyRiskTracker(mt5, cache_seconds=0)
        t.check(now, RiskConfig())
        assert t._offset == timedelta(hours=2)
        # Weekend: last tick is days old -> ignore, keep +2h.
        mt5.get_current_price.return_value = {"bid": 1, "ask": 1, "time": now - timedelta(days=2)}
        t.check(now, RiskConfig())
        assert t._offset == timedelta(hours=2)
