"""Human FVG setups (Claude FTMO, 23 Nov 2026)."""

from __future__ import annotations

import pandas as pd

from bot.strategy.fvg_human import HumanFvgConfig, find_human_setups, scan
from bot.strategy.vertex import VertexConfig

# Yesterday (broker date): the low is 100.
DAY1 = [
    (102.0, 103.0, 101.0, 102.0),
    (102.0, 102.5, 100.0, 101.0),    # yesterday's low 100
    (101.0, 102.0, 100.5, 101.5),
    (101.5, 102.0, 101.0, 101.8),
    (101.8, 102.2, 101.2, 102.0),
    (102.0, 102.4, 101.5, 102.1),
]
# Today from 10:00: a high at 110, a lower high at 108, a drop that
# sweeps yesterday's low (99 < 100), a rally with a gap between candle 9's
# high (101) and candle 11's low (102), and a close above 108.
DAY2 = [
    (106.0, 107.0, 105.5, 106.5),
    (106.5, 110.0, 106.0, 109.0),    # 1: swing high 110 (the TP, the next liquidity)
    (109.0, 109.5, 107.0, 107.5),
    (107.5, 107.5, 105.0, 105.5),
    (105.5, 107.0, 105.0, 106.5),
    (106.5, 108.0, 106.0, 107.5),    # 5: lower high 108 (the one broken)
    (107.5, 107.8, 104.0, 104.5),
    (104.5, 105.0, 102.0, 102.5),
    (102.5, 103.0, 99.0, 99.5),      # 8: the sweep low
    (99.5, 101.0, 99.2, 100.8),      # 9: FVG candle 1 (high 101)
    (100.8, 104.0, 100.5, 103.8),
    (103.8, 108.6, 102.0, 108.4),    # 11: FVG candle 3 (low 102) and the BOS
    (108.4, 108.8, 105.0, 106.0),    # 12: pulls back, nowhere near the gap
]
TOUCH = (106.0, 106.5, 101.3, 104.0)  # reaches the middle of the gap (101.5)
FORMING = (104.0, 104.1, 103.9, 104.0)


def _df(day1, day2) -> pd.DataFrame:
    i1 = pd.date_range("2026-11-19 20:00", periods=len(day1), freq="1min", tz="UTC")
    i2 = pd.date_range("2026-11-20 10:00", periods=len(day2), freq="1min", tz="UTC")
    return pd.DataFrame(list(day1) + list(day2), columns=["open", "high", "low", "close"],
                        index=i1.append(i2))


def _flip(rows):
    return [(200 - o, 200 - l, 200 - h, 200 - c) for o, h, l, c in rows]


DEFAULT = HumanFvgConfig()


def _buys(df, cfg=DEFAULT, based=frozenset()):
    return [s for s in find_human_setups(df, cfg, set(based)) if s.direction == "BUY"]


def test_buy_after_a_sweep_and_a_lower_high_broken() -> None:
    df = _df(DAY1, DAY2)
    (s,) = _buys(df)
    assert s.entry == 101.5                       # middle of the 101-102 gap
    assert (s.fvg_bottom, s.fvg_top) == (101.0, 101.5)   # waits from the middle down
    assert s.sl == 98.5                           # 0.5 under the sweep low (99)
    assert s.tp == 109.7                          # the 110 high, 0.3 early
    assert s.bos_level == 108.0
    assert s.formed_at == df.index[6 + 11]


def test_edge_entry_option() -> None:
    (s,) = _buys(_df(DAY1, DAY2), HumanFvgConfig(entry="edge"))
    assert s.entry == 102.0 and (s.fvg_bottom, s.fvg_top) == (101.0, 102.0)


def test_sell_is_the_mirror() -> None:
    sells = [s for s in find_human_setups(_df(_flip(DAY1), _flip(DAY2)), HumanFvgConfig(), set())
             if s.direction == "SELL"]
    (s,) = sells
    assert (s.entry, s.sl, s.tp) == (98.5, 101.5, 90.3)
    assert (s.fvg_bottom, s.fvg_top) == (98.5, 99.0)


def test_no_sweep_no_trade() -> None:
    day1 = list(DAY1)
    day1[1] = (102.0, 102.5, 98.0, 101.0)         # yesterday's low 98: 99 didn't take it
    assert _buys(_df(day1, DAY2)) == []


def test_low_taken_earlier_today_is_not_a_sweep() -> None:
    day2 = list(DAY2)
    day2[0] = (106.0, 107.0, 99.8, 106.5)         # 100 already taken this morning
    assert _buys(_df(DAY1, day2)) == []


def test_a_higher_high_broken_is_not_a_shift() -> None:
    day2 = list(DAY2)
    day2[5] = (106.5, 110.5, 106.0, 107.5)        # 110.5 is above 110: still an uptrend
    day2[11] = (103.8, 111.0, 102.0, 110.8)
    assert _buys(_df(DAY1, day2)) == []


def test_only_the_first_touch_trades() -> None:
    assert len(_buys(_df(DAY1, DAY2))) == 1
    assert _buys(_df(DAY1, DAY2 + [TOUCH])) == []


def test_gone_if_price_reaches_the_tp_first() -> None:
    assert _buys(_df(DAY1, DAY2 + [(106.0, 109.8, 105.5, 109.0)])) == []


def test_gone_after_four_hours() -> None:
    wait = [DAY2[-1]] * 3
    assert len(_buys(_df(DAY1, DAY2 + wait), HumanFvgConfig(wait_minutes=10))) == 1
    assert _buys(_df(DAY1, DAY2 + wait), HumanFvgConfig(wait_minutes=3)) == []


def test_move_from_a_base_is_left_to_the_zone_rules() -> None:
    assert _buys(_df(DAY1, DAY2), based={("BUY", 6 + 8)}) == []


def test_asia_low_counts_as_liquidity() -> None:
    day1 = list(DAY1)
    day1[1] = (102.0, 102.5, 98.0, 101.0)         # yesterday's low out of reach
    asia = [(103.0, 104.0, 100.0, 103.5)] + [(103.5, 104.0, 103.0, 103.5)] * 4
    ia = pd.date_range("2026-11-20 03:00", periods=len(asia), freq="1min", tz="UTC")
    i1 = pd.date_range("2026-11-19 20:00", periods=len(day1), freq="1min", tz="UTC")
    i2 = pd.date_range("2026-11-20 10:00", periods=len(DAY2), freq="1min", tz="UTC")
    df = pd.DataFrame(day1 + asia + DAY2, columns=["open", "high", "low", "close"],
                      index=i1.append(ia).append(i2))
    (s,) = _buys(df)                               # 99 swept the Asia low (100)
    assert s.sl == 98.5


NO_BASE = VertexConfig(min_base=10, max_base=10)


def test_scan_drops_the_forming_candle() -> None:
    # The last row is still forming: it's where the touch can happen.
    (s,) = [x for x in scan(_df(DAY1, DAY2 + [TOUCH]), HumanFvgConfig(), NO_BASE)
            if x.direction == "BUY"]
    assert s.entry == 101.5
    assert [x for x in scan(_df(DAY1, DAY2 + [TOUCH, FORMING]), HumanFvgConfig(), NO_BASE)
            if x.direction == "BUY"] == []
