"""Spike / V BOS + FVG setups (Claude FTMO, 20 Nov 2026)."""

from __future__ import annotations

import pandas as pd

from bot.strategy.fvg_v import FvgVConfig, find_fvg_setups, scan
from bot.strategy.vertex import VertexConfig

# A sharp V down to 92 (no base), a rally with a gap between candle 7's
# high (95.5) and candle 9's low (97), then a close above the 104 swing high.
V_BUY = [
    (100.0, 101.0, 99.5, 100.5),
    (100.5, 102.0, 100.0, 101.5),
    (101.5, 104.0, 101.0, 103.0),   # 2: swing high 104 (the BOS line)
    (103.0, 103.5, 100.0, 100.5),
    (100.5, 101.0, 97.0, 97.5),
    (97.5, 98.0, 94.0, 94.5),
    (94.5, 95.0, 92.0, 92.5),       # 6: the low of the V
    (92.5, 95.5, 92.3, 95.2),       # 7: FVG candle 1 (high 95.5)
    (95.2, 99.0, 95.0, 98.8),
    (98.8, 102.0, 97.0, 101.5),     # 9: FVG candle 3 (low 97)
    (101.5, 105.0, 101.0, 104.6),   # 10: BOS, closes above 104
    (104.6, 105.5, 100.0, 100.5),   # 11: pulls back, doesn't reach 97
]
TOUCH = (100.5, 101.0, 96.5, 98.0)  # reaches into the FVG
FORMING = (100.5, 100.6, 100.4, 100.5)


def _df(rows) -> pd.DataFrame:
    idx = pd.date_range("2026-11-20 10:00", periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def _flip(rows):
    return [(200 - o, 200 - l, 200 - h, 200 - c) for o, h, l, c in rows]


def test_spike_v_buy_setup_m5_levels() -> None:
    closed = _df(V_BUY)
    (s,) = find_fvg_setups(closed, FvgVConfig(sl_mode="low"), set())
    assert s.direction == "BUY"
    assert (s.fvg_bottom, s.fvg_top) == (95.5, 97.0)
    assert s.entry == 97.0          # first touch of the FVG top
    assert s.sl == 91.5             # 0.5 under the low of the V
    assert s.tp == 103.7            # the BOS line, 0.3 early
    assert s.bos_level == 104.0
    assert s.formed_at == closed.index[10]


def test_m1_sl_goes_under_fvg_candle_1() -> None:
    (s,) = find_fvg_setups(_df(V_BUY), FvgVConfig(sl_mode="candle1"), set())
    assert s.sl == 91.8             # 0.5 under candle 7's low (92.3)


def test_sell_is_the_mirror() -> None:
    (s,) = find_fvg_setups(_df(_flip(V_BUY)), FvgVConfig(sl_mode="low"), set())
    assert s.direction == "SELL"
    assert (s.fvg_bottom, s.fvg_top) == (103.0, 104.5)
    assert (s.entry, s.sl, s.tp) == (103.0, 108.5, 96.3)


def test_move_from_a_base_is_left_to_the_zone_rules() -> None:
    # The low candle sits in a base (a W or any other base): no FVG trade.
    assert find_fvg_setups(_df(V_BUY), FvgVConfig(), {("BUY", 6)}) == []
    # A SELL base on the same candle doesn't matter for a BUY.
    assert len(find_fvg_setups(_df(V_BUY), FvgVConfig(), {("SELL", 6)})) == 1


def test_only_the_first_touch_trades() -> None:
    assert len(find_fvg_setups(_df(V_BUY), FvgVConfig(), set())) == 1
    # Once a closed candle has touched it, it's gone.
    assert find_fvg_setups(_df(V_BUY + [TOUCH]), FvgVConfig(), set()) == []


def test_touched_before_the_bos_never_trades() -> None:
    rows = list(V_BUY)
    rows[10] = (101.5, 105.0, 96.8, 104.6)   # the BOS candle itself dips into the FVG
    assert find_fvg_setups(_df(rows), FvgVConfig(), set()) == []


def test_no_bos_no_setup() -> None:
    rows = list(V_BUY[:10]) + [(101.5, 103.9, 101.0, 103.8)]   # closes under 104
    assert find_fvg_setups(_df(rows), FvgVConfig(), set()) == []


def test_bos_line_not_past_the_entry_is_skipped() -> None:
    rows = list(V_BUY)
    rows[2] = (101.5, 97.1, 96.0, 96.5)       # swing high only 97.1: TP 96.8 below entry 97
    rows[0] = (95.0, 95.5, 94.6, 95.0)
    rows[1] = (95.0, 96.0, 94.8, 95.5)
    rows[3] = (96.5, 96.9, 95.5, 96.0)
    rows[4] = (96.0, 96.2, 94.0, 94.5)
    assert all(s.tp > s.entry for s in find_fvg_setups(_df(rows), FvgVConfig(), set()))


def test_dropped_after_max_wait() -> None:
    wait = [V_BUY[-1]] * 3
    assert len(find_fvg_setups(_df(V_BUY + wait), FvgVConfig(max_wait_bars=10), set())) == 1
    assert find_fvg_setups(_df(V_BUY + wait), FvgVConfig(max_wait_bars=4), set()) == []


# The tiny V above is too short to say anything about bases; this vertex
# setting can't find one in it, so the scan tests only test the scan.
NO_BASE = VertexConfig(min_base=10, max_base=10)


def test_scan_drops_the_forming_candle() -> None:
    # The last row is still forming: it's where the touch can happen, so
    # it never counts as an earlier touch.
    (s,) = scan(_df(V_BUY + [TOUCH]), FvgVConfig(), NO_BASE)
    assert s.entry == 97.0
    assert scan(_df(V_BUY + [TOUCH, FORMING]), FvgVConfig(), NO_BASE) == []


def test_same_candles_same_setups() -> None:
    a = scan(_df(V_BUY + [FORMING]), FvgVConfig(), NO_BASE)
    b = scan(_df(V_BUY + [FORMING]), FvgVConfig(), NO_BASE)
    assert a == b and len(a) == 1


def test_leg_in_min_frac_zero_turns_off_only_the_leg_in_rule() -> None:
    cfg = VertexConfig(leg_in_min_frac=0.0)
    assert cfg.leg_min_frac == 1.0 and cfg.leg_in_min_frac == 0.0
    assert VertexConfig().leg_in_min_frac is None
