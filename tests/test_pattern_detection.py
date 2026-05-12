"""Tests for ``bot.strategy.pattern_detection`` — RBR/DBD/DBR/RBD.

The S&D methodology pivot (PR #31). Tests are organised by stage:
impulse detection → base detection → classification → top-level
``detect_patterns``.

Test fixtures use bar-by-bar synthetic OHLC where each scenario is
designed to produce (or not produce) a specific outcome at one of
the three stages. Strict control over open/high/low/close gives
exact verdicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.strategy.pattern_detection import (
    Base,
    Impulse,
    Pattern,
    PatternConfig,
    PatternType,
    _atr,
    classify_patterns,
    detect_bases,
    detect_impulses,
    detect_patterns,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_ohlc(
    opens: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    closes: list[float] | None = None,
    start: str = "2026-01-01T00:00:00Z",
) -> pd.DataFrame:
    """Build a DataFrame from explicit OHLC columns.

    Defaults: if ``highs``/``lows``/``closes`` are None, the bar is a
    doji (high = low = close = open).
    """
    n = len(opens)
    closes = list(closes) if closes is not None else list(opens)
    highs = list(highs) if highs is not None else [max(o, c) for o, c in zip(opens, closes)]
    lows = list(lows) if lows is not None else [min(o, c) for o, c in zip(opens, closes)]
    times = pd.date_range(start=start, periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [100] * n},
        index=times,
    )


def make_strong_bar(
    direction: str, open_: float, body: float, wick: float = 0.5,
) -> tuple[float, float, float, float]:
    """OHLC for a single strong-bodied bar.

    body = absolute price change; direction sets sign. Tight wicks
    so the body/range ratio is well above the 0.6 threshold.
    """
    close = open_ + body if direction == "RALLY" else open_ - body
    high = max(open_, close) + wick
    low = min(open_, close) - wick
    return open_, high, low, close


def build_strong_run(
    direction: str, start_price: float, n_bars: int,
    body_per_bar: float = 5.0, wick: float = 0.3,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """A run of ``n_bars`` strong same-direction bars."""
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    price = start_price
    for _ in range(n_bars):
        o, h, l, c = make_strong_bar(direction, price, body_per_bar, wick)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        price = c
    return opens, highs, lows, closes


def quiet_bar(price: float, body: float = 0.1, wick: float = 0.2) -> tuple[float, float, float, float]:
    """A small-bodied bar at ``price``. Used to seed quiet windows
    before/between/after impulses so ATR is well-defined and small."""
    return price, price + body / 2 + wick, price - body / 2 - wick, price + body / 2 - body / 2


def quiet_prelude(price: float, n: int = 20, body: float = 0.1) -> tuple[list[float], list[float], list[float], list[float]]:
    opens, highs, lows, closes = [], [], [], []
    for _ in range(n):
        o, h, l, c = quiet_bar(price, body=body)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
    return opens, highs, lows, closes


# --------------------------------------------------------------------------- #
# ATR helper
# --------------------------------------------------------------------------- #


class TestAtr:
    def test_atr_grows_with_period_history(self) -> None:
        # First (period-1) entries must be NaN.
        closes = np.array([100.0] * 20)
        highs = closes + 0.5
        lows = closes - 0.5
        atr = _atr(highs, lows, closes, period=14)
        assert np.isnan(atr[:13]).all()
        # First defined value at index period-1.
        assert np.isfinite(atr[13])
        # On a fully flat series ATR is ~1.0 (the bar range).
        assert atr[13] == pytest.approx(1.0)

    def test_atr_short_history_returns_all_nan(self) -> None:
        closes = np.array([100.0, 101.0])
        highs = closes + 0.5
        lows = closes - 0.5
        atr = _atr(highs, lows, closes, period=14)
        assert np.isnan(atr).all()


# --------------------------------------------------------------------------- #
# Stage 1 — impulse detection
# --------------------------------------------------------------------------- #


class TestDetectImpulses:
    def test_single_strong_bullish_candle_is_impulse(self) -> None:
        # 20 quiet bars (ATR seed) then one strong rally.
        opens, highs, lows, closes = quiet_prelude(100.0, n=20)
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        assert len(impulses) == 1
        assert impulses[0].direction == "RALLY"
        assert impulses[0].start_index == 20
        assert impulses[0].end_index == 20
        assert impulses[0].candle_count == 1

    def test_single_strong_bearish_candle_is_impulse(self) -> None:
        opens, highs, lows, closes = quiet_prelude(100.0)
        o, h, l, c = make_strong_bar("DROP", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        assert len(impulses) == 1
        assert impulses[0].direction == "DROP"

    def test_wick_heavy_candle_is_not_impulse(self) -> None:
        # Body 0.5, wick 5.0 → body/range = 0.5/10.5 ≈ 0.05 << 0.6.
        opens, highs, lows, closes = quiet_prelude(100.0)
        opens.append(100.0)
        closes.append(100.5)
        highs.append(105.5)
        lows.append(99.5)
        df = make_ohlc(opens, highs, lows, closes)
        assert detect_impulses(df) == []

    def test_small_body_in_low_vol_not_impulse(self) -> None:
        # All bars have ATR ≈ 0.5. A bar with body 0.3 fails the
        # body ≥ 1.0 × ATR check even with good body/range ratio.
        opens, highs, lows, closes = quiet_prelude(100.0, body=0.5)
        # Bar with body 0.3 (below ATR 0.5) and wick 0 (good ratio).
        opens.append(100.0); closes.append(100.3)
        highs.append(100.3); lows.append(100.0)
        df = make_ohlc(opens, highs, lows, closes)
        assert detect_impulses(df) == []

    def test_three_bar_rally_is_single_impulse(self) -> None:
        opens, highs, lows, closes = quiet_prelude(100.0)
        run_o, run_h, run_l, run_c = build_strong_run("RALLY", 100.0, n_bars=3, body_per_bar=5.0)
        opens += run_o; highs += run_h; lows += run_l; closes += run_c
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        assert len(impulses) == 1
        assert impulses[0].candle_count == 3
        # Range = high of last bar - low of first bar.
        # First bar: open 100, close 105, low ~99.7, high ~105.3
        # Last bar:  open 110, close 115, low ~109.7, high ~115.3
        # Range ≈ 115.3 - 99.7 ≈ 15.6
        assert impulses[0].range_size == pytest.approx(15.6, abs=0.5)

    def test_run_breaks_at_direction_flip(self) -> None:
        # 2-bar rally, then 1-bar drop, then 2-bar rally → 3 separate impulses.
        opens, highs, lows, closes = quiet_prelude(100.0)
        for direction, n_bars in [("RALLY", 2), ("DROP", 1), ("RALLY", 2)]:
            run_o, run_h, run_l, run_c = build_strong_run(
                direction, closes[-1] if closes else 100.0, n_bars=n_bars,
            )
            opens += run_o; highs += run_h; lows += run_l; closes += run_c
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        assert len(impulses) == 3
        directions = [imp.direction for imp in impulses]
        assert directions == ["RALLY", "DROP", "RALLY"]

    def test_six_consecutive_bars_capped_at_max_run(self) -> None:
        opens, highs, lows, closes = quiet_prelude(100.0)
        run_o, run_h, run_l, run_c = build_strong_run("RALLY", 100.0, n_bars=6)
        opens += run_o; highs += run_h; lows += run_l; closes += run_c
        df = make_ohlc(opens, highs, lows, closes)
        cfg = PatternConfig(max_impulse_run_candles=5)
        impulses = detect_impulses(df, cfg)
        # First 5 bars are one impulse; bar 6 starts a new one.
        assert len(impulses) == 2
        assert impulses[0].candle_count == 5
        assert impulses[1].candle_count == 1

    def test_insufficient_atr_history_no_impulses(self) -> None:
        # Only 10 bars — less than the default ATR period of 14.
        run_o, run_h, run_l, run_c = build_strong_run("RALLY", 100.0, n_bars=10)
        df = make_ohlc(run_o, run_h, run_l, run_c)
        assert detect_impulses(df) == []

    def test_empty_df_returns_empty(self) -> None:
        df = make_ohlc([])
        assert detect_impulses(df) == []

    def test_doji_at_start_of_window_not_part_of_run(self) -> None:
        # A doji (body=0) at the start of what would otherwise be a
        # rally run isn't strong → doesn't extend the run backwards.
        opens, highs, lows, closes = quiet_prelude(100.0)
        # Doji bar.
        opens.append(100.0); closes.append(100.0)
        highs.append(100.5); lows.append(99.5)
        # Then a real strong bar.
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        assert len(impulses) == 1
        assert impulses[0].candle_count == 1
        assert impulses[0].start_index == 21  # not 20 (the doji)


# --------------------------------------------------------------------------- #
# Stage 2 — base detection
# --------------------------------------------------------------------------- #


class TestDetectBases:
    @pytest.fixture
    def simple_rbr_setup(self) -> pd.DataFrame:
        """20 quiet bars + rally-base-rally with 3-bar base.

        Structure:
          bars 0..19  prelude (quiet at 100)
          bar  20     rally (open 100, close 105)
          bar  21..23 base (3 bars near 105, small bodies)
          bar  24     rally (open 105, close 110)
        """
        opens, highs, lows, closes = quiet_prelude(100.0)
        # Rally 1
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        # Base — 3 quiet bars near 105
        for _ in range(3):
            opens.append(105.0); closes.append(105.0)
            highs.append(105.3); lows.append(104.7)
        # Rally 2
        o, h, l, c = make_strong_bar("RALLY", 105.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        return make_ohlc(opens, highs, lows, closes)

    def test_base_between_two_impulses(self, simple_rbr_setup: pd.DataFrame) -> None:
        df = simple_rbr_setup
        impulses = detect_impulses(df)
        assert len(impulses) == 2
        bases = detect_bases(df, impulses)
        assert len(bases) == 1
        base = bases[0]
        assert base.candle_count == 3
        assert base.start_index == 21
        assert base.end_index == 23
        # Wick-inclusive envelope: highs = 105.3, lows = 104.7.
        assert base.top == pytest.approx(105.3)
        assert base.bottom == pytest.approx(104.7)

    def test_one_bar_base_valid(self) -> None:
        opens, highs, lows, closes = quiet_prelude(100.0)
        # Rally 1
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        # 1-bar base
        opens.append(105.0); closes.append(105.0)
        highs.append(105.2); lows.append(104.8)
        # Rally 2
        o, h, l, c = make_strong_bar("RALLY", 105.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        bases = detect_bases(df, impulses)
        assert len(bases) == 1
        assert bases[0].candle_count == 1

    def test_gap_longer_than_max_no_base(self) -> None:
        # 6-bar gap > max_base_candles=5.
        opens, highs, lows, closes = quiet_prelude(100.0)
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        for _ in range(6):
            opens.append(105.0); closes.append(105.0)
            highs.append(105.1); lows.append(104.9)
        o, h, l, c = make_strong_bar("RALLY", 105.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        bases = detect_bases(df, impulses)
        assert bases == []

    def test_zero_gap_no_base(self) -> None:
        # Adjacent impulses (no bars between) → no base.
        opens, highs, lows, closes = quiet_prelude(100.0)
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        # Direction flip immediately — no base bars in between.
        o, h, l, c = make_strong_bar("DROP", 105.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        assert len(impulses) == 2
        assert detect_bases(df, impulses) == []

    def test_base_too_wide_relative_to_impulse_rejected(self) -> None:
        # Base range = 4.0, impulses range = 5.0 → 4.0 > 0.6 * 5.0 = 3.0 → reject.
        opens, highs, lows, closes = quiet_prelude(100.0)
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        # Wide base — closes drift from 105 to 109 (range 4.0).
        for c_price in (105.0, 107.0, 109.0):
            opens.append(c_price); closes.append(c_price)
            highs.append(c_price + 0.1); lows.append(c_price - 0.1)
        o, h, l, c = make_strong_bar("RALLY", 109.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        impulses = detect_impulses(df)
        bases = detect_bases(df, impulses)
        assert bases == []

    # Note: the "big body in base" criterion is hard to test in
    # isolation because a base bar with body ≥ 0.4× the impulse body
    # tends to be large enough to qualify as its own impulse (fails
    # the body ≥ ATR check the other way) — at which point the gap
    # isn't a single base any more, it's broken into multiple
    # impulses. The criterion still acts as an extra safety in the
    # ``_tight_enough`` helper for unusual cases (e.g. a low-ATR
    # session with a sudden mid-range bar) but doesn't make for a
    # clean black-box test. Exercised implicitly via the
    # ``test_base_too_wide_relative_to_impulse_rejected`` scenario.


# --------------------------------------------------------------------------- #
# Stage 3 — classification
# --------------------------------------------------------------------------- #


class TestClassifyPatterns:
    def test_rbr_classifies_as_buy(self) -> None:
        # Build an RBR scenario via the full pipeline.
        opens, highs, lows, closes = quiet_prelude(100.0)
        for _ in range(2):
            o, h, l, c = make_strong_bar("RALLY", closes[-1] if closes else 100.0, body=5.0)
            opens.append(o); highs.append(h); lows.append(l); closes.append(c)
            for _ in range(3):
                opens.append(closes[-1]); closes.append(closes[-1])
                highs.append(closes[-1] + 0.1); lows.append(closes[-1] - 0.1)
        # Trim to leave one base + final rally (so we have rally → base → rally).
        df = make_ohlc(opens, highs, lows, closes)
        patterns = detect_patterns(df)
        # The two rallies + base in middle is an RBR.
        rbrs = [p for p in patterns if p.pattern_type == PatternType.RBR]
        assert rbrs
        for p in rbrs:
            assert p.direction == "BUY"

    def test_dbd_classifies_as_sell(self) -> None:
        opens, highs, lows, closes = quiet_prelude(100.0)
        for _ in range(2):
            o, h, l, c = make_strong_bar("DROP", closes[-1] if closes else 100.0, body=5.0)
            opens.append(o); highs.append(h); lows.append(l); closes.append(c)
            for _ in range(3):
                opens.append(closes[-1]); closes.append(closes[-1])
                highs.append(closes[-1] + 0.1); lows.append(closes[-1] - 0.1)
        df = make_ohlc(opens, highs, lows, closes)
        patterns = detect_patterns(df)
        dbds = [p for p in patterns if p.pattern_type == PatternType.DBD]
        assert dbds
        for p in dbds:
            assert p.direction == "SELL"

    def test_dbr_classifies_as_buy(self) -> None:
        # Drop → base → rally
        opens, highs, lows, closes = quiet_prelude(100.0)
        # Drop
        o, h, l, c = make_strong_bar("DROP", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        # Base
        for _ in range(3):
            opens.append(95.0); closes.append(95.0)
            highs.append(95.1); lows.append(94.9)
        # Rally
        o, h, l, c = make_strong_bar("RALLY", 95.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        patterns = detect_patterns(df)
        dbrs = [p for p in patterns if p.pattern_type == PatternType.DBR]
        assert len(dbrs) == 1
        assert dbrs[0].direction == "BUY"

    def test_rbd_classifies_as_sell(self) -> None:
        opens, highs, lows, closes = quiet_prelude(100.0)
        # Rally
        o, h, l, c = make_strong_bar("RALLY", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        # Base
        for _ in range(3):
            opens.append(105.0); closes.append(105.0)
            highs.append(105.1); lows.append(104.9)
        # Drop
        o, h, l, c = make_strong_bar("DROP", 105.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        patterns = detect_patterns(df)
        rbds = [p for p in patterns if p.pattern_type == PatternType.RBD]
        assert len(rbds) == 1
        assert rbds[0].direction == "SELL"

    def test_classify_returns_empty_for_no_bases(self) -> None:
        assert classify_patterns([], []) == []

    def test_formed_at_equals_impulse_after_end_time(self) -> None:
        opens, highs, lows, closes = quiet_prelude(100.0)
        o, h, l, c = make_strong_bar("DROP", 100.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        for _ in range(3):
            opens.append(95.0); closes.append(95.0)
            highs.append(95.1); lows.append(94.9)
        o, h, l, c = make_strong_bar("RALLY", 95.0, body=5.0)
        opens.append(o); highs.append(h); lows.append(l); closes.append(c)
        df = make_ohlc(opens, highs, lows, closes)
        patterns = detect_patterns(df)
        assert patterns
        p = patterns[0]
        assert p.formed_at == p.impulse_after.end_time


# --------------------------------------------------------------------------- #
# Top-level detect_patterns
# --------------------------------------------------------------------------- #


class TestDetectPatternsTopLevel:
    def test_empty_df(self) -> None:
        assert detect_patterns(pd.DataFrame()) == []

    def test_short_df(self) -> None:
        df = make_ohlc([100.0] * 5)
        assert detect_patterns(df) == []


# --------------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------------- #


class TestPatternConfigDefaults:
    def test_defaults(self) -> None:
        c = PatternConfig()
        assert c.impulse_body_to_range_ratio_min == 0.6
        assert c.impulse_atr_multiple_min == 1.0
        assert c.atr_period == 14
        assert c.max_impulse_run_candles == 5
        assert c.min_base_candles == 1
        assert c.max_base_candles == 5
        assert c.base_range_to_impulse_ratio_max == 0.6
        assert c.base_max_body_to_impulse_body_ratio == 0.4
        assert c.lookback_bars == 50
