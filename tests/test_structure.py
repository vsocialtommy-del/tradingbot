"""Tests for ``bot.strategy.structure``.

Each test follows the pattern:
    1. Build a synthetic OHLC DataFrame from a list of close prices.
    2. Run the structure analysis (or a single helper).
    3. Assert specific swings, labels, BoS events, or trend state.

The synthetic DataFrames have ``open == high == low == close`` because
this module operates only on close prices — OHLC parity keeps the test
data minimal and the assertions trivial to verify by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.structure import (
    BosEvent,
    ClassifiedSwing,
    StructureConfig,
    StructureState,
    Swing,
    _prune_consecutive_same_kind,
    analyze_structure,
    classify_swings,
    detect_bos,
    detect_swings,
    get_swings_within_lookback,
    infer_state,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_df(closes: list[float], start: str = "2026-01-01T00:00:00Z") -> pd.DataFrame:
    """Build a DataFrame with OHLC all equal to ``closes``."""
    times = pd.date_range(start=start, periods=len(closes), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100] * len(closes),
        },
        index=times,
    )


def labels(swings: list[ClassifiedSwing]) -> list[str | None]:
    return [s.label for s in swings]


def kinds(swings: list[Swing]) -> list[str]:
    return [s.kind for s in swings]


def indices(items: list) -> list[int]:
    return [getattr(it, "index", getattr(it, "bar_index", None)) for it in items]


# --------------------------------------------------------------------------- #
# detect_swings — basic cases
# --------------------------------------------------------------------------- #


class TestDetectSwings:
    def test_simple_v_shape_finds_one_low(self) -> None:
        # Down then up — the trough is a swing low.
        df = make_df([10, 8, 6, 8, 10])
        swings = detect_swings(df, strength=2)
        assert len(swings) == 1
        assert swings[0].kind == "LOW"
        assert swings[0].index == 2
        assert swings[0].price == 6.0

    def test_simple_inverted_v_finds_one_high(self) -> None:
        df = make_df([6, 8, 10, 8, 6])
        swings = detect_swings(df, strength=2)
        assert len(swings) == 1
        assert swings[0].kind == "HIGH"
        assert swings[0].index == 2
        assert swings[0].price == 10.0

    def test_strength_zero_rejected(self) -> None:
        df = make_df([1, 2, 3])
        with pytest.raises(ValueError):
            detect_swings(df, strength=0)

    def test_missing_close_column_rejected(self) -> None:
        df = pd.DataFrame({"open": [1, 2, 3]})
        with pytest.raises(ValueError, match="close"):
            detect_swings(df, strength=1)

    def test_flat_window_yields_no_swing(self) -> None:
        # Perfectly flat — no signal.
        df = make_df([5, 5, 5, 5, 5, 5, 5])
        swings = detect_swings(df, strength=2)
        assert swings == []

    def test_leftmost_tie_breaking_for_flat_top(self) -> None:
        # Plateau at the top: only the leftmost peak bar should register.
        # closes: [4, 6, 8, 8, 8, 6, 4]   indices: 0..6
        df = make_df([4, 6, 8, 8, 8, 6, 4])
        swings = detect_swings(df, strength=2)
        # With strength=2, candidates are bars 2..4. Only bar 2 should
        # be the leftmost max in its window.
        highs = [s for s in swings if s.kind == "HIGH"]
        assert len(highs) == 1
        assert highs[0].index == 2


# --------------------------------------------------------------------------- #
# detect_swings — insufficient data
# --------------------------------------------------------------------------- #


class TestInsufficientData:
    def test_empty_df_returns_empty(self) -> None:
        df = make_df([])
        assert detect_swings(df, strength=2) == []

    def test_too_short_returns_empty(self) -> None:
        # strength=3 needs at least 7 bars (2*3+1).
        df = make_df([1, 2, 3, 4, 5, 6])
        assert detect_swings(df, strength=3) == []

    def test_exactly_minimum_bars_finds_one_candidate(self) -> None:
        # 7 bars with strength=3 → only bar 3 is checkable.
        df = make_df([1, 2, 3, 10, 3, 2, 1])
        swings = detect_swings(df, strength=3)
        assert len(swings) == 1
        assert swings[0].index == 3
        assert swings[0].kind == "HIGH"


# --------------------------------------------------------------------------- #
# Trend classification: clean uptrend / downtrend / range
# --------------------------------------------------------------------------- #


# Closes that walk up with regular pullbacks. Each peak is higher than
# the previous; each trough is higher than the previous. Strength=2 picks
# out swings at indices 2, 4, 6, 8, 10, 12, 14, 16.
CLEAN_UPTREND_CLOSES = [
    10, 12, 14, 12, 10, 12, 16, 14, 12, 14, 18, 16, 14, 16, 20, 18, 16, 18, 22, 20,
]

# Mirror of the above.
CLEAN_DOWNTREND_CLOSES = [
    22, 20, 18, 20, 22, 20, 16, 18, 20, 18, 14, 16, 18, 16, 12, 14, 16, 14, 10, 12,
]

# Perfect oscillation between 14 (high) and 10 (low).
RANGE_CLOSES = [
    10, 12, 14, 12, 10, 12, 14, 12, 10, 12, 14, 12, 10, 12, 14, 12, 10,
]


class TestCleanUptrend:
    def test_yields_alternating_hh_hl_pattern(self) -> None:
        df = make_df(CLEAN_UPTREND_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))

        # Expect alternating HIGH/LOW swings starting at index 2.
        assert kinds(snap.swings) == [
            "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW",
        ]
        # First of each kind is unlabelled; the rest should all be HH or HL.
        assert labels(snap.swings) == [None, None, "HH", "HL", "HH", "HL", "HH", "HL"]

    def test_state_is_uptrend(self) -> None:
        df = make_df(CLEAN_UPTREND_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        assert snap.state == StructureState.UPTREND

    def test_last_high_and_low_are_correct(self) -> None:
        df = make_df(CLEAN_UPTREND_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        assert snap.last_swing_high is not None
        assert snap.last_swing_high.price == 20.0
        assert snap.last_swing_low is not None
        assert snap.last_swing_low.price == 16.0


class TestCleanDowntrend:
    def test_yields_alternating_ll_lh_pattern(self) -> None:
        df = make_df(CLEAN_DOWNTREND_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        assert kinds(snap.swings) == [
            "LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH",
        ]
        assert labels(snap.swings) == [None, None, "LL", "LH", "LL", "LH", "LL", "LH"]

    def test_state_is_downtrend(self) -> None:
        df = make_df(CLEAN_DOWNTREND_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        assert snap.state == StructureState.DOWNTREND


class TestRangeBound:
    def test_perfect_oscillation_yields_only_lh_and_hl(self) -> None:
        df = make_df(RANGE_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        # All highs at 14, all lows at 10. Equal-tie convention: equal
        # high → LH, equal low → HL.
        non_none = [s for s in snap.swings if s.label is not None]
        assert all(s.label in ("LH", "HL") for s in non_none)

    def test_state_is_range(self) -> None:
        df = make_df(RANGE_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        # Mixed LH and HL — neither all-HH/HL nor all-LL/LH → RANGE.
        assert snap.state == StructureState.RANGE


# --------------------------------------------------------------------------- #
# Tie / equal-price classification
# --------------------------------------------------------------------------- #


class TestEqualPriceClassification:
    def test_equal_swing_high_classified_as_LH(self) -> None:
        # Two highs at exactly 14, with a low between them.
        # closes: 10, 12, 14, 12, 10, 12, 14, 12, 10  (indices 0..8)
        df = make_df([10, 12, 14, 12, 10, 12, 14, 12, 10])
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        # Highs are at index 2 and 6, both price 14.
        highs = [s for s in snap.swings if s.kind == "HIGH"]
        assert len(highs) == 2
        assert highs[0].label is None  # first high
        assert highs[1].label == "LH"  # equal → LH

    def test_equal_swing_low_classified_as_HL(self) -> None:
        df = make_df([14, 12, 10, 12, 14, 12, 10, 12, 14])
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        lows = [s for s in snap.swings if s.kind == "LOW"]
        assert len(lows) == 2
        assert lows[0].label is None
        assert lows[1].label == "HL"  # equal → HL


# --------------------------------------------------------------------------- #
# BoS detection — reversal cases (the spec's headline scenarios)
# --------------------------------------------------------------------------- #


class TestBosInUptrend:
    """Uptrend, then a close that breaks below a prior swing low."""

    def test_first_bos_after_long_uptrend_is_DOWN_breaking_most_recent_low(
        self,
    ) -> None:
        # 20-bar uptrend then 5 bars closing below the most recent low (16).
        # Last swing low in the uptrend is at index 16, price=16.
        closes = CLEAN_UPTREND_CLOSES + [22, 20, 18, 14, 10]
        df = make_df(closes)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))

        # We must have at least one DOWN BoS event.
        down_events = [e for e in snap.bos_events if e.direction == "DOWN"]
        assert len(down_events) >= 1

        # The FIRST DOWN event in time should break the most recent prior
        # swing low (price=16). Earlier swing lows (12, 14) only get
        # broken later when price closes deeper.
        first_down = down_events[0]
        assert first_down.broken_level == 16.0
        # First close < 16 is bar 23 (close=14).
        assert first_down.bar_index == 23
        assert first_down.break_close == 14.0

    def test_no_bos_up_in_pure_uptrend_data(self) -> None:
        # Without any breakdown, the only BoS events are UPs (continuation).
        df = make_df(CLEAN_UPTREND_CLOSES)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        directions = {e.direction for e in snap.bos_events}
        assert "DOWN" not in directions


class TestBosInDowntrend:
    """Downtrend, then a close that breaks above a prior swing high."""

    def test_first_bos_after_long_downtrend_is_UP_breaking_most_recent_high(
        self,
    ) -> None:
        # 20-bar downtrend then 5 bars closing above the most recent high (16).
        # Last swing high in the downtrend is at index 16, price=16.
        closes = CLEAN_DOWNTREND_CLOSES + [10, 12, 14, 18, 22]
        df = make_df(closes)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))

        up_events = [e for e in snap.bos_events if e.direction == "UP"]
        assert len(up_events) >= 1

        first_up = up_events[0]
        assert first_up.broken_level == 16.0
        # First close > 16 is bar 23 (close=18).
        assert first_up.bar_index == 23
        assert first_up.break_close == 18.0


# --------------------------------------------------------------------------- #
# BoS detection — body confirmation (close, not high/low)
# --------------------------------------------------------------------------- #


class TestBosBodyConfirmation:
    def test_wick_through_swing_does_not_trigger_bos(self) -> None:
        """A bar whose high pokes above the swing but close stays below: no BoS."""
        # Build a 7-bar V with a clear swing high at index 3 (price 10).
        closes = [6, 8, 10, 8, 6, 8, 9.5]  # last close 9.5 — below 10
        df = make_df(closes)
        # Now manually push the LAST bar's high above the swing without
        # changing close.
        df = df.copy()
        df.loc[df.index[-1], "high"] = 11.0  # wick above 10
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        # No UP BoS event because no close exceeded 10.
        assert all(e.direction != "UP" for e in snap.bos_events)

    def test_body_close_past_level_triggers_bos(self) -> None:
        # Symmetric setup but with the last close ABOVE 10.
        closes = [6, 8, 10, 8, 6, 8, 11]
        df = make_df(closes)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        ups = [e for e in snap.bos_events if e.direction == "UP"]
        assert len(ups) == 1
        assert ups[0].broken_level == 10.0
        assert ups[0].break_close == 11.0


# --------------------------------------------------------------------------- #
# Gap-through-pivot edge case
# --------------------------------------------------------------------------- #


class TestGapThroughPivot:
    def test_isolated_spike_high_still_detected(self) -> None:
        # A single bar gaps far above its neighbours — should register as
        # a swing high.
        closes = [10, 10, 10, 50, 10, 10, 10]
        df = make_df(closes)
        swings = detect_swings(df, strength=2)
        # Bars at index 0,1,2,5,6 cannot be checked (out of window range
        # or in flat windows). Only bar 3 has a non-flat window where
        # the center is the max.
        highs = [s for s in swings if s.kind == "HIGH"]
        assert len(highs) == 1
        assert highs[0].index == 3
        assert highs[0].price == 50.0

    def test_isolated_spike_creates_bos(self) -> None:
        # Set up a swing high, then a later bar gaps far above it.
        closes = [6, 8, 10, 8, 6, 8, 10, 8, 6, 100]
        df = make_df(closes)
        snap = analyze_structure(df, StructureConfig(swing_strength=2))
        # The 100-close at bar 9 should register a BoS UP against the
        # swing high at price 10.
        ups = [e for e in snap.bos_events if e.direction == "UP"]
        assert len(ups) >= 1
        assert ups[-1].break_close == 100.0


# --------------------------------------------------------------------------- #
# Pruning behaviour
# --------------------------------------------------------------------------- #


class TestPruning:
    def test_consecutive_highs_collapse_to_highest(self) -> None:
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        swings = [
            Swing(index=2, time=ts, price=10.0, kind="HIGH"),
            Swing(index=4, time=ts, price=12.0, kind="HIGH"),  # higher
            Swing(index=6, time=ts, price=11.0, kind="HIGH"),  # lower → keep prior
            Swing(index=8, time=ts, price=5.0, kind="LOW"),
        ]
        pruned = _prune_consecutive_same_kind(swings)
        assert len(pruned) == 2
        assert pruned[0].price == 12.0  # the highest of the run
        assert pruned[0].index == 4
        assert pruned[1].kind == "LOW"

    def test_consecutive_lows_collapse_to_lowest(self) -> None:
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        swings = [
            Swing(index=2, time=ts, price=20.0, kind="LOW"),
            Swing(index=4, time=ts, price=18.0, kind="LOW"),
            Swing(index=6, time=ts, price=15.0, kind="LOW"),
            Swing(index=8, time=ts, price=17.0, kind="LOW"),
        ]
        pruned = _prune_consecutive_same_kind(swings)
        assert len(pruned) == 1
        assert pruned[0].price == 15.0


# --------------------------------------------------------------------------- #
# infer_state — boundary behaviour
# --------------------------------------------------------------------------- #


class TestInferState:
    def _cs(self, label: str | None) -> ClassifiedSwing:
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        return ClassifiedSwing(
            index=0, time=ts, price=0.0, kind="HIGH", label=label
        )

    def test_too_few_swings_is_range(self) -> None:
        # 3 labelled swings, min=4 → RANGE.
        swings = [self._cs("HH"), self._cs("HL"), self._cs("HH")]
        assert infer_state(swings, min_swings=4) == StructureState.RANGE

    def test_all_hh_hl_is_uptrend(self) -> None:
        swings = [self._cs(l) for l in ("HH", "HL", "HH", "HL")]
        assert infer_state(swings) == StructureState.UPTREND

    def test_all_ll_lh_is_downtrend(self) -> None:
        swings = [self._cs(l) for l in ("LL", "LH", "LL", "LH")]
        assert infer_state(swings) == StructureState.DOWNTREND

    def test_mixed_is_range(self) -> None:
        swings = [self._cs(l) for l in ("HH", "HL", "LH", "LL")]
        assert infer_state(swings) == StructureState.RANGE

    def test_only_uses_last_n_swings(self) -> None:
        # First 4 are downtrend, last 4 are uptrend. State should reflect
        # the most recent regime.
        swings = [
            self._cs(l) for l in ("LL", "LH", "LL", "LH", "HH", "HL", "HH", "HL")
        ]
        assert infer_state(swings, min_swings=4) == StructureState.UPTREND


# --------------------------------------------------------------------------- #
# get_swings_within_lookback — the SL-placement helper
# --------------------------------------------------------------------------- #


class TestSwingsWithinLookback:
    def _swings(self, indices_kinds: list[tuple[int, str]]) -> list[Swing]:
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        return [
            Swing(index=i, time=ts, price=float(i), kind=k)  # type: ignore[arg-type]
            for i, k in indices_kinds
        ]

    def test_keeps_swings_within_window(self) -> None:
        sw = self._swings([(5, "LOW"), (12, "HIGH"), (18, "LOW"), (25, "HIGH")])
        # from_bar=20, lookback=10 → keep swings with 10 < idx <= 20.
        out = get_swings_within_lookback(sw, from_bar=20, lookback=10)
        assert [s.index for s in out] == [12, 18]

    def test_inclusive_at_from_bar(self) -> None:
        sw = self._swings([(20, "HIGH")])
        out = get_swings_within_lookback(sw, from_bar=20, lookback=5)
        assert len(out) == 1
        assert out[0].index == 20

    def test_exclusive_at_from_bar_minus_lookback(self) -> None:
        sw = self._swings([(10, "HIGH"), (11, "LOW")])
        # lookback=10, from_bar=20 → window is (10, 20]. Index 10 is OUT.
        out = get_swings_within_lookback(sw, from_bar=20, lookback=10)
        assert [s.index for s in out] == [11]

    def test_lookback_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            get_swings_within_lookback([], from_bar=10, lookback=0)


# --------------------------------------------------------------------------- #
# classify_swings + detect_bos — direct unit checks
# --------------------------------------------------------------------------- #


class TestClassifyDirect:
    def test_first_of_each_kind_unlabelled(self) -> None:
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        sw = [
            Swing(index=2, time=ts, price=10.0, kind="HIGH"),
            Swing(index=4, time=ts, price=8.0, kind="LOW"),
        ]
        out = classify_swings(sw)
        assert out[0].label is None
        assert out[1].label is None


class TestDetectBosDirect:
    def test_no_swings_no_events(self) -> None:
        df = make_df([1, 2, 3, 4])
        assert detect_bos(df, []) == []

    def test_swing_never_broken_no_event(self) -> None:
        df = make_df([6, 8, 10, 8, 6])  # peak at idx 2, never re-tested above
        swings = detect_swings(df, strength=2)
        events = detect_bos(df, swings)
        assert events == []

    def test_first_break_only_per_swing(self) -> None:
        # Swing high at idx 2 (price 10). Three later closes above it.
        df = make_df([6, 8, 10, 8, 6, 11, 12, 13])
        swings = detect_swings(df, strength=2)
        events = detect_bos(df, swings)
        # Exactly one UP event for the swing — its FIRST break (close 11
        # at bar 5).
        ups = [e for e in events if e.direction == "UP"]
        assert len(ups) == 1
        assert ups[0].bar_index == 5
        assert ups[0].break_close == 11.0


# --------------------------------------------------------------------------- #
# End-to-end integration on the canonical sequences
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_uptrend_snapshot_shape(self) -> None:
        snap = analyze_structure(
            make_df(CLEAN_UPTREND_CLOSES), StructureConfig(swing_strength=2)
        )
        assert snap.state == StructureState.UPTREND
        assert snap.last_swing_high is not None
        assert snap.last_swing_low is not None
        # Continuation BoS UP events expected (each new HH breaks the prior).
        ups = [e for e in snap.bos_events if e.direction == "UP"]
        assert len(ups) >= 3

    def test_downtrend_snapshot_shape(self) -> None:
        snap = analyze_structure(
            make_df(CLEAN_DOWNTREND_CLOSES), StructureConfig(swing_strength=2)
        )
        assert snap.state == StructureState.DOWNTREND
        downs = [e for e in snap.bos_events if e.direction == "DOWN"]
        assert len(downs) >= 3
