"""Tests for ``bot.strategy.zone_lifecycle``.

Pure-logic tests (no Supabase, no MT5). The orchestrator-level wiring
(per-bar pass + dedup pre-flight) is exercised in ``test_main.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bot.strategy.structure import StructureConfig, Swing
from bot.strategy.zone_lifecycle import (
    SKIP_NEW_SETUP_STATUSES,
    TERMINAL_ZONE_STATUSES,
    FlipResult,
    IllegalZoneTransitionError,
    ZoneRef,
    _VALID_ZONE_TRANSITIONS,
    _nearest_bos_target,
    check_consumption,
    check_flip,
    check_violation,
    flipped_zone_body_broken_since_flip,
    validate_zone_transition,
    zone_bounds_overlap,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def buy_zone(top: float = 105.0, bottom: float = 100.0) -> ZoneRef:
    return ZoneRef(direction="BUY", top=top, bottom=bottom)


def sell_zone(top: float = 105.0, bottom: float = 100.0) -> ZoneRef:
    return ZoneRef(direction="SELL", top=top, bottom=bottom)


def make_bar_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Each row: (open, high, low, close). Sequential UTC index from 2026-01-01."""
    times = pd.date_range(
        "2026-01-01T00:00:00Z", periods=len(rows), freq="5min", tz="UTC",
    )
    return pd.DataFrame(
        {
            "open":  [r[0] for r in rows],
            "high":  [r[1] for r in rows],
            "low":   [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": [100] * len(rows),
        },
        index=times,
    )


# --------------------------------------------------------------------------- #
# Transition table
# --------------------------------------------------------------------------- #


class TestValidateZoneTransition:
    @pytest.mark.parametrize(
        "current,new",
        [
            ("CONFIRMED", "ACTIVE"),
            ("CONFIRMED", "CONSUMED"),
            ("CONFIRMED", "VIOLATED"),
            ("ACTIVE", "CONSUMED"),
            ("ACTIVE", "VIOLATED"),
            ("CONSUMED", "VIOLATED"),
            ("VIOLATED", "FLIPPED"),
            # PR #38: FLIPPED zones become tradeable in flipped_direction.
            ("FLIPPED", "ACTIVE"),
        ],
    )
    def test_valid_transitions(self, current: str, new: str) -> None:
        validate_zone_transition(current, new)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "current,new",
        [
            ("CONFIRMED", "FLIPPED"),    # must go via VIOLATED
            ("ACTIVE", "CONFIRMED"),     # no backwards
            ("ACTIVE", "FLIPPED"),       # must go via VIOLATED
            ("CONSUMED", "ACTIVE"),      # Q3: no re-arming
            ("CONSUMED", "CONFIRMED"),   # Q3: no re-arming
            ("CONSUMED", "FLIPPED"),     # must go via VIOLATED
            ("VIOLATED", "CONSUMED"),    # backwards
            ("VIOLATED", "CONFIRMED"),
            # FLIPPED can only go to ACTIVE (PR #38). Other targets
            # are illegal.
            ("FLIPPED", "VIOLATED"),
            ("FLIPPED", "CONFIRMED"),
            ("FLIPPED", "CONSUMED"),
            ("FLIPPED", "FLIPPED"),      # self-loop
            ("CONFIRMED", "CONFIRMED"),  # self-loop
        ],
    )
    def test_invalid_transitions_raise(self, current: str, new: str) -> None:
        with pytest.raises(IllegalZoneTransitionError):
            validate_zone_transition(current, new)  # type: ignore[arg-type]

    def test_flipped_promotes_to_active(self) -> None:
        # PR #38: FLIPPED zones become tradeable in flipped_direction.
        # Placing a setup transitions FLIPPED → ACTIVE.
        assert _VALID_ZONE_TRANSITIONS["FLIPPED"] == frozenset({"ACTIVE"})
        # No state is truly terminal any more — every status has at
        # least one outgoing edge.
        assert TERMINAL_ZONE_STATUSES == frozenset()

    def test_skip_new_setup_statuses_covers_terminal_in_direction(self) -> None:
        assert SKIP_NEW_SETUP_STATUSES == frozenset(
            {"CONSUMED", "VIOLATED", "FLIPPED"}
        )


# --------------------------------------------------------------------------- #
# check_consumption — Q1: any touch consumes
# --------------------------------------------------------------------------- #


class TestCheckConsumption:
    def test_wick_into_zone_consumes_buy(self) -> None:
        # Low wicks INTO the zone (low=102 inside [100, 105]).
        zone = buy_zone()
        assert check_consumption(zone, bar_high=110.0, bar_low=102.0) is True

    def test_wick_into_zone_consumes_sell(self) -> None:
        # SELL zone consumed when bar's high pokes into [100, 105].
        zone = sell_zone()
        assert check_consumption(zone, bar_high=102.0, bar_low=95.0) is True

    def test_bar_engulfs_zone_consumes(self) -> None:
        # A bar that completely spans the zone obviously touches it.
        zone = buy_zone()
        assert check_consumption(zone, bar_high=120.0, bar_low=80.0) is True

    def test_bar_strictly_above_zone_does_not_consume(self) -> None:
        zone = buy_zone()
        assert check_consumption(zone, bar_high=110.0, bar_low=106.0) is False

    def test_bar_strictly_below_zone_does_not_consume(self) -> None:
        zone = buy_zone()
        assert check_consumption(zone, bar_high=99.0, bar_low=95.0) is False

    def test_exact_top_touch_consumes(self) -> None:
        # Endpoints inclusive: a wick touching exactly zone.top is consumed.
        zone = buy_zone()
        assert check_consumption(zone, bar_high=105.0, bar_low=105.0) is True

    def test_exact_bottom_touch_consumes(self) -> None:
        zone = buy_zone()
        assert check_consumption(zone, bar_high=100.0, bar_low=100.0) is True

    def test_consumption_is_fill_agnostic(self) -> None:
        # Documented Q1 semantics: doesn't matter whether the wick
        # caused a layer to fill — the touch alone consumes.
        # (Encoded just as a check_consumption sanity test; layer-fill
        # state is the orchestrator's concern.)
        zone = buy_zone()
        # Single-point wick touch on zone.top
        assert check_consumption(zone, bar_high=105.0, bar_low=104.99) is True


# --------------------------------------------------------------------------- #
# check_violation — body close past wrong-side bound
# --------------------------------------------------------------------------- #


class TestCheckViolation:
    def test_body_close_below_buy_zone_violates(self) -> None:
        zone = buy_zone()  # [100, 105]
        assert check_violation(zone, bar_close=99.99) is True

    def test_body_close_above_sell_zone_violates(self) -> None:
        zone = sell_zone()  # [100, 105]
        assert check_violation(zone, bar_close=105.01) is True

    def test_body_close_at_bottom_does_not_violate(self) -> None:
        # close == zone.bottom is NOT a violation (strict inequality).
        zone = buy_zone()
        assert check_violation(zone, bar_close=100.0) is False

    def test_body_close_at_top_does_not_violate(self) -> None:
        zone = sell_zone()
        assert check_violation(zone, bar_close=105.0) is False

    def test_wick_only_does_not_violate(self) -> None:
        # Body close stays inside zone even if wick poked through —
        # not a violation. (check_violation reads only the close;
        # callers must pass the bar's close, not its low.)
        zone = buy_zone()
        # Bar's low poked to 95 but body closed at 101.
        assert check_violation(zone, bar_close=101.0) is False


# --------------------------------------------------------------------------- #
# _nearest_bos_target — helper for check_flip
# --------------------------------------------------------------------------- #


def _swing(
    index: int, price: float, kind: str = "LOW",
    time: pd.Timestamp | None = None,
) -> Swing:
    return Swing(
        index=index,
        time=time or pd.Timestamp("2026-01-01", tz="UTC"),
        price=price,
        kind=kind,  # type: ignore[arg-type]
    )


class TestNearestBosTarget:
    def test_buy_picks_highest_low_below_zone(self) -> None:
        zone = buy_zone(bottom=100.0)
        swings = [
            _swing(0, 90.0, "LOW"),
            _swing(2, 95.0, "LOW"),   # highest below zone bottom
            _swing(4, 98.0, "LOW"),   # higher but still below — winner
            _swing(6, 100.0, "LOW"),  # not strictly below
            _swing(8, 110.0, "HIGH"),  # wrong kind
        ]
        target = _nearest_bos_target(zone, swings, up_to_index=10)
        assert target is not None
        assert target.price == 98.0

    def test_sell_picks_lowest_high_above_zone(self) -> None:
        zone = sell_zone(top=105.0)
        swings = [
            _swing(0, 110.0, "HIGH"),
            _swing(2, 108.0, "HIGH"),  # lowest above zone.top — winner
            _swing(4, 115.0, "HIGH"),
            _swing(6, 105.0, "HIGH"),  # not strictly above
            _swing(8, 100.0, "LOW"),   # wrong kind
        ]
        target = _nearest_bos_target(zone, swings, up_to_index=10)
        assert target is not None
        assert target.price == 108.0

    def test_up_to_index_excludes_later_swings(self) -> None:
        zone = buy_zone(bottom=100.0)
        swings = [
            _swing(0, 90.0, "LOW"),
            _swing(15, 98.0, "LOW"),  # later than up_to_index=10
        ]
        target = _nearest_bos_target(zone, swings, up_to_index=10)
        assert target is not None
        assert target.price == 90.0

    def test_no_qualifying_swing_returns_none(self) -> None:
        zone = buy_zone(bottom=100.0)
        # No LOW below the zone.
        swings = [
            _swing(0, 101.0, "LOW"),   # above zone bottom — disqualified
            _swing(2, 110.0, "HIGH"),  # wrong kind
        ]
        assert _nearest_bos_target(zone, swings, up_to_index=10) is None


# --------------------------------------------------------------------------- #
# check_flip — VIOLATED → FLIPPED across multiple bars
# --------------------------------------------------------------------------- #


class TestCheckFlip:
    """The flip detector should:

    * Recompute structure from the current df (option B / design Q2).
    * Accept BoS confirmation on the **same** bar as the violation or
      on **any later** bar (no time window).
    * Pick the nearest opposite-side swing as the BoS target.
    * Return ``flipped=False`` if no qualifying close has appeared yet
      (caller re-checks on later bars).
    """

    def _structured_df_with_bos_later(self) -> pd.DataFrame:
        """Builds a df where a swing low exists, then a body close
        falls below it several bars later. Calibrated so the structure
        detector finds the swing (needs 3 bars on each side at default
        strength=3).
        """
        # 30 quiet bars at 100 establish baseline noise. Then a clear
        # dip-and-recovery creates a swing LOW. Then a downward break.
        rows: list[tuple[float, float, float, float]] = []
        for i in range(30):
            rows.append((100.0, 100.2, 99.8, 100.0))
        # Dip: bars 30-33 establish a swing low at 95.
        rows.append((100.0, 100.0, 99.0, 99.0))
        rows.append((99.0, 99.0, 95.0, 95.0))
        rows.append((95.0, 99.0, 95.0, 99.0))
        rows.append((99.0, 100.0, 99.0, 100.0))
        # Quiet bars so the swing low at index 31 (price 95) confirms.
        for _ in range(15):
            rows.append((100.0, 100.2, 99.8, 100.0))
        # Now drop: body close at 90 — below the swing low of 95.
        rows.append((100.0, 100.0, 90.0, 90.0))
        return make_bar_df(rows)

    def test_buy_zone_flips_when_close_below_bos_target(self) -> None:
        df = self._structured_df_with_bos_later()
        # Zone above the dip — say [102, 110]. Violation will be where
        # we body-close below zone.bottom=102; the swing low at price
        # 95 sits below the zone, so BoS-down target = 95. The 90-close
        # at the end is below 95 → flips.
        zone = ZoneRef(direction="BUY", top=110.0, bottom=102.0)
        violation_index = len(df) - 1  # the 90-close bar
        result = check_flip(zone, df, violation_index)
        assert result.flipped is True
        assert result.new_direction == "SELL"
        assert result.bos_swing is not None
        assert result.bos_swing.kind == "LOW"
        assert result.broken_at == df.index[-1]

    def test_buy_zone_no_flip_when_no_bos(self) -> None:
        # df where price never closes below any swing low: no flip.
        df = make_bar_df([
            (100.0, 100.5, 99.5, 100.0),
        ] * 20 + [
            (100.0, 100.5, 99.5, 99.5),  # body close at 99.5; still > any swing low
        ])
        zone = ZoneRef(direction="BUY", top=105.0, bottom=99.0)
        result = check_flip(zone, df, violation_index=len(df) - 1)
        # No swing low at default strength=3 in this flat-noise df, so
        # there's nothing to BoS through → flipped=False.
        assert result.flipped is False
        assert result.new_direction is None

    def test_sell_zone_flips_when_close_above_bos_target(self) -> None:
        # Mirror of the BUY case: build a swing HIGH, then close above it.
        rows: list[tuple[float, float, float, float]] = []
        for _ in range(30):
            rows.append((100.0, 100.2, 99.8, 100.0))
        rows.append((100.0, 101.0, 100.0, 101.0))
        rows.append((101.0, 105.0, 101.0, 105.0))  # swing HIGH at 105
        rows.append((105.0, 105.0, 101.0, 101.0))
        rows.append((101.0, 101.0, 100.0, 100.0))
        for _ in range(15):
            rows.append((100.0, 100.2, 99.8, 100.0))
        rows.append((100.0, 110.0, 100.0, 110.0))  # body close 110 > 105 swing
        df = make_bar_df(rows)
        # Zone below the spike: e.g. [98, 99].
        zone = ZoneRef(direction="SELL", top=99.0, bottom=98.0)
        result = check_flip(zone, df, violation_index=len(df) - 1)
        assert result.flipped is True
        assert result.new_direction == "BUY"
        assert result.bos_swing is not None
        assert result.bos_swing.kind == "HIGH"

    def test_flip_with_explicit_structure_config(self) -> None:
        # Same setup as test_buy_zone_flips_when_close_below_bos_target
        # but with a custom StructureConfig. Just exercises that the
        # config is accepted and the detector runs.
        df = self._structured_df_with_bos_later()
        zone = ZoneRef(direction="BUY", top=110.0, bottom=102.0)
        result = check_flip(
            zone, df, violation_index=len(df) - 1,
            structure_config=StructureConfig(swing_strength=2),
        )
        # With smaller swing_strength, more swings detected — still
        # finds the BoS target somewhere; assertion is permissive.
        assert isinstance(result, FlipResult)

    def test_invalid_violation_index_raises(self) -> None:
        df = make_bar_df([(100.0, 100.5, 99.5, 100.0)] * 5)
        zone = buy_zone()
        with pytest.raises(ValueError, match="out of df range"):
            check_flip(zone, df, violation_index=99)
        with pytest.raises(ValueError, match="out of df range"):
            check_flip(zone, df, violation_index=-1)


# --------------------------------------------------------------------------- #
# zone_bounds_overlap — dedup helper
# --------------------------------------------------------------------------- #


class TestZoneBoundsOverlap:
    def test_identical_zones_overlap(self) -> None:
        a = buy_zone(top=105.0, bottom=100.0)
        b = buy_zone(top=105.0, bottom=100.0)
        assert zone_bounds_overlap(a, b) is True

    def test_partial_overlap_qualifies(self) -> None:
        a = buy_zone(top=105.0, bottom=100.0)
        b = buy_zone(top=110.0, bottom=103.0)
        assert zone_bounds_overlap(a, b) is True

    def test_within_tolerance_qualifies(self) -> None:
        # 0.3 gap is within the default 0.5 tolerance.
        a = buy_zone(top=105.0, bottom=100.0)
        b = buy_zone(top=110.0, bottom=105.3)
        assert zone_bounds_overlap(a, b) is True

    def test_outside_tolerance_does_not_qualify(self) -> None:
        a = buy_zone(top=105.0, bottom=100.0)
        b = buy_zone(top=110.0, bottom=106.0)  # 1.0 gap > 0.5 tolerance
        assert zone_bounds_overlap(a, b) is False

    def test_opposite_directions_never_overlap(self) -> None:
        # Even with identical bounds — direction differs.
        a = buy_zone(top=105.0, bottom=100.0)
        b = sell_zone(top=105.0, bottom=100.0)
        assert zone_bounds_overlap(a, b) is False

    def test_tolerance_can_be_tuned(self) -> None:
        a = buy_zone(top=105.0, bottom=100.0)
        b = buy_zone(top=110.0, bottom=107.0)
        # Default 0.5 tolerance: 2.0 gap → no overlap.
        assert zone_bounds_overlap(a, b) is False
        # Tighten to 5.0: 2.0 gap absorbed → overlap.
        assert zone_bounds_overlap(a, b, tolerance=5.0) is True


# --------------------------------------------------------------------------- #
# flipped_zone_body_broken_since_flip — PR #38 pre-trade safety helper
# --------------------------------------------------------------------------- #


class TestFlippedZoneBodyBrokenSinceFlip:
    """The pre-trade safety check rejects flipped zones whose flip
    premise has been broken by a body close past the wrong side
    AFTER the flip moment. Bars before ``flipped_at`` are ignored
    even if they show breaks — that history predates the flip and
    isn't relevant."""

    @staticmethod
    def _df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
        """rows: (open, high, low, close). Sequential 5-min bars from 2026-01-01."""
        times = pd.date_range(
            "2026-01-01T00:00:00Z", periods=len(rows), freq="5min", tz="UTC",
        )
        return pd.DataFrame(
            {
                "open":  [r[0] for r in rows],
                "high":  [r[1] for r in rows],
                "low":   [r[2] for r in rows],
                "close": [r[3] for r in rows],
                "volume": [100] * len(rows),
            },
            index=times,
        )

    def test_no_bars_after_flip_means_no_break(self) -> None:
        # flipped_at equals the last bar's timestamp → no bar strictly
        # after flip → False.
        df = self._df([(100, 101, 99, 100)] * 5)
        flipped_at = df.index[-1]
        assert flipped_zone_body_broken_since_flip(
            zone_top=110.0, zone_bottom=105.0,
            flipped_direction="BUY",
            flipped_at=flipped_at, df=df,
        ) is False

    def test_buy_break_below_bottom_after_flip_qualifies(self) -> None:
        # Zone [105, 110], flipped BUY. A body close at 100 after the
        # flip is a body break below bottom → True.
        df = self._df([
            (100, 101, 99, 100),    # 0 — pre-flip noise (close 100 < 105 but BEFORE flip; ignored)
            (100, 101, 99, 100),    # 1 — flip happens here
            (100, 101, 99, 100),    # 2 — post-flip noise (close 100, but 100 < 105 → break)
        ])
        flipped_at = df.index[1]
        # Note: bar 2's close=100 IS < zone.bottom=105, so it qualifies.
        assert flipped_zone_body_broken_since_flip(
            zone_top=110.0, zone_bottom=105.0,
            flipped_direction="BUY",
            flipped_at=flipped_at, df=df,
        ) is True

    def test_sell_break_above_top_after_flip_qualifies(self) -> None:
        df = self._df([
            (100, 101, 99, 100),    # 0 — pre-flip
            (100, 101, 99, 100),    # 1 — flip
            (100, 121, 99, 120),    # 2 — close 120 > top 110
        ])
        flipped_at = df.index[1]
        assert flipped_zone_body_broken_since_flip(
            zone_top=110.0, zone_bottom=105.0,
            flipped_direction="SELL",
            flipped_at=flipped_at, df=df,
        ) is True

    def test_break_before_flip_is_ignored(self) -> None:
        # A close below zone.bottom on bar 0 doesn't disqualify if flip
        # only happened on bar 2. Bar 0 < flipped_at → not scanned.
        df = self._df([
            (100, 101, 99, 90),     # 0 — close 90 < 105, but BEFORE flip
            (100, 101, 99, 107),    # 1
            (100, 101, 99, 107),    # 2 — flip happens here
            (100, 101, 99, 107),    # 3 — post-flip, all closes inside [105, 110]
        ])
        flipped_at = df.index[2]
        assert flipped_zone_body_broken_since_flip(
            zone_top=110.0, zone_bottom=105.0,
            flipped_direction="BUY",
            flipped_at=flipped_at, df=df,
        ) is False

    def test_wick_only_does_not_qualify(self) -> None:
        # A bar with low=90 (well below zone.bottom=105) but close=107
        # (inside zone) is NOT a body break. The helper reads close,
        # not low/high — same convention as check_violation.
        df = self._df([
            (107, 108, 90, 107),    # 0 — flip happens here
            (107, 108, 90, 107),    # 1 — deep wick down but close inside
        ])
        flipped_at = df.index[0]
        assert flipped_zone_body_broken_since_flip(
            zone_top=110.0, zone_bottom=105.0,
            flipped_direction="BUY",
            flipped_at=flipped_at, df=df,
        ) is False

    def test_close_at_bottom_does_not_qualify_buy(self) -> None:
        # Strict inequality: close == zone.bottom is NOT a body break.
        df = self._df([
            (105, 105, 105, 105),   # 0 — flip
            (105, 105, 105, 105),   # 1 — close exactly at bottom
        ])
        flipped_at = df.index[0]
        assert flipped_zone_body_broken_since_flip(
            zone_top=110.0, zone_bottom=105.0,
            flipped_direction="BUY",
            flipped_at=flipped_at, df=df,
        ) is False

    def test_close_at_top_does_not_qualify_sell(self) -> None:
        df = self._df([
            (110, 110, 110, 110),
            (110, 110, 110, 110),
        ])
        flipped_at = df.index[0]
        assert flipped_zone_body_broken_since_flip(
            zone_top=110.0, zone_bottom=105.0,
            flipped_direction="SELL",
            flipped_at=flipped_at, df=df,
        ) is False
