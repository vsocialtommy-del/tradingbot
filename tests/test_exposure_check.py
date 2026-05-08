"""Tests for ``bot.risk.exposure_check``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from bot.risk.exposure_check import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    ExposureCheckResult,
    check_exposure,
    count_active_setups,
)


# --------------------------------------------------------------------------- #
# check_exposure — core behaviour
# --------------------------------------------------------------------------- #


class TestCheckExposureCore:
    def test_zero_active_max_three_can_open(self) -> None:
        r = check_exposure(active_count=0, max_simultaneous=3)
        assert r.can_open_new is True
        assert r.current_count == 0
        assert r.max_allowed == 3
        assert r.reason is None

    def test_two_active_max_three_can_open(self) -> None:
        r = check_exposure(active_count=2, max_simultaneous=3)
        assert r.can_open_new is True
        assert r.reason is None

    def test_three_active_max_three_full(self) -> None:
        r = check_exposure(active_count=3, max_simultaneous=3)
        assert r.can_open_new is False
        assert r.reason == "MAX_EXPOSURE_REACHED"
        assert r.current_count == 3
        assert r.max_allowed == 3

    def test_four_active_max_three_over_exposure(self) -> None:
        # Defensive: shouldn't normally happen but the function should
        # not crash and should return False.
        r = check_exposure(active_count=4, max_simultaneous=3)
        assert r.can_open_new is False
        assert r.reason == "MAX_EXPOSURE_REACHED"
        assert r.current_count == 4

    def test_max_zero_is_kill_switch(self) -> None:
        # max=0 should always reject — useful as a config-level halt.
        for active in (0, 1, 5):
            r = check_exposure(active_count=active, max_simultaneous=0)
            assert r.can_open_new is False
            assert r.reason == "MAX_EXPOSURE_REACHED"


class TestCheckExposureCandidate:
    """The with_candidate flag is purely cosmetic; math doesn't change."""

    def test_three_active_max_three_with_candidate_still_false(self) -> None:
        r = check_exposure(
            active_count=3, max_simultaneous=3, with_candidate=True
        )
        assert r.can_open_new is False

    def test_two_active_max_three_with_candidate_true(self) -> None:
        # Candidate would fill the third slot — allowed.
        r = check_exposure(
            active_count=2, max_simultaneous=3, with_candidate=True
        )
        assert r.can_open_new is True

    def test_candidate_flag_does_not_change_math(self) -> None:
        # Same active and max, same answer regardless of flag.
        for active in range(5):
            without = check_exposure(active_count=active, max_simultaneous=3)
            with_flag = check_exposure(
                active_count=active, max_simultaneous=3, with_candidate=True
            )
            assert without.can_open_new == with_flag.can_open_new


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_negative_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_simultaneous"):
            check_exposure(active_count=0, max_simultaneous=-1)

    def test_negative_active_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="active_count"):
            check_exposure(active_count=-1, max_simultaneous=3)


# --------------------------------------------------------------------------- #
# count_active_setups
# --------------------------------------------------------------------------- #


# Lightweight Setup-like for object-access tests.
@dataclass
class FakeSetup:
    status: str


class TestCountActiveSetups:
    def test_empty_iterable_returns_zero(self) -> None:
        assert count_active_setups([]) == 0

    def test_all_active_statuses_count(self) -> None:
        # Spec lists exactly three: PENDING, ACTIVE, TP1_HIT.
        setups = [
            FakeSetup(status="PENDING"),
            FakeSetup(status="ACTIVE"),
            FakeSetup(status="TP1_HIT"),
        ]
        assert count_active_setups(setups) == 3

    def test_terminal_statuses_do_not_count(self) -> None:
        setups = [
            FakeSetup(status="CLOSED"),
            FakeSetup(status="SKIPPED"),
            FakeSetup(status="STOPPED_OUT"),
        ]
        assert count_active_setups(setups) == 0

    def test_mixed_statuses_only_actives_counted(self) -> None:
        # 5 CLOSED + 1 ACTIVE = 1 active total
        setups = [
            FakeSetup(status="CLOSED"),
            FakeSetup(status="CLOSED"),
            FakeSetup(status="CLOSED"),
            FakeSetup(status="ACTIVE"),
            FakeSetup(status="CLOSED"),
            FakeSetup(status="CLOSED"),
        ]
        assert count_active_setups(setups) == 1

    def test_combination_of_three_active_states(self) -> None:
        # 1 PENDING + 1 ACTIVE + 1 TP1_HIT = 3 active total
        setups = [
            FakeSetup(status="PENDING"),
            FakeSetup(status="ACTIVE"),
            FakeSetup(status="TP1_HIT"),
            FakeSetup(status="CLOSED"),
        ]
        assert count_active_setups(setups) == 3

    def test_dict_shaped_setups_supported(self) -> None:
        # Supabase rows come as dicts; should work without conversion.
        setups = [
            {"status": "ACTIVE", "id": "abc"},
            {"status": "CLOSED", "id": "def"},
            {"status": "PENDING", "id": "ghi"},
        ]
        assert count_active_setups(setups) == 2

    def test_mixed_dict_and_object_shapes(self) -> None:
        setups = [
            {"status": "ACTIVE"},
            FakeSetup(status="TP1_HIT"),
            {"status": "CLOSED"},
        ]
        assert count_active_setups(setups) == 2

    def test_items_without_status_not_counted(self) -> None:
        # Defensive — items that don't have a status field at all are
        # silently ignored rather than raising.
        @dataclass
        class NoStatus:
            id: str

        setups = [
            NoStatus(id="x"),
            FakeSetup(status="ACTIVE"),
            {"id": "no-status-key"},
        ]
        assert count_active_setups(setups) == 1

    def test_unknown_status_strings_not_counted(self) -> None:
        # If the status string isn't one we recognise, it's treated
        # as inactive (defensive — schema migrations could add new
        # statuses we don't yet know about; we don't double-count
        # them as active).
        setups = [
            FakeSetup(status="WEIRD_NEW_STATUS"),
            FakeSetup(status="ACTIVE"),
        ]
        assert count_active_setups(setups) == 1


# --------------------------------------------------------------------------- #
# Integration — count_active_setups feeding check_exposure
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_real_workflow_at_capacity(self) -> None:
        setups = [
            FakeSetup(status="ACTIVE"),
            FakeSetup(status="TP1_HIT"),
            FakeSetup(status="PENDING"),
            FakeSetup(status="CLOSED"),
            FakeSetup(status="SKIPPED"),
        ]
        active = count_active_setups(setups)
        assert active == 3
        r = check_exposure(active_count=active, max_simultaneous=3)
        assert r.can_open_new is False
        assert r.reason == "MAX_EXPOSURE_REACHED"

    def test_real_workflow_with_room(self) -> None:
        setups = [
            FakeSetup(status="ACTIVE"),
            FakeSetup(status="CLOSED"),
            FakeSetup(status="CLOSED"),
        ]
        active = count_active_setups(setups)
        assert active == 1
        r = check_exposure(
            active_count=active, max_simultaneous=3, with_candidate=True
        )
        assert r.can_open_new is True
        assert r.current_count == 1
        assert r.max_allowed == 3


# --------------------------------------------------------------------------- #
# Constants sanity
# --------------------------------------------------------------------------- #


class TestConstants:
    def test_active_and_terminal_disjoint(self) -> None:
        # No status should appear in both — sanity check on the constants.
        assert ACTIVE_STATUSES.isdisjoint(TERMINAL_STATUSES)

    def test_active_statuses_match_spec(self) -> None:
        assert ACTIVE_STATUSES == {"PENDING", "ACTIVE", "TP1_HIT"}

    def test_terminal_statuses_match_spec(self) -> None:
        assert TERMINAL_STATUSES == {"CLOSED", "SKIPPED", "STOPPED_OUT"}

    def test_constants_are_frozen(self) -> None:
        with pytest.raises(AttributeError):
            ACTIVE_STATUSES.add("HACK")  # type: ignore[attr-defined]
