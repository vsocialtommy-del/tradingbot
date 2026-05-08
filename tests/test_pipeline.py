"""Tests for ``bot.strategy.pipeline``.

The per-stage modules are exhaustively tested already; this file only
verifies the composition: stages wired in the right order, filters
applied at the right cut-points, errors in one pattern don't kill the
batch. Stages are stubbed via mocker.patch so each test can drive a
specific scenario.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.strategy.imbalance import ImbalanceZone
from bot.strategy.pipeline import (
    StrategyPipelineConfig,
    run_strategy_pipeline,
)
from bot.strategy.pattern_detection import MPattern, WPattern
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.structure import BosEvent, StructureSnapshot, StructureState, Swing
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _ts() -> pd.Timestamp:
    return pd.Timestamp("2026-05-08T12:00:00Z")


def make_w() -> WPattern:
    return WPattern(
        low1=Swing(index=2, time=_ts(), price=1895.0, kind="LOW"),
        low2=Swing(index=9, time=_ts(), price=1895.0, kind="LOW"),
        peak_index=6, peak_time=_ts(), peak_price=1905.0,
        formed_at=_ts(), completed=True,
    )


def make_m() -> MPattern:
    return MPattern(
        high1=Swing(index=2, time=_ts(), price=1905.0, kind="HIGH"),
        high2=Swing(index=9, time=_ts(), price=1905.0, kind="HIGH"),
        trough_index=6, trough_time=_ts(), trough_price=1895.0,
        formed_at=_ts(), completed=True,
    )


def make_zone(direction: str = "BUY") -> Zone:
    return Zone(
        direction=direction,  # type: ignore[arg-type]
        top=1900.0, bottom=1895.0, formed_at=_ts(),
        source_pattern=make_w() if direction == "BUY" else make_m(),
    )


def make_refined(
    *, is_tradeable: bool = True,
    rejection_reason: str | None = None,
    direction: str = "BUY",
) -> RefinedZone:
    z = make_zone(direction)
    return RefinedZone(
        direction=z.direction, top=z.top, bottom=z.bottom,
        formed_at=z.formed_at, source_pattern=z.source_pattern,
        is_tradeable=is_tradeable,
        rejection_reason=rejection_reason,  # type: ignore[arg-type]
        original_zone=z,
    )


def make_validated(
    *, is_strong: bool = True,
    is_tradeable: bool = True,
    bos_event: BosEvent | None = None,
    direction: str = "BUY",
) -> ValidatedZone:
    refined = make_refined(is_tradeable=is_tradeable, direction=direction)
    if bos_event is None:
        bos_event = BosEvent(
            bar_index=12, time=_ts(),
            direction="UP" if direction == "BUY" else "DOWN",
            broken_swing_index=4, broken_level=1907.5, break_close=1908.0,
        )
    return ValidatedZone(
        direction=refined.direction, top=refined.top, bottom=refined.bottom,
        formed_at=refined.formed_at, source_pattern=refined.source_pattern,
        is_tradeable=refined.is_tradeable, rejection_reason=None,
        original_zone=refined.original_zone, refined_zone=refined,
        is_strong_point=is_strong, validation_failures=[],
        bos_event=bos_event if is_strong else None,
    )


def make_imbalance_zone(
    *, is_strong: bool = True, is_imbalance: bool = True,
    is_tapped: bool = False, direction: str = "BUY",
) -> ImbalanceZone:
    v = make_validated(is_strong=is_strong, direction=direction)
    return ImbalanceZone(
        direction=v.direction, top=v.top, bottom=v.bottom,
        formed_at=v.formed_at, source_pattern=v.source_pattern,
        is_tradeable=v.is_tradeable, rejection_reason=None,
        original_zone=v.original_zone, refined_zone=v.refined_zone,
        is_strong_point=v.is_strong_point,
        validation_failures=v.validation_failures, bos_event=v.bos_event,
        validated_zone=v,
        approach_count=2 if is_imbalance else 0,
        is_imbalance=is_imbalance, approach_events=[],
        qualified_at=v.formed_at if is_imbalance else None,
        is_tapped=is_tapped, tapped_at=v.formed_at if is_tapped else None,
    )


def make_structure_snapshot(bos_events: list[BosEvent]) -> StructureSnapshot:
    return StructureSnapshot(
        swings=[],
        bos_events=bos_events,
        state=StructureState.RANGE,
        last_swing_high=None,
        last_swing_low=None,
        last_bos=bos_events[-1] if bos_events else None,
    )


def make_df(n: int = 30) -> pd.DataFrame:
    times = pd.date_range(
        start="2026-05-08T08:00:00Z", periods=n, freq="5min", tz="UTC",
    )
    return pd.DataFrame(
        {
            "open": [1900.0] * n, "high": [1901.0] * n,
            "low": [1899.0] * n, "close": [1900.0] * n,
            "volume": [100] * n,
        },
        index=times,
    )


# --------------------------------------------------------------------------- #
# Fixture helpers — patch every stage of the pipeline
# --------------------------------------------------------------------------- #


def patch_stages(
    mocker: MockerFixture,
    *,
    bos_events: list[BosEvent] | None = None,
    w_patterns: list[WPattern] | None = None,
    m_patterns: list[MPattern] | None = None,
    refined_results: list[RefinedZone] | None = None,
    validated_results: list[ValidatedZone] | None = None,
    imbalance_results: list[ImbalanceZone] | None = None,
) -> dict[str, MagicMock]:
    """Patch every public function the pipeline calls.

    Each ``*_results`` list provides successive return values for the
    stage; if a single value is given for a stage that's called per
    pattern we replicate it.
    """
    stubs: dict[str, MagicMock] = {}
    stubs["analyze"] = mocker.patch(
        "bot.strategy.pipeline.analyze_structure",
        return_value=make_structure_snapshot(bos_events or []),
    )
    stubs["w"] = mocker.patch(
        "bot.strategy.pipeline.detect_w_patterns",
        return_value=w_patterns or [],
    )
    stubs["m"] = mocker.patch(
        "bot.strategy.pipeline.detect_m_patterns",
        return_value=m_patterns or [],
    )
    # mark_zone is called per pattern; return the same Zone each time.
    stubs["mark"] = mocker.patch(
        "bot.strategy.pipeline.mark_zone",
        side_effect=lambda p, df: make_zone(
            direction="BUY" if isinstance(p, WPattern) else "SELL",
        ),
    )
    if refined_results is not None:
        stubs["refine"] = mocker.patch(
            "bot.strategy.pipeline.refine_zone",
            side_effect=refined_results,
        )
    else:
        stubs["refine"] = mocker.patch(
            "bot.strategy.pipeline.refine_zone",
            side_effect=lambda z, df, cfg: make_refined(direction=z.direction),
        )
    if validated_results is not None:
        stubs["validate"] = mocker.patch(
            "bot.strategy.pipeline.validate_strong_point",
            side_effect=validated_results,
        )
    else:
        stubs["validate"] = mocker.patch(
            "bot.strategy.pipeline.validate_strong_point",
            side_effect=lambda r, df, bos, cfg: make_validated(
                direction=r.direction,
            ),
        )
    if imbalance_results is not None:
        stubs["imbalance"] = mocker.patch(
            "bot.strategy.pipeline.track_imbalance",
            side_effect=imbalance_results,
        )
    else:
        stubs["imbalance"] = mocker.patch(
            "bot.strategy.pipeline.track_imbalance",
            side_effect=lambda v, df, cfg: make_imbalance_zone(
                direction=v.direction,
            ),
        )
    return stubs


# --------------------------------------------------------------------------- #
# Composition tests
# --------------------------------------------------------------------------- #


class TestPipelineComposition:
    def test_no_patterns_returns_empty_no_downstream_calls(
        self, mocker: MockerFixture,
    ) -> None:
        stubs = patch_stages(mocker)
        result = run_strategy_pipeline(make_df())
        assert result == []
        # Downstream stages never called when nothing was detected.
        stubs["mark"].assert_not_called()
        stubs["refine"].assert_not_called()
        stubs["validate"].assert_not_called()
        stubs["imbalance"].assert_not_called()

    def test_one_w_pattern_passes_all_gates_returns_one_zone(
        self, mocker: MockerFixture,
    ) -> None:
        stubs = patch_stages(mocker, w_patterns=[make_w()])
        result = run_strategy_pipeline(make_df())
        assert len(result) == 1
        assert result[0].direction == "BUY"
        # Every stage hit exactly once.
        assert stubs["mark"].call_count == 1
        assert stubs["refine"].call_count == 1
        assert stubs["validate"].call_count == 1
        assert stubs["imbalance"].call_count == 1

    def test_w_and_m_patterns_both_processed(
        self, mocker: MockerFixture,
    ) -> None:
        patch_stages(
            mocker, w_patterns=[make_w()], m_patterns=[make_m()],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 2
        directions = {z.direction for z in result}
        assert directions == {"BUY", "SELL"}

    def test_bos_events_passed_into_validate(
        self, mocker: MockerFixture,
    ) -> None:
        bos = BosEvent(
            bar_index=10, time=_ts(), direction="UP",
            broken_swing_index=4, broken_level=1908.0, break_close=1909.0,
        )
        stubs = patch_stages(
            mocker, w_patterns=[make_w()], bos_events=[bos],
        )
        run_strategy_pipeline(make_df())
        # validate_strong_point received the bos_events list.
        passed_bos = stubs["validate"].call_args.args[2]
        assert passed_bos == [bos]


# --------------------------------------------------------------------------- #
# Filter cut-points
# --------------------------------------------------------------------------- #


class TestFilters:
    def test_non_tradeable_refined_zone_short_circuits(
        self, mocker: MockerFixture,
    ) -> None:
        # Refined zone fails the size filter → don't proceed.
        stubs = patch_stages(
            mocker, w_patterns=[make_w()],
            refined_results=[make_refined(
                is_tradeable=False, rejection_reason="ZONE_TOO_NARROW",
            )],
        )
        result = run_strategy_pipeline(make_df())
        assert result == []
        # validate / imbalance NOT called once tradeable=False.
        stubs["validate"].assert_not_called()
        stubs["imbalance"].assert_not_called()

    def test_non_strong_point_short_circuits(
        self, mocker: MockerFixture,
    ) -> None:
        # Refined ok, but validation gates failed.
        stubs = patch_stages(
            mocker, w_patterns=[make_w()],
            validated_results=[make_validated(is_strong=False)],
        )
        result = run_strategy_pipeline(make_df())
        assert result == []
        # imbalance NOT called once is_strong_point=False.
        stubs["imbalance"].assert_not_called()

    def test_tapped_imbalance_zone_filtered_out(
        self, mocker: MockerFixture,
    ) -> None:
        # Strong + Imbalance, but already tapped → first-touch consumed.
        patch_stages(
            mocker, w_patterns=[make_w()],
            imbalance_results=[make_imbalance_zone(is_tapped=True)],
        )
        result = run_strategy_pipeline(make_df())
        assert result == []

    def test_strong_but_not_imbalance_still_returned(
        self, mocker: MockerFixture,
    ) -> None:
        # Strong Point that hasn't qualified as Imbalance yet — still
        # tradeable on first touch (STRONG_POINT_FIRST_TOUCH mode).
        patch_stages(
            mocker, w_patterns=[make_w()],
            imbalance_results=[
                make_imbalance_zone(is_strong=True, is_imbalance=False),
            ],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 1
        assert result[0].is_strong_point is True
        assert result[0].is_imbalance is False


# --------------------------------------------------------------------------- #
# Per-pattern error isolation
# --------------------------------------------------------------------------- #


class TestPerPatternErrorIsolation:
    def test_one_bad_pattern_does_not_kill_the_others(
        self, mocker: MockerFixture,
    ) -> None:
        # Three patterns; the middle refine_zone call raises. Pipeline
        # should skip and continue.
        good_refined = make_refined()
        patch_stages(
            mocker,
            w_patterns=[make_w(), make_w(), make_w()],
            refined_results=[
                good_refined,
                ValueError("kaboom"),  # type: ignore[list-item]
                good_refined,
            ],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 2  # 1st + 3rd survive

    def test_validate_raises_skips_pattern(
        self, mocker: MockerFixture,
    ) -> None:
        patch_stages(
            mocker,
            w_patterns=[make_w(), make_w()],
            validated_results=[
                make_validated(is_strong=True),
                RuntimeError("bad bar index"),  # type: ignore[list-item]
            ],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 1


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #


class TestConfigPlumbing:
    def test_pipeline_config_propagates_to_stages(
        self, mocker: MockerFixture,
    ) -> None:
        # Use non-default values so we can confirm they actually flow.
        cfg = StrategyPipelineConfig(
            swing_strength=5,
            pattern_tolerance_pct=0.005,
            zone_min_size_points=2.0, zone_max_size_points=120.0,
            impulse_min_body_ratio=0.7,
            imbalance_approach_distance=10.0,
            imbalance_approach_threshold=3,
        )
        stubs = patch_stages(mocker, w_patterns=[make_w()])
        run_strategy_pipeline(make_df(), config=cfg)

        # analyze_structure got swing_strength
        assert stubs["analyze"].call_args.args[1].swing_strength == 5
        # detect_w got pattern config
        assert stubs["w"].call_args.args[1].swing_strength == 5
        assert stubs["w"].call_args.args[1].pattern_tolerance_pct == 0.005
        # refine got refinement config
        rcfg = stubs["refine"].call_args.args[2]
        assert rcfg.zone_min_size_points == 2.0
        assert rcfg.zone_max_size_points == 120.0
        # validate got strong point config
        scfg = stubs["validate"].call_args.args[3]
        assert scfg.impulse_min_body_ratio == 0.7
        # imbalance got its config
        icfg = stubs["imbalance"].call_args.args[2]
        assert icfg.imbalance_approach_distance == 10.0
        assert icfg.imbalance_approach_threshold == 3

    def test_default_config_used_when_none_passed(
        self, mocker: MockerFixture,
    ) -> None:
        patch_stages(mocker)
        # Doesn't raise.
        run_strategy_pipeline(make_df())


class TestDataclassDefaults:
    def test_pipeline_config_defaults_match_per_module_defaults(self) -> None:
        c = StrategyPipelineConfig()
        assert c.swing_strength == 3
        assert c.pattern_tolerance_pct == 0.001
        assert c.pattern_lookback_bars == 50
        assert c.zone_min_size_points == 5.0
        assert c.zone_max_size_points == 80.0
        assert c.impulse_min_body_ratio == 0.6
        assert c.base_max_range_ratio == 0.5
        assert c.imbalance_approach_distance == 7.5
        assert c.imbalance_retreat_distance == 5.0
        assert c.imbalance_approach_threshold == 2
