"""Tests for ``bot.strategy.pipeline`` — S&D pipeline composition.

Stages are patched so each test drives a specific scenario without
depending on the per-stage modules' real logic. The per-stage logic
is tested in its own test file.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from bot.strategy.pattern_detection import (
    Base,
    Impulse,
    Pattern,
    PatternType,
)
from bot.strategy.pipeline import (
    StrategyPipelineConfig,
    run_strategy_pipeline,
)
from bot.strategy.strong_point import ValidatedZone
from bot.strategy.structure import (
    StructureSnapshot,
    StructureState,
    Swing,
)
from bot.strategy.zone_marking import Zone
from bot.strategy.zone_refinement import RefinedZone


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ts() -> pd.Timestamp:
    return pd.Timestamp("2026-05-08T12:00:00Z")


def make_pattern(
    pattern_type: PatternType = PatternType.RBR,
    direction: str = "BUY",
) -> Pattern:
    impulse = Impulse(
        direction="RALLY", start_index=0, end_index=0,
        start_time=_ts(), end_time=_ts(),
        range_size=5.0, largest_body=5.0, candle_count=1,
    )
    base = Base(
        start_index=1, end_index=1, candle_count=1,
        top=100.5, bottom=100.0, range_size=0.5, largest_body=0.5,
    )
    return Pattern(
        pattern_type=pattern_type,
        impulse_before=impulse, base=base, impulse_after=impulse,
        direction=direction,  # type: ignore[arg-type]
        formed_at=_ts(),
    )


def make_zone(direction: str = "BUY", pattern: Pattern | None = None) -> Zone:
    pat = pattern or make_pattern()
    return Zone(
        direction=direction,  # type: ignore[arg-type]
        top=100.5, bottom=100.0, formed_at=_ts(), source_pattern=pat,
    )


def make_refined(
    *,
    is_tradeable: bool = True,
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
    *,
    is_strong_point: bool = True,
    direction: str = "BUY",
) -> ValidatedZone:
    refined = make_refined(direction=direction)
    return ValidatedZone(
        direction=refined.direction, top=refined.top, bottom=refined.bottom,
        formed_at=refined.formed_at, source_pattern=refined.source_pattern,
        refined_zone=refined,
        is_strong_point=is_strong_point,
        validation_failures=[] if is_strong_point else ["NO_BREAK_YET"],
        broken_swing=Swing(
            index=2, time=_ts(), price=105.0, kind="HIGH",
        ) if is_strong_point else None,
        broken_at=_ts() if is_strong_point else None,
        sl_anchor_swing=Swing(
            index=2, time=_ts(), price=99.0, kind="LOW",
        ) if is_strong_point else None,
    )


def make_structure_snapshot() -> StructureSnapshot:
    return StructureSnapshot(
        swings=[], bos_events=[], state=StructureState.RANGE,
        last_swing_high=None, last_swing_low=None, last_bos=None,
    )


def make_df(n: int = 30) -> pd.DataFrame:
    times = pd.date_range(
        "2026-05-08T08:00:00Z", periods=n, freq="5min", tz="UTC",
    )
    return pd.DataFrame(
        {
            "open": [100.0] * n, "high": [100.5] * n,
            "low": [99.5] * n, "close": [100.0] * n,
            "volume": [100] * n,
        },
        index=times,
    )


def patch_stages(
    mocker: MockerFixture,
    *,
    patterns: list[Pattern] | None = None,
    refined_results: list[RefinedZone] | None = None,
    validated_results: list[ValidatedZone] | None = None,
) -> dict[str, MagicMock]:
    """Patch every public function the pipeline calls."""
    stubs: dict[str, MagicMock] = {}
    stubs["analyze"] = mocker.patch(
        "bot.strategy.pipeline.analyze_structure",
        return_value=make_structure_snapshot(),
    )
    stubs["detect"] = mocker.patch(
        "bot.strategy.pipeline.detect_patterns",
        return_value=patterns or [],
    )
    stubs["mark"] = mocker.patch(
        "bot.strategy.pipeline.mark_zone",
        side_effect=lambda p, df: make_zone(p.direction, p),
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
            side_effect=lambda r, df, sw, cfg: make_validated(direction=r.direction),
        )
    return stubs


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


class TestPipelineComposition:
    def test_no_patterns_returns_empty(self, mocker: MockerFixture) -> None:
        stubs = patch_stages(mocker)
        result = run_strategy_pipeline(make_df())
        assert result == []
        stubs["mark"].assert_not_called()
        stubs["refine"].assert_not_called()
        stubs["validate"].assert_not_called()

    def test_one_pattern_all_gates_pass_returns_one_zone(
        self, mocker: MockerFixture,
    ) -> None:
        stubs = patch_stages(mocker, patterns=[make_pattern()])
        result = run_strategy_pipeline(make_df())
        assert len(result) == 1
        assert result[0].direction == "BUY"
        assert stubs["mark"].call_count == 1
        assert stubs["refine"].call_count == 1
        assert stubs["validate"].call_count == 1

    def test_swings_passed_into_validate(
        self, mocker: MockerFixture,
    ) -> None:
        # Pre-built structure with a couple of swings; pipeline should
        # forward them to validate_strong_point.
        sw = [Swing(index=2, time=_ts(), price=105.0, kind="HIGH")]
        snapshot = StructureSnapshot(
            swings=sw, bos_events=[], state=StructureState.RANGE,
            last_swing_high=None, last_swing_low=None, last_bos=None,
        )
        mocker.patch(
            "bot.strategy.pipeline.analyze_structure", return_value=snapshot,
        )
        mocker.patch(
            "bot.strategy.pipeline.detect_patterns",
            return_value=[make_pattern()],
        )
        mocker.patch(
            "bot.strategy.pipeline.mark_zone",
            side_effect=lambda p, df: make_zone(p.direction, p),
        )
        mocker.patch(
            "bot.strategy.pipeline.refine_zone",
            side_effect=lambda z, df, cfg: make_refined(direction=z.direction),
        )
        validate_stub = mocker.patch(
            "bot.strategy.pipeline.validate_strong_point",
            side_effect=lambda r, df, sw_, cfg: make_validated(direction=r.direction),
        )
        run_strategy_pipeline(make_df())
        # 3rd positional arg to validate is the swings list.
        passed_swings = validate_stub.call_args.args[2]
        assert passed_swings == sw


# --------------------------------------------------------------------------- #
# Filter cut-points
# --------------------------------------------------------------------------- #


class TestFilters:
    def test_non_tradeable_refined_zone_short_circuits_at_validate(
        self, mocker: MockerFixture,
    ) -> None:
        # validate is called even for non-tradeable refined zones — it
        # short-circuits internally with "NOT_TRADEABLE" failure.
        # Pipeline keeps the call but filters non-strong-point results.
        stubs = patch_stages(
            mocker,
            patterns=[make_pattern()],
            refined_results=[make_refined(
                is_tradeable=False, rejection_reason="ZONE_TOO_NARROW",
            )],
            validated_results=[make_validated(is_strong_point=False)],
        )
        result = run_strategy_pipeline(make_df())
        assert result == []
        # Pipeline DOES call validate for non-tradeable zones; the
        # validator itself returns is_strong_point=False with the
        # NOT_TRADEABLE failure reason. Pipeline filters at the end.
        stubs["validate"].assert_called_once()

    def test_non_strong_point_validated_filtered_out(
        self, mocker: MockerFixture,
    ) -> None:
        patch_stages(
            mocker,
            patterns=[make_pattern()],
            validated_results=[make_validated(is_strong_point=False)],
        )
        result = run_strategy_pipeline(make_df())
        assert result == []


# --------------------------------------------------------------------------- #
# Per-pattern error isolation
# --------------------------------------------------------------------------- #


class TestPerPatternErrorIsolation:
    def test_one_bad_pattern_does_not_kill_the_batch(
        self, mocker: MockerFixture,
    ) -> None:
        good = make_refined()
        # 3 patterns; the middle refine raises.
        patch_stages(
            mocker,
            patterns=[make_pattern(), make_pattern(), make_pattern()],
            refined_results=[good, ValueError("kaboom"), good],
            validated_results=[
                make_validated(),
                make_validated(),  # never reached due to refine error
                make_validated(),
            ],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 2

    def test_validate_raises_skips_pattern(
        self, mocker: MockerFixture,
    ) -> None:
        patch_stages(
            mocker,
            patterns=[make_pattern(), make_pattern()],
            validated_results=[make_validated(), RuntimeError("nope")],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 1


# --------------------------------------------------------------------------- #
# Config + defaults
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_default_config_used_when_none_passed(
        self, mocker: MockerFixture,
    ) -> None:
        patch_stages(mocker)
        run_strategy_pipeline(make_df())  # no raise

    def test_default_config_values(self) -> None:
        c = StrategyPipelineConfig()
        assert c.swing_strength == 2
        assert c.impulse_body_to_range_ratio_min == 0.6
        assert c.impulse_atr_multiple_min == 1.0
        assert c.atr_period == 14
        assert c.max_impulse_run_candles == 5
        assert c.min_base_candles == 1
        assert c.max_base_candles == 5
        assert c.base_range_to_impulse_ratio_max == 0.6
        assert c.base_max_body_to_impulse_body_ratio == 0.4
        assert c.zone_min_size_points == 5.0
        assert c.zone_max_size_points == 80.0
        assert c.sl_buffer_points == 17.5
