"""Tests for ``bot.strategy.pipeline`` — S&D pipeline composition.

Stages are patched so each test drives a specific scenario without
depending on the per-stage modules' real logic. The per-stage logic
is tested in its own test file.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
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
        validation_failures=[] if is_strong_point else ["NOT_TRADEABLE"],
        broken_swing=None,
        broken_at=None,
        sl_anchor_swing=None,
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
    """Patch every public function the pipeline calls.

    Post May-2026 loosened-rules PR: ``analyze_structure`` is no
    longer called by the pipeline (no break-target lookup), and
    :func:`validate_strong_point` takes ``(refined, df, cfg)`` —
    no swings list.
    """
    stubs: dict[str, MagicMock] = {}
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
            side_effect=lambda r, df, cfg: make_validated(direction=r.direction),
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

    def test_validate_called_with_df_and_config_no_swings(
        self, mocker: MockerFixture,
    ) -> None:
        # Loosened flow: validate_strong_point takes (refined, df, cfg)
        # — no swings list, no structure analysis.
        stubs = patch_stages(mocker, patterns=[make_pattern()])
        run_strategy_pipeline(make_df())
        call = stubs["validate"].call_args
        # Three positional args: refined, df, cfg.
        assert len(call.args) == 3
        assert isinstance(call.args[1], pd.DataFrame)


# --------------------------------------------------------------------------- #
# Filter cut-points
# --------------------------------------------------------------------------- #


class TestPipelineReturnSurface:
    """The pipeline returns *every* validated zone (tradeable or not) so
    the caller can persist them for the data trail. Filtering for
    placement happens downstream in ``main._maybe_run_strategy``."""

    def test_non_tradeable_refined_zone_still_returned(
        self, mocker: MockerFixture,
    ) -> None:
        # A zone that failed the size filter returns
        # ``is_strong_point=False`` from the validator (short-circuits
        # with NOT_TRADEABLE). The pipeline still emits it — the
        # caller decides what to do (persist? skip?).
        stubs = patch_stages(
            mocker,
            patterns=[make_pattern()],
            refined_results=[make_refined(
                is_tradeable=False, rejection_reason="ZONE_TOO_NARROW",
            )],
            validated_results=[make_validated(is_strong_point=False)],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 1
        assert result[0].is_strong_point is False
        stubs["validate"].assert_called_once()

    def test_non_strong_point_validated_still_returned(
        self, mocker: MockerFixture,
    ) -> None:
        patch_stages(
            mocker,
            patterns=[make_pattern()],
            validated_results=[make_validated(is_strong_point=False)],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 1
        assert result[0].is_strong_point is False

    def test_mixed_strong_and_weak_zones_all_returned(
        self, mocker: MockerFixture,
    ) -> None:
        # Three patterns: one is a Strong Point, one isn't tradeable,
        # one is tradeable but body-broken (is_strong_point=False).
        # All three come through.
        patch_stages(
            mocker,
            patterns=[make_pattern(), make_pattern(), make_pattern()],
            validated_results=[
                make_validated(is_strong_point=True),
                make_validated(is_strong_point=False),
                make_validated(is_strong_point=False),
            ],
        )
        result = run_strategy_pipeline(make_df())
        assert len(result) == 3
        assert sum(1 for v in result if v.is_strong_point) == 1


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
        assert c.impulse_body_to_range_ratio_min == 0.6
        # PR #44 loosened from 1.0.
        assert c.impulse_atr_multiple_min == 0.7
        assert c.atr_period == 14
        assert c.max_impulse_run_candles == 5
        assert c.min_base_candles == 1
        assert c.max_base_candles == 5
        # PR #44 loosened from 0.6.
        assert c.base_range_to_impulse_ratio_max == 1.0
        assert c.base_max_body_to_impulse_body_ratio == 0.4
        assert c.zone_min_size_points == 5.0
        assert c.zone_max_size_points == 80.0
        assert c.sl_buffer_points == 17.5
        assert c.tp1_local_peak_lookback_bars == 50
