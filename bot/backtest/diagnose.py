"""Setup-detection funnel — diagnose where setups are dying.

Walks the same bar-by-bar loop the engine does, but at each stage
records a count. The output funnel reveals exactly which filter is
the bottleneck::

    Funnel over 800 bars (700 detection bars):
      Pattern candidates:               423      (RBR: 102, DBD: 98,
                                                  DBR: 119, RBD: 104)
      Tradeable refined zones:          312      (74% — size filter)
      Strong Points:                     48      (15% — break + close)
      Unique zones (post-dedup):         12

Usage::

    python -m bot.backtest.diagnose path/to/xauusd_m5.csv
    python -m bot.backtest.diagnose path/to/xauusd_m5.csv --bars 1000
    python -m bot.backtest.diagnose path/to/xauusd_m5.csv --json

PR #31 rewrite: replaces the W/M + Imbalance funnel with the S&D
methodology (pattern_detection → refine → Strong Point).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from bot.backtest.data_loader import load_dukascopy_csv
from bot.strategy.pattern_detection import (
    PatternConfig,
    PatternType,
    detect_patterns,
)
from bot.strategy.strong_point import StrongPointConfig, validate_strong_point
from bot.strategy.structure import StructureConfig, analyze_structure
from bot.strategy.zone_marking import mark_zone
from bot.strategy.zone_refinement import RefinementConfig, refine_zone


@dataclass
class FunnelCounts:
    """Per-stage cumulative counts across the diagnostic run."""

    bars_processed: int = 0
    detection_bars: int = 0

    pattern_candidates: int = 0
    """Patterns returned by ``detect_patterns`` — already past impulse +
    base tightness gates inside detection."""
    rbr_candidates: int = 0
    dbd_candidates: int = 0
    dbr_candidates: int = 0
    rbd_candidates: int = 0

    refined_tradeable: int = 0
    refined_too_narrow: int = 0
    refined_too_wide: int = 0

    strong_points: int = 0
    """Passed break-and-close validation."""
    # Failures are mutually exclusive (primary failure only). Sum
    # exactly equals (refined_tradeable − strong_points).
    sp_failed_no_swing: int = 0
    """No structural swing exists on the opposite side of the zone."""
    sp_failed_no_sl_anchor: int = 0
    """No same-side swing exists for the SL to pin to."""
    sp_failed_no_break_yet: int = 0
    """Pending — break + close hasn't happened yet."""
    sp_failed_invalidated: int = 0
    """Opposite-side body close happened before any valid break."""

    unique_zones: int = 0
    """Setups the engine would create after deduplication
    by ``(direction, round(top, 2), round(bottom, 2))``."""

    def funnel_lines(self) -> list[str]:
        """Human-readable funnel rows."""
        rows = [
            ("Bars processed", str(self.bars_processed)),
            ("Detection bars", str(self.detection_bars)),
            ("", ""),
            (
                "Pattern candidates",
                f"{self.pattern_candidates:,}  "
                f"(RBR: {self.rbr_candidates:,}, "
                f"DBD: {self.dbd_candidates:,}, "
                f"DBR: {self.dbr_candidates:,}, "
                f"RBD: {self.rbd_candidates:,})",
            ),
            (
                "Refined tradeable",
                f"{self.refined_tradeable:,}  "
                f"({_pct(self.refined_tradeable, self.pattern_candidates)})",
            ),
            ("  ↳ rejected too narrow", str(self.refined_too_narrow)),
            ("  ↳ rejected too wide", str(self.refined_too_wide)),
            (
                "Strong Points",
                f"{self.strong_points:,}  "
                f"({_pct(self.strong_points, self.refined_tradeable)})",
            ),
            ("  ↳ failed: no opposite swing", str(self.sp_failed_no_swing)),
            ("  ↳ failed: no SL anchor", str(self.sp_failed_no_sl_anchor)),
            ("  ↳ failed: no break yet (pending)",
             str(self.sp_failed_no_break_yet)),
            ("  ↳ failed: invalidated", str(self.sp_failed_invalidated)),
            (
                "Unique zones (post-dedup)",
                f"{self.unique_zones:,}  "
                f"({_pct(self.unique_zones, self.strong_points)})",
            ),
        ]
        return [
            f"{label:.<40s} {value}" if label and value else ""
            for label, value in rows
        ]


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "—"
    return f"{part / whole * 100:.0f}%"


def diagnose(
    df: pd.DataFrame,
    *,
    pattern_config: PatternConfig | None = None,
    refinement_config: RefinementConfig | None = None,
    strong_point_config: StrongPointConfig | None = None,
    min_history_bars: int = 100,
    pipeline_window_bars: int = 250,
) -> FunnelCounts:
    """Walk ``df`` bar-by-bar; return per-stage cumulative counts.

    Mirrors the engine's pipeline gate but instruments every stage.
    """
    pcfg = pattern_config or PatternConfig()
    rcfg = refinement_config or RefinementConfig()
    scfg = strong_point_config or StrongPointConfig()

    counts = FunnelCounts(bars_processed=len(df))
    seen_zones: set[tuple[str, float, float]] = set()

    if len(df) <= min_history_bars:
        return counts

    for i in range(min_history_bars, len(df)):
        counts.detection_bars += 1
        window_start = max(0, i + 1 - pipeline_window_bars)
        history = df.iloc[window_start : i + 1]

        structure = analyze_structure(
            history,
            StructureConfig(
                swing_strength=2,  # match pipeline default
            ),
        )
        swings = list(structure.swings)
        patterns = detect_patterns(history, pcfg)

        for p in patterns:
            counts.pattern_candidates += 1
            if p.pattern_type == PatternType.RBR:
                counts.rbr_candidates += 1
            elif p.pattern_type == PatternType.DBD:
                counts.dbd_candidates += 1
            elif p.pattern_type == PatternType.DBR:
                counts.dbr_candidates += 1
            elif p.pattern_type == PatternType.RBD:
                counts.rbd_candidates += 1

        for pattern in patterns:
            zone = mark_zone(pattern, history)
            refined = refine_zone(zone, history, rcfg)
            if not refined.is_tradeable:
                if refined.rejection_reason == "ZONE_TOO_NARROW":
                    counts.refined_too_narrow += 1
                elif refined.rejection_reason == "ZONE_TOO_WIDE":
                    counts.refined_too_wide += 1
                continue
            counts.refined_tradeable += 1

            validated = validate_strong_point(refined, history, swings, scfg)
            if not validated.is_strong_point:
                # Primary (first listed) failure — mutually exclusive
                # buckets so they sum exactly to
                # (refined_tradeable − strong_points).
                primary = (
                    validated.validation_failures[0]
                    if validated.validation_failures else "OTHER"
                )
                if primary in ("NO_SWING_ABOVE", "NO_SWING_BELOW"):
                    counts.sp_failed_no_swing += 1
                elif primary == "NO_SL_ANCHOR":
                    counts.sp_failed_no_sl_anchor += 1
                elif primary == "NO_BREAK_YET":
                    counts.sp_failed_no_break_yet += 1
                elif primary == "INVALIDATED":
                    counts.sp_failed_invalidated += 1
                # NOT_TRADEABLE never reaches here (refined.is_tradeable
                # already filtered above).
                continue
            counts.strong_points += 1

            zone_key = (
                validated.direction,
                round(validated.top, 2),
                round(validated.bottom, 2),
            )
            if zone_key in seen_zones:
                continue
            seen_zones.add(zone_key)
            counts.unique_zones += 1

    return counts


def _format_report(counts: FunnelCounts, source: str) -> str:
    head = f"Funnel — {source}"
    rule = "─" * max(len(head), 50)
    body = "\n".join(line for line in counts.funnel_lines() if line)
    return f"\n{head}\n{rule}\n{body}\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bot.backtest.diagnose",
        description=(
            "Print a setup-detection funnel for a CSV of OHLC data. "
            "Reveals which filter stage is dropping setups."
        ),
    )
    p.add_argument("csv", type=Path, help="OHLC CSV (Dukascopy / MT5 / ISO)")
    p.add_argument(
        "--bars", type=int, default=None,
        help="Limit to the most recent N bars (default: all)",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit JSON instead of the human-readable funnel",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    df = load_dukascopy_csv(args.csv)
    if args.bars is not None and len(df) > args.bars:
        df = df.iloc[-args.bars :]
    counts = diagnose(df)
    if args.json:
        print(json.dumps(asdict(counts), indent=2))
    else:
        print(_format_report(counts, source=str(args.csv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
