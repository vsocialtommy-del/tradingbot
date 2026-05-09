"""Setup-detection funnel — diagnose where setups are dying.

Walks the same bar-by-bar loop the engine does, but at each stage of
the strategy pipeline records a count: how many candidates entered,
how many survived. The output funnel reveals exactly which filter is
the bottleneck:

::

    Funnel over 800 bars (700 detection bars):
      Swings detected (cumulative):     1,247
      Pattern candidates:               423      (W: 215, M: 208)
      Passed pattern filters:           147      (35% — tolerance + peak)
      Tradeable refined zones:          112      (76% — size filter)
      Strong Points:                     38      (34% — BoS + base + impulse)
      Imbalance-qualified zones:         11      (29% — ≥2 approaches)
      Untapped at detection:              8      (73% — already tapped if 0)
      Setups created (post dedup):        3
      Trigger fills:                      0      ← here's where they all die

Usage::

    python -m bot.backtest.diagnose path/to/xauusd_m5.csv
    python -m bot.backtest.diagnose path/to/xauusd_m5.csv --bars 1000
    python -m bot.backtest.diagnose path/to/xauusd_m5.csv --json

The diagnostic is independent of ``BacktestEngine`` so it can be run
standalone — useful for quick "why are no trades firing?" checks
without launching the whole backtest.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from bot.backtest.data_loader import load_dukascopy_csv
from bot.strategy.imbalance import ImbalanceConfig, track_imbalance
from bot.strategy.pattern_detection import (
    MPattern,
    PatternConfig,
    WPattern,
    detect_m_patterns,
    detect_w_patterns,
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
    """Patterns returned by detect_w/m_patterns BEFORE downstream
    filtering. Already past tolerance + peak threshold (those are
    inside the detectors)."""
    w_candidates: int = 0
    m_candidates: int = 0
    refined_tradeable: int = 0
    """Survived the size-filter (5-80 points by default)."""
    refined_too_narrow: int = 0
    refined_too_wide: int = 0
    strong_points: int = 0
    """Passed BoS + base-compactness + impulse-strength validation."""
    # Failures are categorised by the FIRST listed failure reason —
    # mutually exclusive so the bucket counts sum to (refined_tradeable
    # − strong_points).
    sp_failed_no_bos: int = 0
    sp_failed_base: int = 0
    sp_failed_impulse: int = 0
    sp_failed_other: int = 0
    # Imbalance breakdown — why Strong Points failed to become tradeable.
    imbalance_qualified: int = 0
    """Tracked ≥ approach_threshold approaches before tap."""
    imb_zero_approaches: int = 0
    imb_too_few_approaches: int = 0
    untapped_at_detection: int = 0
    """Imbalance-qualified AND not tapped at the time we detect it.
    A tapped zone is one price has already entered → first-touch
    consumed → not a fresh setup."""
    imb_qualified_but_tapped: int = 0
    unique_zones: int = 0
    """Setups the engine would create after deduplication."""

    # Per-tick-driven (not stage-by-stage) — populated only when
    # ``include_triggers=True``.
    triggers_fired: int | None = None

    def funnel_lines(self) -> list[str]:
        """Return human-readable funnel rows."""
        rows = [
            ("Bars processed", str(self.bars_processed)),
            ("Detection bars", str(self.detection_bars)),
            ("", ""),
            ("Pattern candidates",
             f"{self.pattern_candidates:,}  "
             f"(W: {self.w_candidates:,}, M: {self.m_candidates:,})"),
            ("Refined tradeable",
             f"{self.refined_tradeable:,}"
             f"  ({_pct(self.refined_tradeable, self.pattern_candidates)})"),
            ("  ↳ rejected too narrow", str(self.refined_too_narrow)),
            ("  ↳ rejected too wide", str(self.refined_too_wide)),
            ("Strong Points",
             f"{self.strong_points:,}"
             f"  ({_pct(self.strong_points, self.refined_tradeable)})"),
            ("  ↳ failed: no BoS", str(self.sp_failed_no_bos)),
            ("  ↳ failed: base not compact", str(self.sp_failed_base)),
            ("  ↳ failed: impulse too weak", str(self.sp_failed_impulse)),
            ("  ↳ failed: other", str(self.sp_failed_other)),
            ("Imbalance-qualified",
             f"{self.imbalance_qualified:,}"
             f"  ({_pct(self.imbalance_qualified, self.strong_points)})"),
            ("  ↳ rejected: 0 approaches", str(self.imb_zero_approaches)),
            ("  ↳ rejected: <threshold approaches",
             str(self.imb_too_few_approaches)),
            ("Untapped at detection",
             f"{self.untapped_at_detection:,}"
             f"  ({_pct(self.untapped_at_detection, self.imbalance_qualified)})"),
            ("  ↳ rejected: tapped already",
             str(self.imb_qualified_but_tapped)),
            ("Unique zones (post-dedup)",
             f"{self.unique_zones:,}"
             f"  ({_pct(self.unique_zones, self.untapped_at_detection)})"),
        ]
        if self.triggers_fired is not None:
            rows.append(("", ""))
            rows.append(("Trigger fills",
                         f"{self.triggers_fired:,}"
                         f"  ({_pct(self.triggers_fired, self.unique_zones)} of zones)"))
        return [
            f"{label:.<35s} {value}" if label and value
            else ""
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
    imbalance_config: ImbalanceConfig | None = None,
    min_history_bars: int = 100,
    pipeline_window_bars: int = 250,
) -> FunnelCounts:
    """Walk ``df`` bar-by-bar; return per-stage cumulative counts.

    Mirrors the engine's pipeline gate but instruments every stage.
    Does NOT track triggers (no broker simulation here) — see the
    ``BacktestResult.metrics`` for live trigger / fill counts when you
    want the full picture.
    """
    pcfg = pattern_config or PatternConfig()
    rcfg = refinement_config or RefinementConfig()
    scfg = strong_point_config or StrongPointConfig()
    icfg = imbalance_config or ImbalanceConfig()

    counts = FunnelCounts(bars_processed=len(df))
    seen_zones: set[tuple[str, float, float]] = set()

    if len(df) <= min_history_bars:
        return counts

    for i in range(min_history_bars, len(df)):
        counts.detection_bars += 1
        window_start = max(0, i + 1 - pipeline_window_bars)
        history = df.iloc[window_start : i + 1]

        structure = analyze_structure(
            history, StructureConfig(swing_strength=pcfg.swing_strength),
        )
        swings = list(structure.swings)
        ws = detect_w_patterns(history, pcfg, swings=swings)
        ms = detect_m_patterns(history, pcfg, swings=swings)
        counts.w_candidates += len(ws)
        counts.m_candidates += len(ms)
        counts.pattern_candidates += len(ws) + len(ms)

        for pattern in (*ws, *ms):
            initial = mark_zone(pattern, history)
            refined = refine_zone(initial, history, rcfg)
            if not refined.is_tradeable:
                if refined.rejection_reason == "ZONE_TOO_NARROW":
                    counts.refined_too_narrow += 1
                elif refined.rejection_reason == "ZONE_TOO_WIDE":
                    counts.refined_too_wide += 1
                continue
            counts.refined_tradeable += 1

            validated = validate_strong_point(
                refined, history, structure.bos_events, scfg,
            )
            if not validated.is_strong_point:
                # Categorise by the FIRST listed failure (priority order:
                # NO_BOS_EVENT > BASE_NOT_COMPACT > IMPULSE_TOO_WEAK >
                # other). Mutually exclusive so the four buckets sum
                # exactly to (refined_tradeable − strong_points).
                fails = list(validated.validation_failures)
                primary = fails[0] if fails else "OTHER"
                if primary == "NO_BOS_EVENT":
                    counts.sp_failed_no_bos += 1
                elif primary == "BASE_NOT_COMPACT":
                    counts.sp_failed_base += 1
                elif primary == "IMPULSE_TOO_WEAK":
                    counts.sp_failed_impulse += 1
                else:
                    counts.sp_failed_other += 1
                continue
            counts.strong_points += 1

            imbalance = track_imbalance(validated, history, icfg)
            if not imbalance.is_imbalance:
                if imbalance.approach_count == 0:
                    counts.imb_zero_approaches += 1
                else:
                    counts.imb_too_few_approaches += 1
                continue
            counts.imbalance_qualified += 1

            if imbalance.is_tapped:
                counts.imb_qualified_but_tapped += 1
                continue
            counts.untapped_at_detection += 1

            zone_key = (
                imbalance.direction,
                round(imbalance.top, 2),
                round(imbalance.bottom, 2),
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
