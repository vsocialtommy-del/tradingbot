"""Market structure tracking on close-price swings.

Pure-Python logic on pandas DataFrames — no MT5, no Supabase, no I/O.
The orchestrator (``bot.main``) is responsible for fetching tunables
from ``bot_config`` and passing them in via :class:`StructureConfig`.

Glossary
--------
swing high
    A bar whose close equals the maximum close in the window
    ``[i-strength, i+strength]``. Ties are broken to the leftmost bar
    (so a flat top registers exactly once, at its left edge).
swing low
    Symmetric — local minimum on close.
HH / HL / LL / LH
    A swing's label vs the previous swing of the same kind. Equal swing
    high → ``LH`` (no new high made). Equal swing low → ``HL`` (held the
    low). This matches conventional reading.
BoS — Break of Structure
    A later bar's close moves past a swing's price level (``> swing.price``
    for UP, ``< swing.price`` for DOWN). "Body confirmation" means we
    use the close (not high/low) — i.e. a wick poking through the level
    is **not** a BoS.
state
    ``UPTREND`` when the last N classified swings are all HH/HL,
    ``DOWNTREND`` when all LL/LH, otherwise ``RANGE``. ``N`` defaults to
    4 (two highs and two lows alternating).

What this module does **not** do
--------------------------------
- It does not detect W / M patterns — that's
  :mod:`bot.strategy.pattern_detection`.
- It does not place SL — :mod:`bot.exits.sl_manager` consumes
  :func:`get_swings_within_lookback` for that.
- It does not fetch config from Supabase — pass a :class:`StructureConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd
from loguru import logger

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

SwingKind = Literal["HIGH", "LOW"]
SwingLabel = Literal["HH", "HL", "LL", "LH"]
BosDirection = Literal["UP", "DOWN"]


class StructureState(str, Enum):
    """Market regime inferred from the recent swing labels."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Swing:
    """A raw swing pivot, before HH/HL classification."""

    index: int  # bar index in the source DataFrame
    time: pd.Timestamp  # UTC timestamp of the bar
    price: float  # close at the bar
    kind: SwingKind


@dataclass(frozen=True)
class ClassifiedSwing(Swing):
    """A swing tagged with its HH/HL/LL/LH label.

    ``label is None`` for the first swing of each kind (no prior to
    compare against).
    """

    label: SwingLabel | None


@dataclass(frozen=True)
class BosEvent:
    """A close that broke past a previous swing's price level."""

    bar_index: int
    time: pd.Timestamp
    direction: BosDirection
    broken_swing_index: int  # bar index of the swing that was broken
    broken_level: float  # the swing's close price
    break_close: float  # the close that broke it


@dataclass(frozen=True)
class StructureSnapshot:
    """Complete structural read of a DataFrame."""

    swings: list[ClassifiedSwing]
    bos_events: list[BosEvent]  # sorted by bar_index ascending
    state: StructureState
    last_swing_high: ClassifiedSwing | None
    last_swing_low: ClassifiedSwing | None
    last_bos: BosEvent | None


@dataclass(frozen=True)
class StructureConfig:
    """Tunables for structure analysis.

    These mirror keys the orchestrator pulls from the ``bot_config``
    table. Defaults match the spec's open-items section (13).
    """

    swing_strength: int = 3  # bars on each side for swing detection
    trend_min_swings: int = 4  # min classified swings to declare a trend


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def analyze_structure(
    df: pd.DataFrame,
    config: StructureConfig | None = None,
) -> StructureSnapshot:
    """Run the full pipeline: detect → prune → classify → BoS → state."""
    cfg = config or StructureConfig()

    swings_raw = detect_swings(df, cfg.swing_strength)
    swings = _prune_consecutive_same_kind(swings_raw)
    classified = classify_swings(swings)
    bos_events = detect_bos(df, swings)
    state = infer_state(classified, cfg.trend_min_swings)

    last_high = next((s for s in reversed(classified) if s.kind == "HIGH"), None)
    last_low = next((s for s in reversed(classified) if s.kind == "LOW"), None)
    last_bos = bos_events[-1] if bos_events else None

    # Lazy format — loguru skips str-build when DEBUG is filtered.
    # This call fires once per bar in backtest (1000s of times); eager
    # f-string formatting was measurable overhead in profiling.
    logger.debug(
        "structure: bars={} swings={} bos={} state={}",
        len(df), len(swings), len(bos_events), state.value,
    )
    return StructureSnapshot(
        swings=classified,
        bos_events=bos_events,
        state=state,
        last_swing_high=last_high,
        last_swing_low=last_low,
        last_bos=last_bos,
    )


def detect_swings(df: pd.DataFrame, strength: int) -> list[Swing]:
    """Find all swing pivots on close prices.

    A bar at index ``i`` is a swing high iff ``close[i]`` equals the
    maximum close in ``close[i-strength : i+strength+1]`` AND ``i`` is
    the leftmost bar achieving that max within the window. Symmetric for
    swing low. A flat window (max == min) yields no swing.

    Returns swings in chronological order. If ``len(df) < 2*strength+1``
    no swings can be confirmed (no bar has full shoulders) and an empty
    list is returned.
    """
    if strength < 1:
        raise ValueError(f"strength must be >= 1, got {strength}")
    if "close" not in df.columns:
        raise ValueError("df must have a 'close' column")

    n = len(df)
    if n < 2 * strength + 1:
        return []

    closes = df["close"].to_numpy()
    times = df.index
    swings: list[Swing] = []

    for i in range(strength, n - strength):
        window = closes[i - strength : i + strength + 1]
        wmax = window.max()
        wmin = window.min()
        if wmax == wmin:
            # Perfectly flat window — no swing.
            continue

        center = closes[i]
        # The center's offset within the window is exactly `strength`.
        if center == wmax and int(window.argmax()) == strength:
            swings.append(
                Swing(index=i, time=times[i], price=float(center), kind="HIGH")
            )
        elif center == wmin and int(window.argmin()) == strength:
            swings.append(
                Swing(index=i, time=times[i], price=float(center), kind="LOW")
            )
        # A bar can't be both strict max and strict min in a non-flat
        # window, so `elif` is safe.

    return swings


def classify_swings(swings: list[Swing]) -> list[ClassifiedSwing]:
    """Tag each swing with HH / HL / LL / LH vs the previous same-kind swing.

    The first swing of each kind has ``label=None``. Equal-price ties:
    equal swing high → ``LH`` (no new high made), equal swing low →
    ``HL`` (held the low). This is a deliberate convention so a
    double-top reads bearish and a double-bottom reads bullish.
    """
    last_high: Swing | None = None
    last_low: Swing | None = None
    out: list[ClassifiedSwing] = []

    for s in swings:
        label: SwingLabel | None = None
        if s.kind == "HIGH":
            if last_high is not None:
                label = "HH" if s.price > last_high.price else "LH"
            last_high = s
        else:  # LOW
            if last_low is not None:
                label = "HL" if s.price >= last_low.price else "LL"
            last_low = s

        out.append(
            ClassifiedSwing(
                index=s.index,
                time=s.time,
                price=s.price,
                kind=s.kind,
                label=label,
            )
        )
    return out


def detect_bos(df: pd.DataFrame, swings: list[Swing]) -> list[BosEvent]:
    """Find the first close that breaks past each swing.

    Each swing produces at most one event — its first break. Subsequent
    re-tests are ignored. Events are returned sorted by ``bar_index``,
    which means "first BoS in time" is ``events[0]``.

    Body confirmation: we compare against ``close``, not ``high`` /
    ``low``. A wick poking through the level does not register.
    """
    if not swings:
        return []

    closes = df["close"].to_numpy()
    times = df.index
    n = len(df)
    events: list[BosEvent] = []

    for swing in swings:
        for i in range(swing.index + 1, n):
            close = float(closes[i])
            if swing.kind == "HIGH" and close > swing.price:
                events.append(
                    BosEvent(
                        bar_index=i,
                        time=times[i],
                        direction="UP",
                        broken_swing_index=swing.index,
                        broken_level=swing.price,
                        break_close=close,
                    )
                )
                break
            if swing.kind == "LOW" and close < swing.price:
                events.append(
                    BosEvent(
                        bar_index=i,
                        time=times[i],
                        direction="DOWN",
                        broken_swing_index=swing.index,
                        broken_level=swing.price,
                        break_close=close,
                    )
                )
                break

    events.sort(key=lambda e: e.bar_index)
    return events


def infer_state(
    classified: list[ClassifiedSwing],
    min_swings: int = 4,
) -> StructureState:
    """Declare ``UPTREND`` / ``DOWNTREND`` / ``RANGE`` from recent labels.

    Looks at the last ``min_swings`` classified swings. If they're all
    HH/HL → ``UPTREND``. All LL/LH → ``DOWNTREND``. Anything mixed →
    ``RANGE``. Fewer than ``min_swings`` labelled → ``RANGE`` (we don't
    have enough evidence yet).
    """
    labelled = [s for s in classified if s.label is not None]
    if len(labelled) < min_swings:
        return StructureState.RANGE

    recent = labelled[-min_swings:]
    if all(s.label in ("HH", "HL") for s in recent):
        return StructureState.UPTREND
    if all(s.label in ("LL", "LH") for s in recent):
        return StructureState.DOWNTREND
    return StructureState.RANGE


def get_swings_within_lookback(
    swings: list[Swing],
    from_bar: int,
    lookback: int,
) -> list[Swing]:
    """Return swings whose ``index`` is in ``(from_bar - lookback, from_bar]``.

    Historical helper: ``sl_manager`` used this for the lookback
    heuristic that selected the SL anchor swing. PR #31 moved anchor
    selection into the strategy layer (``strong_point.validate_strong_point``
    publishes ``sl_anchor_swing`` on the ValidatedZone), so this
    helper has no production callers any more. Retained for future
    re-use and its own test coverage.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    return [s for s in swings if from_bar - lookback < s.index <= from_bar]


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _prune_consecutive_same_kind(swings: list[Swing]) -> list[Swing]:
    """Collapse runs of same-kind swings into the most extreme one.

    Without pruning, the swing detector can emit two consecutive HIGHs
    (or LOWs) when there's no opposing local extreme between them. This
    breaks the HH/HL alternation classification expects. We keep only
    the highest of each consecutive HIGH run and the lowest of each
    consecutive LOW run.

    Equal prices: keep the earlier swing (don't replace).
    """
    if not swings:
        return swings

    pruned: list[Swing] = []
    for s in swings:
        if pruned and pruned[-1].kind == s.kind:
            prev = pruned[-1]
            replace = (s.kind == "HIGH" and s.price > prev.price) or (
                s.kind == "LOW" and s.price < prev.price
            )
            if replace:
                pruned[-1] = s
            # else: keep the existing extreme
        else:
            pruned.append(s)
    return pruned
