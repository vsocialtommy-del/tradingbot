"""Zone visibility — PR #48.

"The bot has eyes at current price. Looking back in time, it can only
see zones that haven't been obscured by candles bodying through them."

This module is the pure-logic implementation of that rule. The caller
(``main._try_place_setup`` for the new-placement gate;
``main._run_zone_death_pass`` for the existing-setup WAITING-layer
cancellation) provides the OHLC dataframe, the zone bounds, and the
reference timestamp (zone formation or flip).

Rule
----

A candle "bodies through the zone" iff its body — the interval
``[min(open, close), max(open, close)]`` — overlaps the zone's price
range ``[zone.bottom, zone.top]``:

::

    body_bottom = min(open, close)
    body_top    = max(open, close)
    crossed     = body_bottom < zone_top AND body_top > zone_bottom

**Wicks alone don't count.** A bar that wicks into the zone but whose
body stays outside is not obscuring the zone. This matches the
operator's body-close-based intuition (see also ``strong_point.py``
which uses body closes for break detection).

Visibility verdicts:

* **0 bodies through since reference time**: zone is fresh — tradeable
  in its original direction. (Pipeline path: place setup.)
* **1 body through**: zone has been flipped. Tradeable only in the
  flipped direction (PR #38's ``_load_flipped_candidates`` path).
* **2 or more bodies through**: zone is obscured — dead, not tradeable
  in any direction.

This module returns the raw count; the caller decides what threshold
to apply (the pipeline gate uses ``< 1``; the flipped path's existing
body-broken-since-flip check has its own semantics).

What's NOT here
---------------

* Reading the OHLC dataframe — caller's responsibility (use
  ``self._latest_ohlc``).
* Reading zone bounds from Supabase — caller resolves zone metadata
  before invoking.
* Deciding when to run the check — caller wires it into the per-tick
  loop or M5-close pass as appropriate.
"""

from __future__ import annotations

import pandas as pd


def count_bodies_through_zone(
    df: pd.DataFrame,
    *,
    zone_top: float,
    zone_bottom: float,
    since_time: pd.Timestamp,
) -> int:
    """Count completed candles AFTER ``since_time`` whose body overlaps
    the zone price range ``[zone_bottom, zone_top]``.

    Parameters
    ----------
    df
        OHLC bars indexed by tz-aware timestamps. Must contain
        ``open`` and ``close`` columns. Wicks (``high`` / ``low``) are
        intentionally NOT consulted — only bodies count.
    zone_top, zone_bottom
        Zone price bounds. Must satisfy ``zone_top >= zone_bottom``
        (caller enforces; we don't assert here to keep this hot-path
        cheap).
    since_time
        Reference timestamp. Strictly greater-than filter — bars with
        ``index == since_time`` are excluded (the formation bar itself,
        or the flip bar, IS in the zone's body by construction and
        shouldn't count against visibility).

    Returns
    -------
    int
        Count of overlapping-body bars. Caller compares against a
        threshold:

        * pipeline placement gate: ``count == 0`` ⇒ visible.
        * pipeline existing-setup cancellation: ``count >= 1`` ⇒ dead.
        * (flipped path uses its own helper; not this function.)
    """
    if len(df) == 0:
        return 0
    post = df[df.index > since_time]
    if len(post) == 0:
        return 0
    body_top = post[["open", "close"]].max(axis=1)
    body_bottom = post[["open", "close"]].min(axis=1)
    overlaps = (body_bottom < zone_top) & (body_top > zone_bottom)
    return int(overlaps.sum())


def is_zone_visible_for_pipeline(
    df: pd.DataFrame,
    *,
    zone_top: float,
    zone_bottom: float,
    formed_at: pd.Timestamp,
) -> bool:
    """True iff the zone is tradeable by the pipeline (original-direction) path.

    A pipeline setup is valid only if NO candle since formation has
    bodied through the zone. Any body-through means the zone has
    either flipped (1) or died (2+); either way the original direction
    is no longer the right thesis.

    The flipped path (PR #38) has its own per-flip safety guard
    (``flipped_zone_body_broken_since_flip``); this helper isn't used
    by that path.
    """
    return count_bodies_through_zone(
        df,
        zone_top=zone_top,
        zone_bottom=zone_bottom,
        since_time=formed_at,
    ) == 0
