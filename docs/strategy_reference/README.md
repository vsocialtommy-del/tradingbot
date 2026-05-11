# Strategy reference — institutional Supply & Demand methodology

This directory holds the canonical visual + text reference for the
bot's trading methodology. The bot codename calls patterns "W" and "M"
because that's what they look like on a line chart — but the
underlying algorithm is **institutional Supply & Demand**, built on
four base patterns and four setup types.

## The four base patterns

Every tradeable zone is the **base** of one of these patterns:

| Pattern | Composition | Zone | Direction |
|---|---|---|---|
| **RBR** | Rally → Base → Rally | Demand | BUY (continuation) |
| **DBD** | Drop → Base → Drop | Supply | SELL (continuation) |
| **DBR** | Drop → Base → Rally | Demand | BUY (reversal) |
| **RBD** | Rally → Base → Drop | Supply | SELL (reversal) |

The **base** is 1-5 candles of compact consolidation between two
**impulse** moves. The zone is marked from the base's candle bodies
only (`top = max(max(open, close))`, `bottom = min(min(open, close))`
across all base candles); wicks are excluded.

### Impulse criteria

* Body-to-range ratio ≥ 0.6 (mostly body, not wick)
* Body size ≥ 1.0 × ATR(14) (significant move relative to recent vol)
* An impulse can span 1-5 consecutive same-direction candles each
  meeting both criteria. A weak / opposite-direction candle ends the
  run.

### Base criteria

* 1-5 candles between two impulses
* Total base range ≤ 0.6 × mean(impulse_before.range, impulse_after.range)
* Largest base candle body ≤ 0.4 × largest impulse body

## The four setup types

All four setups trade the zones (demand / supply) from the four base
patterns. They differ in **what extra confirmation** is required
before entering. We implement them in order of complexity:

| # | Setup | Phase | Complexity |
|---|---|---|---|
| 1 | **Strong Point** | **v1 — implemented now** | Lowest |
| 2 | CHoCH | future | Medium |
| 3 | Imbalance Zone | future | Medium-high |
| 4 | SnD Flip | future | Highest |

The setups share the same RBR/DBD/DBR/RBD foundation — only the
validation logic differs. Each future setup is a separate validator
function that consumes a `Pattern` and returns a tradeable verdict.

---

## Setup 1: Strong Point

> References: `03_strong_point_for_sell.png`, `03b_strong_point_for_buy.png`

The simplest setup. No trend context required — the validation is
purely local to the zone.

### SELL (DBD or RBD pattern → supply zone)

1. Pattern forms with a supply zone (the base of a DBD or RBD).
2. After the pattern forms, price **breaks below the nearest swing
   low** with a **body close** below it (not just a wick).
3. Price **pulls back to the same supply zone** = entry trigger.
4. **TP1** = the same swing low that was broken (BoS level).
5. **SL** = 15-20 pips **above the nearest swing high** (NOT just
   above the zone). The "nearest swing high" is the swing that
   immediately precedes the zone — protects against liquidity grabs.

### BUY (RBR or DBR pattern → demand zone)

Mirror. Break above nearest swing high with body close; pull back to
demand zone; TP1 = the broken swing high; SL = 15-20 pips below
nearest swing low.

### Re-entry rules

* **Fresh zone** (first retest) = highest probability. Logged for
  analytics but doesn't block subsequent entries.
* Each subsequent retest is **also** tradeable as long as TP1 hasn't
  fired and the zone hasn't been invalidated.
* Entry can fire **immediately on first touch** (no wait for confirmation).

### Invalidation

* **Opposite-side body close**: BUY zone invalidated by a bar that
  closes (body) below the zone bottom; SELL mirror. Zone becomes
  untradeable until a new pattern forms.
* **TP1 hit**: cycle complete. Zone becomes a candidate for the
  future SnD Flip setup.

---

## Setup 2: CHoCH (Change of Character) — future

> References: `01_choch_for_buy.png`, `02_choch_for_sell.png`

A trend-reversal setup. Requires established trend context.

### CHoCH for BUY

1. Market shows **clear downtrend** — at least 2 Lower Highs (LH) and
   2 Lower Lows (LL).
2. A Strong Point for SELL must exist that price now needs to break.
3. **Bullish CHoCH**: price breaks above the **most recent LH** with
   a body close (not wick).
4. The break must be a **clear body close** with **strong momentum**
   (strong-bodied candles). Weak breaks / wick-only breaks =
   unreliable.
5. The Break of Structure (BOS) that follows confirms the trend
   reversal.
6. **TP1** = the nearest opposing zone = the first supply zone (POI).
7. **SL** = below the CHoCH demand zone + 5-10 pip buffer.

### CHoCH for SELL

Mirror. Clear uptrend (2 HH + 2 HL), bearish CHoCH = body close below
most recent HH, TP1 = first demand zone, SL = above CHoCH supply
zone + 5-10 pip buffer.

CHoCH is **meaningless without an established trend** — the algorithm
must verify the trend context before flagging a CHoCH candidate.

---

## Setup 3: Imbalance Zone — future

> References: `06_imbalance_zone_for_buy.png`, `07_imbalance_zone_for_sell.png`

A "pent-up strength" signal. Builds on a fresh Strong Point zone.

### Imbalance for BUY (fresh demand zone)

1. Identify a **fresh** Strong Point demand zone (never touched).
2. Price must **approach the zone at least 2 times**, both attempts
   **failing to enter**. Price gets close (a few pips away) but
   never actually taps into the zone body.
3. After the **second failed attempt**, the zone is reclassified as
   an **Imbalance Zone**.
4. The pent-up strength signals a high-probability push toward TP2
   on the first real touch.
5. **SL** = 15-20 pips below the nearest swing low.

### Imbalance for SELL

Mirror. Fresh supply zone + 2 failed approaches → Imbalance Zone →
high-probability push to TP2 on first touch. SL = 15-20 pips above
nearest swing high.

The Imbalance qualification is an **upgrade** on top of Strong Point —
it doesn't replace Strong Point. A zone can be a tradeable Strong
Point on first touch even before it becomes an Imbalance Zone.

---

## Setup 4: SnD Flip — future

> References: `04_snd_flip_for_buy.png`, `05_snd_flip_for_sell.png`

The most complex setup. Combines a Strong Point cycle (now complete)
with a new opposite-direction zone formed in or near the old one.

### SnD Flip for BUY

1. **Pre-condition**: an existing **supply zone** (Strong Point for
   SELL) whose cycle has **completed** — it already broke the
   nearest low and had its quality setup.
2. The new **demand zone** (RBR) can form **outside, inside, or
   above** the old supply zone.
3. **Clean break**: price must break **above** the supply zone with
   a **full body close** above it.
4. The new demand zone must **break the nearest swing high** after
   forming — confirms strength.
5. The old supply zone + new demand zone are **combined** into a
   single Flip zone.
6. **Entry**: wait for pullback to the new demand zone → enter
   confidently on first touch (this setup is entered more
   aggressively).
7. **TP1** = the nearest high closest to the new demand zone.
8. **SL** = 15-20 pips below the new demand zone.

### SnD Flip for SELL

Mirror. Old demand zone (Strong Point cycle complete) + new supply
zone (DBD) breaking above, then below; combined Flip zone; TP1 =
nearest low; SL = 15-20 pips above.

---

## Reference images

The screenshots in this directory are the canonical visual reference.
Filenames map to setups as follows:

| File | Setup |
|---|---|
| `01_choch_for_buy.png` | CHoCH for BUY |
| `02_choch_for_sell.png` | CHoCH for SELL |
| `03_strong_point_for_sell.png` | **Strong Point for SELL (v1)** |
| `03b_strong_point_for_buy.png` | **Strong Point for BUY (v1, mirror of 03)** |
| `04_snd_flip_for_buy.png` | SnD Flip for BUY |
| `05_snd_flip_for_sell.png` | SnD Flip for SELL |
| `06_imbalance_zone_for_buy.png` | Imbalance Zone for BUY |
| `07_imbalance_zone_for_sell.png` | Imbalance Zone for SELL |

PNGs are dropped into this directory manually (chat-attached images
can't be saved to disk programmatically). If a filename in the table
above doesn't exist as a file yet, the spec text above is still
authoritative — the PNGs are visual confirmation, not the source of
truth.

## Why the codebase uses "W" and "M" terminology in some places

Earlier iterations of the strategy used "W pattern" and "M pattern"
as a shorthand — what the patterns LOOK like on a line chart. Those
terms persist in some places (e.g. legacy variable names, the
deprecated `bot/strategy/imbalance.py`) but the **correct
terminology is RBR / DBD / DBR / RBD**. New code should use the S&D
names.

A "W" on a line chart is most often a **DBR** (drop-base-rally) where
the base sits at the trough of the W. An "M" is most often an **RBD**.
But neither mapping is one-to-one — RBR and DBD shapes also exist
that don't look like W or M.
