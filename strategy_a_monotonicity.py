"""
Strategy A: Monotonicity-violator scanner.

Finds clusters of related markets where probabilities MUST satisfy ordering
constraints. Two flavors tested:

  (1) Over/Under sports markets: P(total > x) >= P(total > y) for x < y
  (2) Date-based "by date Y" markets: P(by earlier) <= P(by later)

For each cluster, check current prices for arbitrage violations and quantify
the magnitude.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

markets = json.loads(Path("markets.json").read_text())

def parse_prices(m):
    """Return Yes-price for binary markets, else None."""
    p = m.get("outcomePrices")
    if not p:
        return None
    try:
        arr = json.loads(p) if isinstance(p, str) else p
    except json.JSONDecodeError:
        return None
    if not arr or len(arr) < 2:
        return None
    try:
        return float(arr[0])  # Yes price
    except (TypeError, ValueError):
        return None

def has_liquidity(m, min_liq=200, min_v24=50):
    try:
        l = float(m.get("liquidityNum") or 0)
        v = float(m.get("volume24hr") or 0)
    except (TypeError, ValueError):
        return False
    return l >= min_liq and v >= min_v24

# ============================================================
# Approach 1: Over/Under monotonicity
# ============================================================
# Look at markets with groupItemTitle like "O/U 2.5", "Over/Under 1.5", etc.
# These are within an event - we need to group BY EVENT, not by O/U value.
# The event group is usually identified by the events array or the eventId.

print("=" * 70)
print("APPROACH 1: Over/Under monotonicity within sports events")
print("=" * 70)

# We need to find groups of markets that are O/U variants of SAME event.
# Use the 'events' field if present, else fall back to slug parsing.

ou_pattern = re.compile(r"^(O/U|Over/Under|Over\b|Under\b)\s*([\d.]+)$", re.IGNORECASE)

def parse_ou_threshold(title):
    if not title:
        return None
    m = ou_pattern.match(title.strip())
    if m:
        try:
            return float(m.group(2))
        except ValueError:
            return None
    return None

# Group markets by their parent event. Polymarket includes an `events` array.
event_buckets = defaultdict(list)
for mk in markets:
    title = mk.get("groupItemTitle", "")
    threshold = parse_ou_threshold(title)
    if threshold is None:
        continue
    # Find event ID (eventID / events array)
    ev_id = None
    if mk.get("events"):
        try:
            ev_id = mk["events"][0].get("id") or mk["events"][0].get("slug")
        except (KeyError, IndexError, AttributeError, TypeError):
            ev_id = None
    if ev_id is None:
        # fallback: strip the O/U part from slug
        slug = mk.get("slug", "")
        ev_id = re.sub(r"-(o-u|over|under)[\d-]*$", "", slug)
    price = parse_prices(mk)
    if price is None:
        continue
    event_buckets[ev_id].append({
        "id": mk.get("id"),
        "question": mk.get("question"),
        "threshold": threshold,
        "yes_price": price,
        "liquidity": float(mk.get("liquidityNum") or 0),
        "v24": float(mk.get("volume24hr") or 0),
        "slug": mk.get("slug"),
        "end": mk.get("endDate"),
    })

# Filter to events with >=2 O/U thresholds AND non-zero liquidity
ou_events = {k: v for k, v in event_buckets.items() if len(v) >= 2}
print(f"Events with >=2 O/U markets: {len(ou_events):,}")

# Now check monotonicity: for x < y, must have P(>x) >= P(>y)
# (yes_price represents P(YES) which for "over X" means P(total > X))
# Find violations
violations_ou = []
checked = 0
for ev_id, mkts in ou_events.items():
    mkts_sorted = sorted(mkts, key=lambda x: x["threshold"])
    for i in range(len(mkts_sorted) - 1):
        a = mkts_sorted[i]   # lower threshold
        b = mkts_sorted[i+1] # higher threshold
        # require both have minimal liquidity to be tradeable
        if a["liquidity"] < 100 or b["liquidity"] < 100:
            checked += 1
            continue
        checked += 1
        # Constraint: P(>a) >= P(>b), so a["yes_price"] >= b["yes_price"]
        diff = b["yes_price"] - a["yes_price"]
        if diff > 0.005:  # b higher than a → violation
            violations_ou.append({
                "event": ev_id,
                "low_thr": a["threshold"], "low_price": a["yes_price"], "low_liq": a["liquidity"],
                "high_thr": b["threshold"], "high_price": b["yes_price"], "high_liq": b["liquidity"],
                "violation_size": diff,
                "low_q": a["question"],
                "high_q": b["question"],
            })

print(f"Adjacent O/U pairs checked: {checked:,}")
print(f"Monotonicity violations (>0.5 cent): {len(violations_ou):,}")
if violations_ou:
    sizes = [v["violation_size"] for v in violations_ou]
    print(f"Violation size distribution: min={min(sizes):.3f} max={max(sizes):.3f} mean={np.mean(sizes):.3f} median={np.median(sizes):.3f}")
    print(f"\nTop 10 violations (size descending):")
    for v in sorted(violations_ou, key=lambda x: -x["violation_size"])[:10]:
        print(f"  Δ={v['violation_size']:.3f}  P({v['low_thr']})={v['low_price']:.3f}  P({v['high_thr']})={v['high_price']:.3f}  liq=${v['low_liq']:.0f}/${v['high_liq']:.0f}")
        print(f"     {v['low_q'][:90]}")

# ============================================================
# Approach 2: Date-based monotonicity ("by date Y" markets)
# ============================================================
print()
print("=" * 70)
print("APPROACH 2: Date-based monotonicity ('by date X' clusters)")
print("=" * 70)

# Find markets matching patterns like "...by July 1", "...before 2026", etc.
# Cluster by underlying event (strip date), check ordering by date.

date_pattern = re.compile(
    r"\b(by|before|in)\b.*?\b(20\d{2}|January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Q[1-4])\b",
    re.IGNORECASE,
)

# Better: look at groupItemTitle which often is just the date phrase
date_in_title = re.compile(
    r"^(?:by|before|in)\s+(.+)$|^(20\d{2})$|^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:,?\s*20\d{2})?$",
    re.IGNORECASE,
)

# Simpler approach: cluster markets by event (via events[0].id) and look for
# different endDates within the same event.
event_to_markets = defaultdict(list)
for mk in markets:
    if not mk.get("events"):
        continue
    try:
        ev_id = mk["events"][0].get("id")
    except (KeyError, IndexError, AttributeError, TypeError):
        continue
    if not ev_id:
        continue
    p = parse_prices(mk)
    if p is None:
        continue
    event_to_markets[ev_id].append({
        "id": mk.get("id"),
        "question": mk.get("question"),
        "yes_price": p,
        "endDate": mk.get("endDate"),
        "liquidity": float(mk.get("liquidityNum") or 0),
        "v24": float(mk.get("volume24hr") or 0),
        "title": mk.get("groupItemTitle", ""),
    })

# For each event, look at markets with same SHAPE of question (e.g., "Will Trump be impeached")
# but different endDate or groupItemTitle implying a date. Group naively by replacing dates
# in question text.
date_in_question = re.compile(
    r"\b(?:by|before|in)\s+("
    r"\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?"
    r"|Q[1-4]\s*20\d{2}"
    r"|20\d{2}"
    r")\b",
    re.IGNORECASE,
)

# Strip dates from question to find shape
def shape(q):
    if not q:
        return ""
    s = date_in_question.sub("<DATE>", q)
    s = re.sub(r"\b20\d{2}\b", "<YR>", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

shape_buckets = defaultdict(list)
for mk in markets:
    q = mk.get("question") or ""
    if "<DATE>" not in shape(q) and "<YR>" not in shape(q):
        # No date pattern - skip
        continue
    p = parse_prices(mk)
    if p is None:
        continue
    end = mk.get("endDate") or ""
    if not end:
        continue
    shape_buckets[shape(q)].append({
        "id": mk.get("id"),
        "question": q,
        "yes_price": p,
        "endDate": end,
        "liquidity": float(mk.get("liquidityNum") or 0),
        "v24": float(mk.get("volume24hr") or 0),
    })

shape_buckets = {k: v for k, v in shape_buckets.items() if len(v) >= 2}
print(f"Question shapes with >=2 dated variants: {len(shape_buckets):,}")

violations_date = []
checked_date = 0
for shp, mkts in shape_buckets.items():
    s = sorted(mkts, key=lambda x: x["endDate"])
    for i in range(len(s) - 1):
        a, b = s[i], s[i+1]  # a is earlier date
        if a["liquidity"] < 100 or b["liquidity"] < 100:
            checked_date += 1
            continue
        # If b is "later date", then P(by b) >= P(by a), so b_price - a_price >= 0
        # Violation: a > b (earlier deadline has higher probability)
        checked_date += 1
        diff = a["yes_price"] - b["yes_price"]
        if diff > 0.005:
            violations_date.append({
                "shape": shp,
                "early_date": a["endDate"], "early_price": a["yes_price"], "early_liq": a["liquidity"],
                "late_date": b["endDate"], "late_price": b["yes_price"], "late_liq": b["liquidity"],
                "violation_size": diff,
                "early_q": a["question"],
                "late_q": b["question"],
            })

print(f"Adjacent date pairs checked: {checked_date:,}")
print(f"Date monotonicity violations (>0.5 cent): {len(violations_date):,}")
if violations_date:
    sizes = [v["violation_size"] for v in violations_date]
    print(f"Violation size distribution: min={min(sizes):.3f} max={max(sizes):.3f} mean={np.mean(sizes):.3f} median={np.median(sizes):.3f}")
    print(f"\nTop 10 date violations:")
    for v in sorted(violations_date, key=lambda x: -x["violation_size"])[:10]:
        print(f"  Δ={v['violation_size']:.3f}  early={v['early_price']:.3f}  late={v['late_price']:.3f}  liq=${v['early_liq']:.0f}/${v['late_liq']:.0f}")
        print(f"     EARLY: {v['early_q'][:90]}")
        print(f"     LATE:  {v['late_q'][:90]}")

# Save results
Path("results_A.json").write_text(json.dumps({
    "ou_violations": violations_ou,
    "date_violations": violations_date,
}, indent=2, default=str))
print("\nSaved violations to results_A.json")
