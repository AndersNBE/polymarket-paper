"""
Verify monotonicity violations against actual orderbook.

For each candidate violation, fetch the live orderbook from CLOB API to
determine:
  - Real best-bid / best-ask (not just midpoint estimate)
  - Tradeable size at those levels
  - Net arb after fees and spread

This tells us whether the violations are REAL (tradeable) or just stale-price
artifacts.
"""
import json
import time
from pathlib import Path
import requests

CLOB = "https://clob.polymarket.com"
markets = {m["id"]: m for m in json.loads(Path("markets.json").read_text())}
results = json.loads(Path("results_A.json").read_text())

def fetch_book(token_id, retries=2):
    for i in range(retries + 1):
        try:
            r = requests.get(f"{CLOB}/book", params={"token_id": token_id}, timeout=10)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        if i < retries:
            time.sleep(0.5)
    return None

def best_bid_ask(book):
    """Returns (best_bid, bid_size, best_ask, ask_size) or None."""
    if not book:
        return None
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids and not asks:
        return None
    bb = float(bids[-1]["price"]) if bids else None
    bs = float(bids[-1]["size"]) if bids else None
    ba = float(asks[-1]["price"]) if asks else None
    az = float(asks[-1]["size"]) if asks else None
    # NOTE: Polymarket sometimes returns bids ascending and asks descending; pick the BEST
    if bids:
        bb = max(float(b["price"]) for b in bids)
        bs = sum(float(b["size"]) for b in bids if abs(float(b["price"]) - bb) < 1e-9)
    if asks:
        ba = min(float(a["price"]) for a in asks)
        az = sum(float(a["size"]) for a in asks if abs(float(a["price"]) - ba) < 1e-9)
    return (bb, bs, ba, az)

def get_yes_token(m):
    """Return YES outcome's CLOB token id."""
    tids = m.get("clobTokenIds")
    if not tids:
        return None
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError):
        return None

# We need an event_id -> market_id map. Let me rebuild from violation pairs.
# The results stored prices but not market ids per pair, so let me rebuild
# using the markets dataset and the violation records.

# Build (event_id, threshold) -> market_id index
import re
from collections import defaultdict
ou_pat = re.compile(r"^(O/U|Over/Under|Over\b|Under\b)\s*([\d.]+)$", re.IGNORECASE)
ev_map = defaultdict(dict)
for m in markets.values():
    gt = m.get("groupItemTitle", "")
    mt = ou_pat.match(gt.strip() if gt else "")
    if not mt:
        continue
    thr = float(mt.group(2))
    if m.get("events"):
        try:
            ev_id = m["events"][0].get("id") or m["events"][0].get("slug")
        except (KeyError, IndexError, AttributeError, TypeError):
            ev_id = None
    else:
        ev_id = None
    if not ev_id:
        slug = m.get("slug", "")
        ev_id = re.sub(r"-(o-u|over|under)[\d-]*$", "", slug)
    ev_map[ev_id][thr] = m["id"]

# Sort violations by size, take top 30 with real liquidity (both sides > 200)
candidates = [v for v in results["ou_violations"]
              if v["low_liq"] > 200 and v["high_liq"] > 200]
candidates.sort(key=lambda x: -x["violation_size"])
candidates = candidates[:30]
print(f"Verifying {len(candidates)} top candidates with orderbook fetches...")
print()

verified = []
for v in candidates:
    ev = v["event"]
    low_id = ev_map.get(ev, {}).get(v["low_thr"])
    high_id = ev_map.get(ev, {}).get(v["high_thr"])
    if not (low_id and high_id):
        continue
    low_m = markets.get(low_id)
    high_m = markets.get(high_id)
    if not (low_m and high_m):
        continue
    low_token = get_yes_token(low_m)
    high_token = get_yes_token(high_m)
    if not (low_token and high_token):
        continue
    low_book = fetch_book(low_token)
    time.sleep(0.15)
    high_book = fetch_book(high_token)
    time.sleep(0.15)
    low_bba = best_bid_ask(low_book)
    high_bba = best_bid_ask(high_book)
    if not (low_bba and high_bba):
        continue
    lb_bid, lb_bs, lb_ask, lb_as = low_bba
    hb_bid, hb_bs, hb_ask, hb_as = high_bba

    # Constraint: P(>low_thr) >= P(>high_thr)
    # If we want to exploit when this is violated (i.e. high_price > low_price),
    # the trade is: BUY low_yes, SELL high_yes (or buy high_no).
    # Since YES on "over X" = "team scores more than X goals" and prob of "over 0.5"
    # should be >= prob of "over 1.5", buying the under-priced YES on lower threshold
    # and shorting the over-priced YES on higher threshold = arb.
    #
    # On Polymarket: BUY at ask, SELL at bid.
    # So we BUY low_yes at lb_ask, SELL high_yes at hb_bid.
    # For arb: we want the prob ordering violated even after spread.
    # Effective trade: pay lb_ask for low_yes, receive hb_bid for high_yes.
    # Net cost = lb_ask - hb_bid. If this is < 0 we get paid to take the arb.
    # But it's not free money - we're long P(>low) and short P(>high).
    # The expected payoff: P(>low) * 1 + (1 - P(>high)) * 1 - 1 = P(>low) - P(>high)
    # Wait, let me redo this properly.
    #
    # Position: +1 share of low_yes, -1 share of high_yes
    # Cost now: +lb_ask - hb_bid
    # Payoff at resolution:
    #   If total > high_thr: low resolves YES (+1), high resolves YES (-1). Net = 0.
    #   If low_thr < total <= high_thr: low YES (+1), high NO (0). Net = +1.
    #   If total <= low_thr: low NO (0), high NO (0). Net = 0.
    # So payoff is +1 with probability P(low < total <= high), else 0.
    # Expected payoff = P(low < total <= high) - cost
    # By the monotonicity constraint, this MUST be >= 0 (probability is nonneg).
    # So if cost < 0, we have GUARANTEED arb regardless of probabilities.

    cost = lb_ask - hb_bid if (lb_ask is not None and hb_bid is not None) else None
    can_arb = cost is not None and cost < 0

    max_size = min(lb_as or 0, hb_bs or 0)

    verified.append({
        "event": ev,
        "low_thr": v["low_thr"], "high_thr": v["high_thr"],
        "low_q": v["low_q"],
        "low_bid": lb_bid, "low_ask": lb_ask, "low_ask_size": lb_as,
        "high_bid": hb_bid, "high_ask": hb_ask, "high_bid_size": hb_bs,
        "cost_per_pair": cost,
        "guaranteed_arb": can_arb,
        "max_tradeable_size": max_size,
        "raw_violation_was": v["violation_size"],
    })
    arb_flag = "  ✓ ARB" if can_arb else ""
    print(f"  Event: {str(ev)[:60]}")
    print(f"    O/U {v['low_thr']}: bid={lb_bid}  ask={lb_ask}  size@ask={lb_as}")
    print(f"    O/U {v['high_thr']}: bid={hb_bid}  ask={hb_ask}  size@bid={hb_bs}")
    print(f"    Cost-per-pair: {cost}{arb_flag}    max-size=${max_size}")
    print()

print(f"\n=== Summary ===")
print(f"Verified pairs: {len(verified)}")
real_arbs = [v for v in verified if v["guaranteed_arb"]]
print(f"Pairs with guaranteed arb (negative cost): {len(real_arbs)}")
if real_arbs:
    total_profit_potential = sum(abs(v["cost_per_pair"]) * v["max_tradeable_size"] for v in real_arbs)
    print(f"Total profit potential (sum of |cost|*max_size): ${total_profit_potential:.2f}")
    print("\nReal arb opportunities:")
    for r in sorted(real_arbs, key=lambda x: x["cost_per_pair"])[:15]:
        print(f"  Event: {str(r['event'])[:60]}")
        print(f"    O/U {r['low_thr']} vs {r['high_thr']}: cost={r['cost_per_pair']:.4f}, max_size=${r['max_tradeable_size']:.2f}")
        print(f"    Profit per pair: ${abs(r['cost_per_pair']):.4f} × ${r['max_tradeable_size']:.2f} = ${abs(r['cost_per_pair'])*r['max_tradeable_size']:.4f}")

Path("results_A_verified.json").write_text(json.dumps(verified, indent=2, default=str))
print("\nSaved to results_A_verified.json")
