"""
Strategy B v2: Refined analysis.

1. Measure REAL spreads on long-tail markets (sample)
2. Parameter sweep over entry threshold z
3. Filter by market liquidity tiers
4. Compute net PnL with realistic per-market spreads
"""
import json
import time
import random
from pathlib import Path
import requests
import numpy as np

CLOB = "https://clob.polymarket.com"
trades = json.loads(Path("results_B.json").read_text())
markets = {m["id"]: m for m in json.loads(Path("markets.json").read_text())}

# Step 1: Measure spreads on the markets actually tested
print("Measuring live spreads on markets used in backtest...")
unique_mids = list({t["market_id"] for t in trades})
random.seed(0)
spread_sample = random.sample(unique_mids, min(50, len(unique_mids)))

def get_yes_token(m):
    tids = m.get("clobTokenIds")
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

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

spreads = []
for mid in spread_sample:
    m = markets.get(mid)
    if not m:
        continue
    tok = get_yes_token(m)
    if not tok:
        continue
    book = fetch_book(tok)
    time.sleep(0.12)
    if not book:
        continue
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        continue
    bb = max(float(b["price"]) for b in bids)
    ba = min(float(a["price"]) for a in asks)
    sp = ba - bb
    spreads.append({"mid": mid, "bid": bb, "ask": ba, "spread": sp,
                    "liq": float(m.get("liquidityNum") or 0),
                    "v24": float(m.get("volume24hr") or 0)})

print(f"\nMeasured spreads on {len(spreads)} long-tail markets:")
sps = sorted(s["spread"] for s in spreads)
print(f"  p10: {sps[len(sps)//10]:.3f}")
print(f"  p25: {sps[len(sps)//4]:.3f}")
print(f"  p50: {sps[len(sps)//2]:.3f}")
print(f"  p75: {sps[3*len(sps)//4]:.3f}")
print(f"  p90: {sps[9*len(sps)//10]:.3f}")
print(f"  mean: {np.mean(sps):.3f}")

# Correlate spread with liquidity bucket
print(f"\nSpread by liquidity tier:")
tiers = [(500, 2000), (2000, 5000), (5000, 10000), (10000, 25000), (25000, 1e9)]
for lo, hi in tiers:
    bucket = [s["spread"] for s in spreads if lo <= s["liq"] < hi]
    if bucket:
        print(f"  ${lo:>6}-${hi:>6}: n={len(bucket):3d}  median_spread={np.median(bucket):.3f}  mean={np.mean(bucket):.3f}")

Path("spread_sample.json").write_text(json.dumps(spreads, indent=2))

# Step 2: Sensitivity sweep over entry z-threshold and per-spread bucket
print("\n" + "="*60)
print("Sensitivity sweep: per-market spread → PnL")
print("="*60)

# We have rets per trade. Let's compute net PnL under different spread assumptions
rets = np.array([t["ret_per_share"] for t in trades])
entries = np.array([t["entry_price"] for t in trades])
exits = np.array([t["exit_price"] for t in trades])
entry_zs = np.array([abs(t["entry_z"]) for t in trades])

# Build fee per trade (approximate; Polymarket category-dependent, use 3% base)
def fee_per_trade(p, fee_rate=0.03):
    """fee = C * fee_rate * p * (1-p), per side. Round trip = 2x."""
    return 2 * fee_rate * p * (1 - p)

# Sensitivity over (spread, z_min)
print(f"\nNet avg return per trade under various assumptions (positive = profitable):")
print(f"{'spread':>8} {'z_min':>6} {'n_trades':>9} {'gross_avg':>10} {'net_avg':>10} {'net_total':>11} {'win%':>6}")
for spread in [0.005, 0.01, 0.02, 0.03, 0.05]:
    for z_min in [2.0, 2.5, 3.0, 4.0, 5.0]:
        mask = entry_zs >= z_min
        if mask.sum() < 5:
            continue
        sel_rets = rets[mask]
        sel_entries = entries[mask]
        sel_exits = exits[mask]
        fees = np.array([fee_per_trade(p) for p in 0.5*(sel_entries+sel_exits)])
        net = sel_rets - spread - fees
        wins = (net > 0).sum()
        print(f"  {spread:>5.3f}  {z_min:>4.1f}  {mask.sum():>8}  {sel_rets.mean():>+9.4f}  {net.mean():>+9.4f}  {net.sum():>+10.3f}  {100*wins/mask.sum():>5.1f}%")
