"""
Strategy B Blindspot Tests.

Re-uses backtest data from strategy_b_v3 (results_B_v3.json) and runs
additional stress tests:

  1. Random baseline: same exit logic, random entry points
  2. Survivorship: distribution of test markets, resolved-during-test
  3. Out-of-sample: train/test temporal split
  4. Concentration: how is profit distributed across markets/trades?
  5. Boundary effects: signals near 0 / 1 are structurally different
  6. Resolution proximity: are profitable trades just resolution-convergence?
  7. Window sensitivity: 12h, 24h, 48h, 72h rolling windows
"""
import json
import random
import sys
import time
from pathlib import Path
import requests
import numpy as np

CLOB = "https://clob.polymarket.com"
markets_list = json.loads(Path("markets.json").read_text())
markets = {m["id"]: m for m in markets_list}
trades = json.loads(Path("results_B_v3.json").read_text())

def num(m, k, default=0.0):
    v = m.get(k)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def fee_per_trade(p, fee_rate=0.025):
    return 2 * fee_rate * p * (1 - p)

print(f"Loaded {len(trades):,} backtest trades from previous run")
print()

# ============================================================
# Test 1: Trade concentration
# ============================================================
print("=" * 70)
print("TEST 1: Profit concentration (is the edge in a few markets?)")
print("=" * 70)
zs = np.array([abs(t["entry_z"]) for t in trades])
rets = np.array([t["ret_per_share"] for t in trades])

# Focus on z>=4 (the supposedly profitable subset)
mask = zs >= 4.0
sel_trades = [t for t, m in zip(trades, mask) if m]
sel_rets = rets[mask]

print(f"Trades at z>=4: {len(sel_trades):,}")

# Group by market
from collections import defaultdict
per_market = defaultdict(list)
for t, r in zip(sel_trades, sel_rets):
    per_market[t["market_id"]].append(r)

market_pnl = {mid: sum(rs) for mid, rs in per_market.items()}
market_n = {mid: len(rs) for mid, rs in per_market.items()}
print(f"Markets contributing trades: {len(market_pnl):,}")

total_pnl = sum(market_pnl.values())
sorted_mids = sorted(market_pnl.items(), key=lambda x: -x[1])

# Top X% contribution
def top_pct_share(pct):
    n_top = max(1, int(len(sorted_mids) * pct / 100))
    return sum(p for _, p in sorted_mids[:n_top]) / total_pnl * 100

print(f"\nTotal PnL@z>=4: {total_pnl:.4f}")
print(f"Share of total PnL from:")
print(f"  Top 1% markets:  {top_pct_share(1):.1f}%")
print(f"  Top 5% markets:  {top_pct_share(5):.1f}%")
print(f"  Top 10% markets: {top_pct_share(10):.1f}%")
print(f"  Top 25% markets: {top_pct_share(25):.1f}%")
print(f"  Top 50% markets: {top_pct_share(50):.1f}%")

# Largest single market contribution
biggest_mid, biggest_pnl = sorted_mids[0]
biggest_n = market_n[biggest_mid]
big_mkt = markets.get(biggest_mid, {})
print(f"\nLargest contributor: market_id={biggest_mid}")
print(f"  Question: {big_mkt.get('question', '')[:80]}")
print(f"  Trades: {biggest_n}, PnL: {biggest_pnl:.4f}")

# How many markets have positive vs negative net PnL?
pos_markets = sum(1 for p in market_pnl.values() if p > 0)
print(f"\nMarkets with positive z>=4 net PnL: {pos_markets}/{len(market_pnl)} = {100*pos_markets/len(market_pnl):.1f}%")

# Without top contributor
without_top = total_pnl - biggest_pnl
print(f"PnL without largest market: {without_top:.4f} ({100*without_top/total_pnl:.1f}% of original)")

# ============================================================
# Test 2: Boundary effects
# ============================================================
print()
print("=" * 70)
print("TEST 2: Boundary effects (signals near 0 or 1)")
print("=" * 70)

entries = np.array([t["entry_price"] for t in trades])
exits = np.array([t["exit_price"] for t in trades])

# Near-boundary trades (entry within 0.1 of 0 or 1)
near_low = (entries < 0.10)
near_high = (entries > 0.90)
middle = ~(near_low | near_high)

print(f"All z>=4 trades, partitioned:")
for label, partition_mask in [("entry < 0.10", near_low), ("entry > 0.90", near_high), ("entry in [0.10, 0.90]", middle)]:
    m = partition_mask & (zs >= 4.0)
    if m.sum() < 5:
        continue
    sel = rets[m]
    fees = np.array([fee_per_trade(p) for p in 0.5*(entries[m]+exits[m])])
    net = sel - 0.02 - fees
    win = (sel > 0).mean()
    print(f"  {label:25s}: n={m.sum():4d}  gross_avg={sel.mean()*100:+.2f}¢  net@2¢={net.mean()*100:+.2f}¢  win%={win*100:.1f}")

# ============================================================
# Test 3: Resolution proximity
# ============================================================
print()
print("=" * 70)
print("TEST 3: Resolution proximity (is edge just resolution convergence?)")
print("=" * 70)

import datetime
now = time.time()

# For each trade, compute days-to-resolution
def days_to_end(t):
    mid = t["market_id"]
    m = markets.get(mid)
    if not m:
        return None
    end = m.get("endDate")
    if not end:
        return None
    try:
        end_ts = datetime.datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None
    return (end_ts - t["entry_ts"]) / 86400

z4_trades = [t for t in trades if abs(t["entry_z"]) >= 4.0]
days = [days_to_end(t) for t in z4_trades]
valid = [(t, d) for t, d in zip(z4_trades, days) if d is not None]
print(f"Z>=4 trades with valid end-date: {len(valid):,}")

for lo, hi, label in [(0, 1, "<1 day"), (1, 7, "1-7 days"), (7, 30, "7-30 days"), (30, 365, "30-365 days"), (365, 1e6, ">365 days")]:
    bucket = [(t, d) for t, d in valid if lo <= d < hi]
    if len(bucket) < 5:
        continue
    rets_b = np.array([b[0]["ret_per_share"] for b in bucket])
    ents = np.array([b[0]["entry_price"] for b in bucket])
    exs = np.array([b[0]["exit_price"] for b in bucket])
    fees = np.array([fee_per_trade(p) for p in 0.5*(ents+exs)])
    net = rets_b - 0.02 - fees
    print(f"  Days-to-end {label:>13}: n={len(bucket):4d}  gross={rets_b.mean()*100:+.2f}¢  net@2¢={net.mean()*100:+.2f}¢  win%={(rets_b>0).mean()*100:.1f}")

# ============================================================
# Test 4: Out-of-sample temporal split
# ============================================================
print()
print("=" * 70)
print("TEST 4: Temporal out-of-sample (train/test split)")
print("=" * 70)

# Sort by entry_ts and split at median
sorted_trades = sorted(trades, key=lambda t: t["entry_ts"])
half = len(sorted_trades) // 2
first_half = sorted_trades[:half]
second_half = sorted_trades[half:]

t_split = sorted_trades[half]["entry_ts"]
t_split_str = datetime.datetime.fromtimestamp(t_split).strftime("%Y-%m-%d %H:%M")
print(f"Split point: {t_split_str}")
print(f"First half: {len(first_half):,} trades, Second half: {len(second_half):,} trades")
print()

for label, batch in [("First half", first_half), ("Second half", second_half)]:
    print(f"  {label}:")
    bz = np.array([abs(t["entry_z"]) for t in batch])
    br = np.array([t["ret_per_share"] for t in batch])
    be = np.array([t["entry_price"] for t in batch])
    bx = np.array([t["exit_price"] for t in batch])
    for z_min in [2.0, 3.0, 4.0, 5.0]:
        m = bz >= z_min
        if m.sum() < 5:
            continue
        sel = br[m]
        fees = np.array([fee_per_trade(p) for p in 0.5*(be[m]+bx[m])])
        net = sel - 0.02 - fees
        print(f"    z>={z_min}: n={m.sum():4d}  gross={sel.mean()*100:+.2f}¢  net@2¢={net.mean()*100:+.2f}¢  win%={(sel>0).mean()*100:.1f}")

# ============================================================
# Test 5: Hold-time and forced-exit breakdown
# ============================================================
print()
print("=" * 70)
print("TEST 5: Hold-time and forced-exit breakdown")
print("=" * 70)

z4 = [t for t in trades if abs(t["entry_z"]) >= 4.0]
forced = [t for t in z4 if t["forced_exit"]]
natural = [t for t in z4 if not t["forced_exit"]]

print(f"Z>=4 trades: {len(z4)} (forced exits: {len(forced)}, natural: {len(natural)})")
if forced:
    rs = np.array([t["ret_per_share"] for t in forced])
    print(f"  Forced exits: avg ret={rs.mean()*100:+.2f}¢  win%={(rs>0).mean()*100:.1f}")
if natural:
    rs = np.array([t["ret_per_share"] for t in natural])
    print(f"  Natural exits: avg ret={rs.mean()*100:+.2f}¢  win%={(rs>0).mean()*100:.1f}")

# ============================================================
# Test 6: Distribution properties (Sharpe, tail risk)
# ============================================================
print()
print("=" * 70)
print("TEST 6: Risk-adjusted metrics + tail analysis")
print("=" * 70)

z4_rets = np.array([t["ret_per_share"] for t in z4])
z4_ents = np.array([t["entry_price"] for t in z4])
z4_exs = np.array([t["exit_price"] for t in z4])
z4_fees = np.array([fee_per_trade(p) for p in 0.5*(z4_ents+z4_exs)])
z4_net = z4_rets - 0.02 - z4_fees

print(f"Z>=4 net returns (after 2¢ spread + fees), n={len(z4_net)}:")
print(f"  Mean:    {z4_net.mean()*100:+.3f}¢")
print(f"  Std:     {z4_net.std()*100:.3f}¢")
print(f"  Sharpe:  {z4_net.mean()/z4_net.std():.3f}")
print(f"  Skew:    {((z4_net - z4_net.mean())**3).mean() / z4_net.std()**3:.3f}")
print(f"  Kurt:    {((z4_net - z4_net.mean())**4).mean() / z4_net.std()**4 - 3:.3f}")
print(f"  Percentiles:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"    p{p:>2}: {np.percentile(z4_net, p)*100:+.2f}¢")

# Worst trades
worst = sorted(z4_net)[:10]
print(f"  10 worst trades: {[f'{r*100:.1f}¢' for r in worst]}")
best = sorted(z4_net)[-10:]
print(f"  10 best trades:  {[f'{r*100:.1f}¢' for r in best]}")

# ============================================================
# Test 7: Random baseline (same markets, random entries)
# ============================================================
print()
print("=" * 70)
print("TEST 7: Random-entry baseline (control)")
print("=" * 70)

# Re-run on the SAME markets but with RANDOM entries (not z-signal-driven)
# We need to re-pull price history; use a subset to save time
print("Re-running with random entries on 100 markets from existing sample...")

def get_yes_token(m):
    tids = m.get("clobTokenIds")
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

def fetch_history(token_id, retries=2):
    for i in range(retries + 1):
        try:
            r = requests.get(
                f"{CLOB}/prices-history",
                params={"market": token_id, "fidelity": 60, "interval": "max"},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("history", [])
        except requests.RequestException:
            pass
        if i < retries:
            time.sleep(0.4)
    return None

z4_market_ids = list({t["market_id"] for t in z4})
random.seed(99)
sample_mids = random.sample(z4_market_ids, min(100, len(z4_market_ids)))

# For each market, do same number of random entries as we did real entries
n_real_per_mkt = {mid: sum(1 for t in z4 if t["market_id"] == mid) for mid in sample_mids}

WINDOW = 24
MAX_HOLD = 48
EXIT_Z = 0.5
random_trades = []

for i, mid in enumerate(sample_mids):
    if i % 20 == 0:
        sys.stdout.write(f"  {i}/{len(sample_mids)}...\n")
        sys.stdout.flush()
    m = markets.get(mid)
    if not m:
        continue
    tok = get_yes_token(m)
    if not tok:
        continue
    hist = fetch_history(tok)
    time.sleep(0.08)
    if not hist or len(hist) < 50:
        continue
    prices = np.array([h["p"] for h in hist], dtype=float)
    if prices.std() < 0.001:
        continue
    n_to_sim = n_real_per_mkt[mid]
    valid_starts = list(range(WINDOW, len(prices) - MAX_HOLD - 1))
    if not valid_starts:
        continue
    rs = random.Random(mid)
    for _ in range(n_to_sim):
        t = rs.choice(valid_starts)
        window = prices[t-WINDOW:t]
        mu, sd = window.mean(), window.std()
        if sd < 0.005:
            continue
        # Random direction
        direction = rs.choice([-1, 1])
        entry = prices[t]
        # Same exit logic
        for j in range(t+1, min(t+MAX_HOLD+1, len(prices))):
            w2 = prices[j-WINDOW:j]
            mu2, sd2 = w2.mean(), w2.std()
            if sd2 < 0.005:
                continue
            z = (prices[j] - mu2) / sd2
            if abs(z) < EXIT_Z or j == t+MAX_HOLD or j == len(prices)-1:
                ret = direction * (prices[j] - entry)
                random_trades.append({
                    "ret": ret, "entry": entry, "exit": prices[j],
                })
                break

print(f"Random-entry trades simulated: {len(random_trades)}")
if random_trades:
    rr = np.array([t["ret"] for t in random_trades])
    re_ = np.array([t["entry"] for t in random_trades])
    rx_ = np.array([t["exit"] for t in random_trades])
    rf = np.array([fee_per_trade(p) for p in 0.5*(re_+rx_)])
    rn = rr - 0.02 - rf
    print(f"  Gross avg: {rr.mean()*100:+.2f}¢  win%={(rr>0).mean()*100:.1f}")
    print(f"  Net@2¢ avg: {rn.mean()*100:+.2f}¢  win%={(rn>0).mean()*100:.1f}")

    # Compare to z>=4 from same markets
    same_market_z4 = [t for t in z4 if t["market_id"] in set(sample_mids)]
    if same_market_z4:
        z4r = np.array([t["ret_per_share"] for t in same_market_z4])
        z4e = np.array([t["entry_price"] for t in same_market_z4])
        z4x = np.array([t["exit_price"] for t in same_market_z4])
        z4f = np.array([fee_per_trade(p) for p in 0.5*(z4e+z4x)])
        z4n = z4r - 0.02 - z4f
        print(f"\n  Compared to z>=4 on SAME 100 markets (n={len(same_market_z4)}):")
        print(f"    Gross avg: {z4r.mean()*100:+.2f}¢  win%={(z4r>0).mean()*100:.1f}")
        print(f"    Net@2¢ avg: {z4n.mean()*100:+.2f}¢  win%={(z4n>0).mean()*100:.1f}")
        # T-statistic for difference
        diff = z4r.mean() - rr.mean()
        pooled_se = np.sqrt(z4r.var()/len(z4r) + rr.var()/len(rr))
        t_stat = diff / pooled_se if pooled_se > 0 else 0
        print(f"\n  Signal vs random: gross diff = {diff*100:+.2f}¢  t-stat={t_stat:.2f}")

# ============================================================
# Test 8: Sample 5 actual z>=5 trades to visualize
# ============================================================
print()
print("=" * 70)
print("TEST 8: Spot-check top z signals (visual sanity)")
print("=" * 70)

z5_sorted = sorted([t for t in trades if abs(t["entry_z"]) >= 5.0],
                   key=lambda t: -abs(t["entry_z"]))[:10]
for t in z5_sorted:
    mid = t["market_id"]
    m = markets.get(mid, {})
    print(f"  z={t['entry_z']:+.1f}  entry=${t['entry_price']:.3f}  exit=${t['exit_price']:.3f}  ret={t['ret_per_share']*100:+.2f}¢  hold={t['hold_hours']}h")
    print(f"    {m.get('question', '')[:90]}")

Path("blindspot_results.json").write_text(json.dumps({
    "concentration_top1pct": top_pct_share(1),
    "concentration_top10pct": top_pct_share(10),
    "biggest_market_pnl": biggest_pnl,
    "total_z4_pnl": total_pnl,
    "n_random_trades": len(random_trades) if random_trades else 0,
}, indent=2, default=str))
print("\nDone. Saved summary to blindspot_results.json")
