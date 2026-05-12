"""
strategy_improvements.py — Test specific improvements on the existing 30-day data.

Tests we can run with current data (results_B_v3.json):
  1. Z-magnitude tiered position sizing
  2. Direction asymmetry (long YES vs short YES which is better?)
  3. Time-of-day / day-of-week patterns
  4. Hold-time analysis (early-exit benefit?)
  5. Market-category (feeType) performance breakdown
  6. Entry-price quintile analysis
  7. Combination filter optimization (which filters add most edge?)

Tests requiring NEW data fetches (deferred):
  - Stop-loss (needs intraday prices)
  - Different rolling windows (needs raw price history reanalysis)
  - Volume-weighted filters (needs orderbook snapshots)
"""
import json
import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

SPREAD_COST = 0.02
FEE_RATE = 0.025

trades_all = json.loads(Path("results_B_v3.json").read_text())
markets = {m["id"]: m for m in json.loads(Path("markets.json").read_text())}

def days_to_end(t):
    m = markets.get(t["market_id"])
    if not m or not m.get("endDate"):
        return None
    try:
        end_ts = datetime.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None
    return (end_ts - t["entry_ts"]) / 86400

for t in trades_all:
    t["dte"] = days_to_end(t)
    m = markets.get(t["market_id"])
    t["fee_type"] = m.get("feeType") if m else None
    t["question"] = m.get("question", "")[:60] if m else ""

# Base filter (the one we converged on)
def baseline_keep(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    if t["dte"] is None or t["dte"] >= 365: return False
    if 7 <= t["dte"] < 30: return False
    return True

def net_pnl_per_100(t):
    if t["direction"] == 1:
        shares = 100.0 / t["entry_price"]
    else:
        shares = 100.0 / (1 - t["entry_price"])
    gross = shares * t["ret_per_share"]
    f_entry = FEE_RATE * t["entry_price"] * (1 - t["entry_price"])
    f_exit = FEE_RATE * t["exit_price"] * (1 - t["exit_price"])
    return gross - shares * (f_entry + f_exit) - shares * SPREAD_COST

filtered = [t for t in trades_all if baseline_keep(t)]
for t in filtered:
    t["pnl_100"] = net_pnl_per_100(t)

print(f"Baseline filtered trades: {len(filtered)}")
print(f"Baseline mean PnL/$100: ${np.mean([t['pnl_100'] for t in filtered]):+.2f}")
print(f"Baseline total PnL on $100 each: ${sum(t['pnl_100'] for t in filtered):+.2f}\n")

# ──────────────────────────────────────────────────────────────────
# Test 1: Z-magnitude tiered position sizing
# ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1: Position size scaled by z-magnitude")
print("=" * 70)
print("Hypothesis: bigger z = stronger mean-revert signal → size up\n")

# Compute fixed-size baseline vs z-scaled sizing
for scale_fn_name, scale_fn in [
    ("Constant $100 (baseline)",        lambda z: 100),
    ("Linear: $50 + $20·z",             lambda z: 50 + 20 * abs(z)),
    ("Linear capped at $300",           lambda z: min(50 + 30 * abs(z), 300)),
    ("Quadratic capped at $300",        lambda z: min(50 + abs(z)**2 * 4, 300)),
    ("Step: z<7=$50, 7-10=$100, 10+=$200", lambda z: 50 if abs(z) < 7 else (100 if abs(z) < 10 else 200)),
]:
    pnls = []
    for t in filtered:
        z = abs(t["entry_z"])
        stake = scale_fn(z)
        if t["direction"] == 1:
            shares = stake / t["entry_price"]
        else:
            shares = stake / (1 - t["entry_price"])
        gross = shares * t["ret_per_share"]
        f_entry = FEE_RATE * t["entry_price"] * (1 - t["entry_price"])
        f_exit = FEE_RATE * t["exit_price"] * (1 - t["exit_price"])
        net = gross - shares * (f_entry + f_exit) - shares * SPREAD_COST
        pnls.append(net)
    pnls = np.array(pnls)
    total_capital = sum(scale_fn(abs(t["entry_z"])) for t in filtered)
    print(f"  {scale_fn_name:<42} total=${pnls.sum():>+9.2f}  mean=${pnls.mean():>+6.2f}  capital=${total_capital:>9.0f}  ROI={100*pnls.sum()/total_capital:>+5.2f}%")

# ──────────────────────────────────────────────────────────────────
# Test 2: Direction asymmetry
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 2: Direction asymmetry")
print("=" * 70)
print("Hypothesis: short YES (z>0, fade up-spike) may differ from long YES\n")

for label, dir_filter in [
    ("All directions", lambda t: True),
    ("Long YES only (z<0, fade down-spike)", lambda t: t["direction"] == 1),
    ("Short YES only (z>0, fade up-spike)", lambda t: t["direction"] == -1),
]:
    sub = [t for t in filtered if dir_filter(t)]
    pnls = np.array([t["pnl_100"] for t in sub])
    if len(sub) == 0:
        continue
    print(f"  {label:<45} n={len(sub):>3}  mean=${pnls.mean():>+6.2f}  win%={(pnls>0).mean()*100:>5.1f}  total=${pnls.sum():>+9.2f}")

# ──────────────────────────────────────────────────────────────────
# Test 3: Time-of-day / day-of-week
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 3: Time-of-day (UTC hour) at entry")
print("=" * 70)
print("Hypothesis: bots are less active during US-night → edge bigger\n")

by_hour = defaultdict(list)
for t in filtered:
    h = datetime.datetime.fromtimestamp(t["entry_ts"], datetime.timezone.utc).hour
    by_hour[h].append(t["pnl_100"])

print(f"  {'UTC hour':>10} {'n':>5} {'mean':>10} {'win%':>7}")
for h in sorted(by_hour):
    pnls = np.array(by_hour[h])
    if len(pnls) >= 5:
        # US east coast hours (UTC-5)
        et = (h - 5) % 24
        print(f"  {h:>2}h UTC ({et:>2}h ET) {len(pnls):>5}  ${pnls.mean():>+7.2f}  {(pnls>0).mean()*100:>5.1f}%")

print()
print("  Aggregate windows:")
us_day = [t["pnl_100"] for t in filtered if 14 <= datetime.datetime.fromtimestamp(t["entry_ts"], datetime.timezone.utc).hour < 22]
us_night = [t["pnl_100"] for t in filtered if not (14 <= datetime.datetime.fromtimestamp(t["entry_ts"], datetime.timezone.utc).hour < 22)]
eu_business = [t["pnl_100"] for t in filtered if 7 <= datetime.datetime.fromtimestamp(t["entry_ts"], datetime.timezone.utc).hour < 16]
print(f"  US daytime (14-22 UTC):  n={len(us_day):>3}  mean=${np.mean(us_day):>+7.2f}  win%={100*np.mean(np.array(us_day)>0):>5.1f}")
print(f"  US off-hours:            n={len(us_night):>3}  mean=${np.mean(us_night):>+7.2f}  win%={100*np.mean(np.array(us_night)>0):>5.1f}")
print(f"  EU business (7-16 UTC):  n={len(eu_business):>3}  mean=${np.mean(eu_business):>+7.2f}  win%={100*np.mean(np.array(eu_business)>0):>5.1f}")

# Day of week
print(f"\n  Day-of-week:")
by_dow = defaultdict(list)
for t in filtered:
    d = datetime.datetime.fromtimestamp(t["entry_ts"], datetime.timezone.utc).strftime("%a")
    by_dow[d].append(t["pnl_100"])
for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
    if d in by_dow:
        pnls = np.array(by_dow[d])
        print(f"  {d}: n={len(pnls):>3}  mean=${pnls.mean():>+7.2f}  win%={(pnls>0).mean()*100:>5.1f}")

# ──────────────────────────────────────────────────────────────────
# Test 4: Hold-time analysis
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 4: Hold-time bucket performance")
print("=" * 70)
print("Hypothesis: short reversions are 'cleaner'; long ones might be momentum\n")

for lo, hi, label in [(0, 4, "0-4h"), (4, 12, "4-12h"), (12, 24, "12-24h"),
                       (24, 48, "24-48h (force exit window)")]:
    sub = [t for t in filtered if lo <= t["hold_hours"] < hi]
    if not sub: continue
    pnls = np.array([t["pnl_100"] for t in sub])
    forced = sum(1 for t in sub if t["forced_exit"])
    print(f"  {label:<30} n={len(sub):>3}  mean=${pnls.mean():>+7.2f}  win%={(pnls>0).mean()*100:>5.1f}  forced_exits={forced}")

# ──────────────────────────────────────────────────────────────────
# Test 5: feeType / market category
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 5: Performance by market category (feeType)")
print("=" * 70)

by_ft = defaultdict(list)
for t in filtered:
    by_ft[t.get("fee_type") or "unknown"].append(t["pnl_100"])

print(f"  {'feeType':<25} {'n':>5} {'mean$':>10} {'win%':>7} {'total$':>10}")
for ft in sorted(by_ft.keys(), key=lambda k: -len(by_ft[k])):
    pnls = np.array(by_ft[ft])
    if len(pnls) >= 3:
        print(f"  {ft:<25} {len(pnls):>5}  ${pnls.mean():>+7.2f}  {(pnls>0).mean()*100:>5.1f}%  ${pnls.sum():>+8.2f}")

# ──────────────────────────────────────────────────────────────────
# Test 6: Entry price quintile
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 6: Entry price band (within [0.10, 0.90])")
print("=" * 70)

for lo, hi in [(0.10, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 0.75), (0.75, 0.90)]:
    sub = [t for t in filtered if lo <= t["entry_price"] < hi]
    if not sub: continue
    pnls = np.array([t["pnl_100"] for t in sub])
    print(f"  Price [{lo:.2f}, {hi:.2f}):  n={len(sub):>3}  mean=${pnls.mean():>+7.2f}  win%={(pnls>0).mean()*100:>5.1f}%  total=${pnls.sum():>+8.2f}")

# ──────────────────────────────────────────────────────────────────
# Test 7: Optimal filter combination (greedy search)
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 7: Try adding filters - which give best return-per-trade?")
print("=" * 70)

# Test adding additional filters one at a time
def evaluate(trades_subset):
    if not trades_subset: return None
    pnls = np.array([t["pnl_100"] for t in trades_subset])
    return {
        "n": len(trades_subset),
        "mean": pnls.mean(),
        "total": pnls.sum(),
        "win%": (pnls > 0).mean() * 100,
        "sharpe": pnls.mean() / pnls.std() if pnls.std() > 0 else 0,
    }

candidate_filters = [
    ("z >= 7", lambda t: abs(t["entry_z"]) >= 7),
    ("z >= 10", lambda t: abs(t["entry_z"]) >= 10),
    ("z in [5, 10]", lambda t: 5 <= abs(t["entry_z"]) < 10),
    ("short YES only (z>0)", lambda t: t["direction"] == -1),
    ("price [0.30, 0.70]", lambda t: 0.30 <= t["entry_price"] <= 0.70),
    ("price [0.20, 0.80]", lambda t: 0.20 <= t["entry_price"] <= 0.80),
    ("dte 30-180 days", lambda t: 30 <= t["dte"] < 180),
    ("dte 60-365", lambda t: 60 <= t["dte"] < 365),
    ("hold < 24h", lambda t: t["hold_hours"] < 24),
    ("natural exits only", lambda t: not t["forced_exit"]),
    ("sports markets only", lambda t: t.get("fee_type") == "sports_fees_v2"),
    ("NOT sports", lambda t: t.get("fee_type") != "sports_fees_v2"),
]

print(f"  {'Filter':<32} {'n':>5} {'mean$':>10} {'win%':>7} {'sharpe':>8} {'total$':>10}")
baseline = evaluate(filtered)
print(f"  {'(baseline)':<32} {baseline['n']:>5}  ${baseline['mean']:>+7.2f}  {baseline['win%']:>5.1f}%  {baseline['sharpe']:>+6.3f}  ${baseline['total']:>+8.2f}")
print()

results = []
for name, fn in candidate_filters:
    sub = [t for t in filtered if fn(t)]
    r = evaluate(sub)
    if r is None: continue
    results.append((name, r))
    # Improvement vs baseline
    improvement = r["mean"] - baseline["mean"]
    flag = "↑" if improvement > 0 else "↓"
    print(f"  {flag} {name:<30} {r['n']:>5}  ${r['mean']:>+7.2f}  {r['win%']:>5.1f}%  {r['sharpe']:>+6.3f}  ${r['total']:>+8.2f}")

# ──────────────────────────────────────────────────────────────────
# Test 8: Combined best filter
# ──────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 8: Combinations of best individual filters")
print("=" * 70)

combos = [
    ("baseline", lambda t: True),
    ("z>=7", lambda t: abs(t["entry_z"]) >= 7),
    ("z>=7 AND short YES", lambda t: abs(t["entry_z"]) >= 7 and t["direction"] == -1),
    ("z>=7 AND price [0.30, 0.70]", lambda t: abs(t["entry_z"]) >= 7 and 0.30 <= t["entry_price"] <= 0.70),
    ("z>=7 AND hold<24h", lambda t: abs(t["entry_z"]) >= 7 and t["hold_hours"] < 24),
    ("z>=7 AND natural exit AND price [0.20, 0.80]", lambda t: abs(t["entry_z"]) >= 7 and not t["forced_exit"] and 0.20 <= t["entry_price"] <= 0.80),
]

print(f"  {'Combo':<55} {'n':>5} {'mean$':>10} {'win%':>7} {'sharpe':>8}")
for name, fn in combos:
    sub = [t for t in filtered if fn(t)]
    r = evaluate(sub)
    if r is None: continue
    print(f"  {name:<55} {r['n']:>5}  ${r['mean']:>+7.2f}  {r['win%']:>5.1f}%  {r['sharpe']:>+6.3f}")

print(f"\n=== END ===")
