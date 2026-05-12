"""
monte_carlo_v2.py — Monte Carlo on IMPROVED strategy + temporal out-of-sample test.

Improved strategy parameters (from strategy_improvements.py analysis):
  - z >= 7 (was 5)
  - short YES only (was both directions)
  - hold cap 12h (was 48h)
  - price in [0.10, 0.90]
  - dte not in [7, 30] and < 365

Out-of-sample test:
  - Split historical trades by entry timestamp (median)
  - "Train" on first half: confirm baseline expectations
  - "Test" on second half: see if same edge holds
  - Critical because we found improvements via greedy search → overfit risk
"""
import json
import datetime
from pathlib import Path
import numpy as np

SPREAD_COST = 0.02
FEE_RATE = 0.025
N_SIM = 10_000

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

def net_pnl_per_100(t):
    if t["direction"] == 1:
        shares = 100.0 / t["entry_price"]
    else:
        shares = 100.0 / (1 - t["entry_price"])
    gross = shares * t["ret_per_share"]
    f_entry = FEE_RATE * t["entry_price"] * (1 - t["entry_price"])
    f_exit = FEE_RATE * t["exit_price"] * (1 - t["exit_price"])
    return gross - shares * (f_entry + f_exit) - shares * SPREAD_COST

# Strategy filters
def filter_v1(t):
    """Original strategy"""
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    if t["dte"] is None or t["dte"] >= 365: return False
    if 7 <= t["dte"] < 30: return False
    return True

def filter_v2(t):
    """Improved: z>=7, short only, hold<12h"""
    if abs(t["entry_z"]) < 7.0: return False
    if t["direction"] != -1: return False  # short only
    if t["hold_hours"] >= 12: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    if t["dte"] is None or t["dte"] >= 365: return False
    if 7 <= t["dte"] < 30: return False
    return True

v1_trades = [t for t in trades_all if filter_v1(t)]
v2_trades = [t for t in trades_all if filter_v2(t)]

print(f"V1 (current strategy): {len(v1_trades)} trades")
print(f"V2 (improved):         {len(v2_trades)} trades")
print()

# ─── Per-trade stats ───
def stats(trades, label):
    if not trades: return
    pnls = np.array([net_pnl_per_100(t) for t in trades])
    print(f"--- {label} ---")
    print(f"  N trades:      {len(trades)}")
    print(f"  Mean PnL/$100: ${pnls.mean():+.2f}")
    print(f"  Median:        ${np.median(pnls):+.2f}")
    print(f"  Std:           ${pnls.std():.2f}")
    print(f"  Win rate:      {(pnls > 0).mean()*100:.1f}%")
    print(f"  Sharpe:        {pnls.mean()/pnls.std():.3f}")
    print(f"  Worst:         ${pnls.min():.2f}")
    print(f"  P5/P95:        ${np.percentile(pnls,5):+.2f} / ${np.percentile(pnls,95):+.2f}")
    return pnls

v1_pnls = stats(v1_trades, "V1 (current)")
print()
v2_pnls = stats(v2_trades, "V2 (improved)")
print()

# ─── OUT-OF-SAMPLE TEST: temporal split ───
print("=" * 70)
print("OUT-OF-SAMPLE TEST — temporal split (median entry_ts)")
print("=" * 70)
print("Risk: v2 filters were found by greedy search → may be overfit.")
print("Test: split data by time, verify v2 edge holds in BOTH halves.\n")

# Use ALL z>=2 base trades for splitting (so we have same temporal coverage for both)
all_sorted = sorted(trades_all, key=lambda t: t["entry_ts"])
split_ts = all_sorted[len(all_sorted)//2]["entry_ts"]
split_dt = datetime.datetime.fromtimestamp(split_ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")

first_half = [t for t in trades_all if t["entry_ts"] < split_ts]
second_half = [t for t in trades_all if t["entry_ts"] >= split_ts]
print(f"Split point: {split_dt} UTC")
print(f"First half:  {len(first_half):,} raw trades")
print(f"Second half: {len(second_half):,} raw trades\n")

for half_name, half in [("FIRST HALF", first_half), ("SECOND HALF", second_half)]:
    print(f"\n{half_name}:")
    for fname, fn in [("V1 (z>=5, both dirs, 48h)", filter_v1), ("V2 (z>=7, short, 12h)", filter_v2)]:
        sub = [t for t in half if fn(t)]
        if not sub:
            print(f"  {fname}: NO trades")
            continue
        pnls = np.array([net_pnl_per_100(t) for t in sub])
        sharpe = pnls.mean()/pnls.std() if pnls.std() > 0 else 0
        print(f"  {fname}: n={len(sub):>4}  mean=${pnls.mean():>+7.2f}  win%={(pnls>0).mean()*100:>5.1f}  sharpe={sharpe:+.3f}  total=${pnls.sum():>+8.2f}")

# Bootstrap CI on second-half v2 to see if edge is statistically significant
print("\n--- Bootstrap CI on SECOND HALF v2 strategy ---")
v2_second = [t for t in second_half if filter_v2(t)]
if v2_second:
    pnls_2nd = np.array([net_pnl_per_100(t) for t in v2_second])
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(5000):
        sample = rng.choice(pnls_2nd, size=len(pnls_2nd), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    print(f"  Observed mean: ${pnls_2nd.mean():+.2f} (n={len(v2_second)})")
    print(f"  Bootstrap 95% CI: [${np.percentile(boot_means, 2.5):+.2f}, ${np.percentile(boot_means, 97.5):+.2f}]")
    print(f"  P(mean > 0):     {(boot_means > 0).mean()*100:.1f}%")
    print(f"  P(mean > $5):    {(boot_means > 5).mean()*100:.1f}%")
    print(f"  P(mean > $10):   {(boot_means > 10).mean()*100:.1f}%")

# ─── Monte Carlo with V2 strategy ───
print("\n" + "=" * 70)
print("MONTE CARLO — V2 strategy")
print("=" * 70)

# Calibrate trade frequency: v2 had 99 trades over ~30 days = 3.3 trades/day
# (vs 10.8/day for v1)
HORIZONS_DAYS = [7, 14, 30, 60, 90]
V2_TRADES_PER_DAY = len(v2_trades) / 30.0
print(f"\nEstimated v2 trade frequency: {V2_TRADES_PER_DAY:.1f}/day")

if v2_pnls is not None:
    rng = np.random.default_rng(42)
    print(f"\n  {'Horizon':>8} {'n_trades':>9} {'p5':>10} {'p50':>10} {'p95':>10} {'mean':>10} {'P(>0)':>8} {'P(>$1k)':>9} {'P(<-$500)':>10}")
    for d in HORIZONS_DAYS:
        n = max(1, int(d * V2_TRADES_PER_DAY))
        sims = np.array([rng.choice(v2_pnls, size=n, replace=True).sum() for _ in range(N_SIM)])
        p5 = np.percentile(sims, 5)
        p50 = np.percentile(sims, 50)
        p95 = np.percentile(sims, 95)
        print(f"  {d:>4}d {n:>9} ${p5:>+8.2f} ${p50:>+8.2f} ${p95:>+8.2f} ${sims.mean():>+8.2f} {100*(sims>0).mean():>6.1f}% {100*(sims>1000).mean():>7.1f}% {100*(sims<-500).mean():>8.1f}%")

# ─── Compare V1 vs V2 head-to-head on 30 days ───
print("\n" + "=" * 70)
print("V1 vs V2 head-to-head (30-day, $5k bankroll, $100/trade)")
print("=" * 70)

rng = np.random.default_rng(42)
v1_per_day = len(v1_trades) / 30.0
v2_per_day = len(v2_trades) / 30.0

for label, pnls, freq in [("V1 (current)", v1_pnls, v1_per_day),
                          ("V2 (improved)", v2_pnls, v2_per_day)]:
    if pnls is None: continue
    n = max(1, int(30 * freq))
    sims = np.array([rng.choice(pnls, size=n, replace=True).sum() for _ in range(N_SIM)])
    print(f"\n  {label}:")
    print(f"    Trades/day:    {freq:.1f}")
    print(f"    Avg trade:     ${pnls.mean():+.2f}")
    print(f"    30d median PnL: ${np.percentile(sims, 50):+.0f}")
    print(f"    30d p5 (bad):   ${np.percentile(sims, 5):+.0f}")
    print(f"    30d p95 (good): ${np.percentile(sims, 95):+.0f}")
    print(f"    P(profit):     {(sims > 0).mean()*100:.1f}%")
