"""
analyze_extended.py — Analysis of extended backtest results vs original 30-day.

Reads extended_trades.jsonl, applies same filters as paper_trader, and compares
to original results_B_v3.json findings. Critical questions:

  1. Does z>=5 + filters preserve edge on longer time horizon?
  2. Was the 30-day result a sampling artifact?
  3. What time-period was strongest/weakest?
  4. How should we revise Monte Carlo expectations?
"""
import json
import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

# Cost model (same as monte_carlo + paper_trader)
SPREAD_COST = 0.02
FEE_RATE = 0.025

def to_dt(ts):
    return datetime.datetime.fromtimestamp(int(ts), datetime.timezone.utc)

def keep(t):
    """Same filter as paper_trader."""
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    dte = t.get("dte_days_at_entry")
    if dte is None: return False
    if dte >= 365: return False
    if 7 <= dte < 30: return False
    if dte < 0: return False
    return True

def dollar_pnl_per_100(t):
    """Net $ PnL on $100 stake."""
    if t["direction"] == 1:
        shares = 100.0 / t["entry_price"]
    else:
        shares = 100.0 / (1 - t["entry_price"])
    gross = shares * t["ret_per_share"]
    f_entry = FEE_RATE * t["entry_price"] * (1 - t["entry_price"])
    f_exit = FEE_RATE * t["exit_price"] * (1 - t["exit_price"])
    return gross - shares * (f_entry + f_exit) - shares * SPREAD_COST

# Load extended trades
ext_trades = []
with open("extended_trades.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                ext_trades.append(json.loads(line))
            except json.JSONDecodeError:
                pass

print(f"=== EXTENDED BACKTEST ANALYSIS ===\n")
print(f"Total raw trades:    {len(ext_trades):,}")

filtered = [t for t in ext_trades if keep(t)]
print(f"Filtered (z>=5):     {len(filtered):,}")

# Compare to original
try:
    orig = json.loads(Path("results_B_v3.json").read_text())
    # We can't filter origs without markets.json for dte, so just count z>=5 + price
    orig_basic = [t for t in orig if abs(t["entry_z"]) >= 5.0 and 0.10 <= t["entry_price"] <= 0.90]
    print(f"\nOriginal 30-day (z>=5 + price filter): {len(orig_basic):,}")
except (FileNotFoundError, json.JSONDecodeError):
    orig_basic = []

if not filtered:
    print("\n⚠ No filtered trades — extended backtest didn't produce enough signal.")
    print("  Likely reasons:")
    print("    - Resolved markets had shorter price histories")
    print("    - Different market mix vs active markets")
    print("    - Strategy edge was specific to active-market regime")
    exit()

# ─── Stats on filtered trades ───
pnls = np.array([dollar_pnl_per_100(t) for t in filtered])
print(f"\n--- Filtered trade stats ---")
print(f"  Mean $/trade:    ${pnls.mean():+.2f}")
print(f"  Median:          ${np.median(pnls):+.2f}")
print(f"  Std:             ${pnls.std():.2f}")
print(f"  Win rate:        {(pnls > 0).mean()*100:.1f}%")
print(f"  P5 / P95:        ${np.percentile(pnls, 5):+.2f} / ${np.percentile(pnls, 95):+.2f}")
print(f"  Worst:           ${pnls.min():.2f}")
print(f"  Best:            ${pnls.max():.2f}")

# ─── Temporal distribution: by quarter ───
print(f"\n--- Temporal distribution (by entry-quarter) ---")
by_q = defaultdict(list)
for t, pnl in zip(filtered, pnls):
    dt = to_dt(t["entry_ts"])
    q_key = f"{dt.year}-Q{(dt.month-1)//3 + 1}"
    by_q[q_key].append(pnl)

for q in sorted(by_q.keys()):
    arr = np.array(by_q[q])
    print(f"  {q}: n={len(arr):>4}  mean=${arr.mean():+7.2f}  win%={(arr>0).mean()*100:>5.1f}  total=${arr.sum():+9.2f}")

# ─── Z magnitude impact ───
print(f"\n--- By z magnitude ---")
zs = np.array([abs(t["entry_z"]) for t in filtered])
for lo, hi in [(5, 7), (7, 10), (10, 15), (15, 30), (30, 100), (100, 1e6)]:
    mask = (zs >= lo) & (zs < hi)
    if mask.sum() == 0: continue
    sel = pnls[mask]
    print(f"  z in [{lo:>3}, {hi:>3}): n={mask.sum():>4}  mean=${sel.mean():+7.2f}  win%={(sel>0).mean()*100:>5.1f}")

# ─── Compare to original baseline ───
if orig_basic:
    # Compute same stats on original
    orig_pnls_per_share = np.array([t["ret_per_share"] for t in orig_basic])
    orig_entries = np.array([t["entry_price"] for t in orig_basic])
    orig_exits = np.array([t["exit_price"] for t in orig_basic])
    orig_dirs = np.array([t["direction"] for t in orig_basic])
    orig_shares = np.where(orig_dirs == 1, 100.0/orig_entries, 100.0/(1-orig_entries))
    orig_gross = orig_shares * orig_pnls_per_share
    orig_fees = orig_shares * (FEE_RATE * orig_entries * (1-orig_entries) + FEE_RATE * orig_exits * (1-orig_exits))
    orig_spread = orig_shares * SPREAD_COST
    orig_net = orig_gross - orig_fees - orig_spread

    print(f"\n--- COMPARISON: Extended vs Original ---")
    print(f"  Metric              {'Original':>14}  {'Extended':>14}")
    print(f"  N trades            {len(orig_basic):>14,}  {len(filtered):>14,}")
    print(f"  Mean $/trade        {f'${orig_net.mean():+.2f}':>14}  {f'${pnls.mean():+.2f}':>14}")
    print(f"  Win rate            {f'{(orig_net>0).mean()*100:.1f}%':>14}  {f'{(pnls>0).mean()*100:.1f}%':>14}")
    print(f"  Std                 {f'${orig_net.std():.2f}':>14}  {f'${pnls.std():.2f}':>14}")
    print(f"  Sharpe              {orig_net.mean()/orig_net.std():>14.3f}  {pnls.mean()/pnls.std():>14.3f}")

# ─── Save filtered trades for further analysis ───
out = Path("extended_filtered.json")
out.write_text(json.dumps([{**t, "pnl_per_100": float(p)} for t, p in zip(filtered, pnls)], indent=2, default=str))
print(f"\nFiltered trades saved to {out}")

# ─── Monte Carlo on extended data ───
if len(filtered) >= 30:
    print(f"\n--- Monte Carlo on extended (10k sims, 30-day horizon) ---")
    # Assume same trade frequency as original backtest
    # Original: 323 filtered in 30 days = ~10.8/day
    # Extended: filtered over (12 months / 30 days) periods worth of data
    # We just bootstrap N=300 trades to simulate 30 days
    rng = np.random.default_rng(42)
    results = []
    for _ in range(10000):
        s = rng.choice(pnls, size=300, replace=True)
        results.append(s.sum())
    results = np.array(results)
    print(f"  30-day total PnL:")
    print(f"    p5:    ${np.percentile(results, 5):+.2f}")
    print(f"    p50:   ${np.percentile(results, 50):+.2f}")
    print(f"    p95:   ${np.percentile(results, 95):+.2f}")
    print(f"    mean:  ${results.mean():+.2f}")
    print(f"  P(profitable): {(results > 0).mean()*100:.1f}%")
    print(f"  P(>+$1000):    {(results > 1000).mean()*100:.1f}%")
    print(f"  P(<-$1000):    {(results < -1000).mean()*100:.1f}%")

print(f"\n=== END ANALYSIS ===")
