"""
Strategy B v4: Apply filters from blindspot analysis and re-test temporal stability.

Filters derived from blindspot tests:
  - z entry >= 5 (vs >= 4) for higher signal-to-noise
  - entry price in [0.10, 0.90] (avoid boundary effects)
  - days-to-resolution NOT in [7, 30] and NOT >365 (those buckets lose)

Question: does the filtered strategy preserve edge in the WEAKER second half?
"""
import json
import datetime
import time
from pathlib import Path
import numpy as np

markets_list = json.loads(Path("markets.json").read_text())
markets = {m["id"]: m for m in markets_list}
trades = json.loads(Path("results_B_v3.json").read_text())

def fee_per_trade(p, fee_rate=0.025):
    return 2 * fee_rate * p * (1 - p)

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

# Add days-to-end to each trade
for t in trades:
    t["dte"] = days_to_end(t)

# Filter rules to test, progressively
filter_specs = [
    ("baseline (z>=2)", lambda t: abs(t["entry_z"]) >= 2.0),
    ("z>=4", lambda t: abs(t["entry_z"]) >= 4.0),
    ("z>=5", lambda t: abs(t["entry_z"]) >= 5.0),
    ("z>=4 + in [0.10,0.90]", lambda t: abs(t["entry_z"]) >= 4.0 and 0.10 <= t["entry_price"] <= 0.90),
    ("z>=5 + in [0.10,0.90]", lambda t: abs(t["entry_z"]) >= 5.0 and 0.10 <= t["entry_price"] <= 0.90),
    ("z>=4 + in [0.10,0.90] + dte OK", lambda t: (abs(t["entry_z"]) >= 4.0
                                                     and 0.10 <= t["entry_price"] <= 0.90
                                                     and t.get("dte") is not None
                                                     and not (7 <= t["dte"] < 30)
                                                     and t["dte"] < 365)),
    ("z>=5 + in [0.10,0.90] + dte OK", lambda t: (abs(t["entry_z"]) >= 5.0
                                                     and 0.10 <= t["entry_price"] <= 0.90
                                                     and t.get("dte") is not None
                                                     and not (7 <= t["dte"] < 30)
                                                     and t["dte"] < 365)),
]

# Split timeline
sorted_trades = sorted(trades, key=lambda t: t["entry_ts"])
half = len(sorted_trades) // 2
splits = {
    "First half (apr 13 - may 1)": sorted_trades[:half],
    "Second half (may 1 - may 12)": sorted_trades[half:],
    "Full sample": sorted_trades,
}

print(f"{'filter':>40s}  {'split':>30s}  {'n':>5}  {'gross':>8}  {'net@2¢':>8}  {'net@1¢':>8}  {'win%':>6}  {'sharpe':>7}")
print("-" * 120)

for label, fn in filter_specs:
    for split_label, batch in splits.items():
        sel = [t for t in batch if fn(t)]
        if len(sel) < 5:
            continue
        rets = np.array([t["ret_per_share"] for t in sel])
        ents = np.array([t["entry_price"] for t in sel])
        exs = np.array([t["exit_price"] for t in sel])
        fees = np.array([fee_per_trade(p) for p in 0.5*(ents+exs)])
        net2 = rets - 0.02 - fees
        net1 = rets - 0.01 - fees
        win = (rets > 0).mean() * 100
        sharpe = rets.mean() / rets.std() if rets.std() > 0 else 0
        print(f"  {label:>38s}  {split_label:>30s}  {len(sel):>5}  {rets.mean()*100:>+6.2f}¢  {net2.mean()*100:>+6.2f}¢  {net1.mean()*100:>+6.2f}¢  {win:>5.1f}  {sharpe:>+6.3f}")
    print()

# Identify the BEST filter and analyze
print("=" * 70)
print("Identify markets where 2nd-half degrades vs 1st-half")
print("=" * 70)

best_filter = lambda t: (abs(t["entry_z"]) >= 5.0
                            and 0.10 <= t["entry_price"] <= 0.90
                            and t.get("dte") is not None
                            and not (7 <= t["dte"] < 30)
                            and t["dte"] < 365)

filt_trades = [t for t in sorted_trades if best_filter(t)]
filt_half = len(filt_trades) // 2
print(f"\nFiltered trades: {len(filt_trades)}")

# How many markets are in each half?
mids_first = set(t["market_id"] for t in filt_trades[:filt_half])
mids_second = set(t["market_id"] for t in filt_trades[filt_half:])
print(f"First half markets:  {len(mids_first)}")
print(f"Second half markets: {len(mids_second)}")
print(f"Overlap:             {len(mids_first & mids_second)}")
print(f"Second-half-only:    {len(mids_second - mids_first)}")
print(f"First-half-only:     {len(mids_first - mids_second)}")

# Question: are second-half markets fundamentally different?
def stats_for_mids(mids):
    ml = [markets.get(mid) for mid in mids if mid in markets]
    if not ml:
        return {}
    return {
        "median_liq": float(np.median([float(m.get("liquidityNum") or 0) for m in ml])),
        "median_v24": float(np.median([float(m.get("volume24hr") or 0) for m in ml])),
    }

print(f"\nFirst-half-only markets stats:")
for k, v in stats_for_mids(mids_first - mids_second).items():
    print(f"  {k}: {v:.0f}")
print(f"Second-half-only markets stats:")
for k, v in stats_for_mids(mids_second - mids_first).items():
    print(f"  {k}: {v:.0f}")

# Sharpe ratio per signal in each half (after filter)
first_filt = [t for t in sorted_trades[:half] if best_filter(t)]
second_filt = [t for t in sorted_trades[half:] if best_filter(t)]
for label, batch in [("First half FILTERED", first_filt), ("Second half FILTERED", second_filt)]:
    if not batch:
        continue
    rets = np.array([t["ret_per_share"] for t in batch])
    ents = np.array([t["entry_price"] for t in batch])
    exs = np.array([t["exit_price"] for t in batch])
    fees = np.array([fee_per_trade(p) for p in 0.5*(ents+exs)])
    net2 = rets - 0.02 - fees
    print(f"\n{label}: n={len(batch)}, gross={rets.mean()*100:+.2f}¢, net@2¢={net2.mean()*100:+.2f}¢, win%={(rets>0).mean()*100:.1f}, sharpe={rets.mean()/rets.std() if rets.std()>0 else 0:.3f}")
    print(f"  Worst 5 trades: {[f'{r*100:.0f}¢' for r in sorted(rets)[:5]]}")
    print(f"  Best 5 trades:  {[f'{r*100:.0f}¢' for r in sorted(rets)[-5:]]}")

# Bootstrap p-value: is the second half net positive?
print("\n=" * 70)
print("Bootstrap test: is filtered second-half net edge significantly > 0?")
print("=" * 70)
import random
random.seed(123)
if second_filt:
    rets = np.array([t["ret_per_share"] for t in second_filt])
    ents = np.array([t["entry_price"] for t in second_filt])
    exs = np.array([t["exit_price"] for t in second_filt])
    fees = np.array([fee_per_trade(p) for p in 0.5*(ents+exs)])
    net2 = rets - 0.02 - fees
    print(f"Observed second-half filtered net@2¢: {net2.mean()*100:+.3f}¢ (n={len(net2)})")
    # Bootstrap CI
    boot_means = []
    for _ in range(5000):
        sample = np.random.choice(net2, size=len(net2), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    print(f"Bootstrap 95% CI: [{np.percentile(boot_means, 2.5)*100:+.3f}¢, {np.percentile(boot_means, 97.5)*100:+.3f}¢]")
    print(f"P(mean > 0): {(boot_means > 0).mean()*100:.1f}%")
    print(f"P(mean > 0.5¢): {(boot_means > 0.005).mean()*100:.1f}%")
    print(f"P(mean > 1¢): {(boot_means > 0.01).mean()*100:.1f}%")
