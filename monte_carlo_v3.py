"""
monte_carlo_v3.py — Monte Carlo with HONEST extended-v3 data + smaller bankroll.

Uses 138 filtered trades from extended_v3 backtest (resolved $5k-$50k markets,
20 months span). This matches our live paper-trader universe most accurately.

Two key changes from monte_carlo_v2:
  1. Use extended_v3 per-trade distribution (smaller mean, bigger std)
  2. Realistic bankroll: 5000 DKK = $720 USD
  3. Position size: $30 (4% of bankroll) — safer for high variance

Tests range of trade-frequency assumptions since that's uncertain.
"""
import json
import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPREAD = 0.02
FEE = 0.025
N_SIM = 10_000

# User's actual setup
BANKROLL_USD = 720.0   # 5000 DKK
POSITION_USD = 30.0    # 4.2% of bankroll
HORIZONS_DAYS = [7, 14, 30, 60, 90, 180]

# Strategy filters (V1 only - V2 was overfit)
def keep_v1(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    d = t.get("dte_days_at_entry")
    if d is None or d >= 365 or d < 0 or 7 <= d < 30: return False
    return True

def net_pnl_per_dollar(t):
    if t["direction"] == 1:
        shares = 1.0 / t["entry_price"]
    else:
        shares = 1.0 / (1 - t["entry_price"])
    gross = shares * t["ret_per_share"]
    f_entry = FEE * t["entry_price"] * (1 - t["entry_price"])
    f_exit = FEE * t["exit_price"] * (1 - t["exit_price"])
    return gross - shares * (f_entry + f_exit) - shares * SPREAD

# Load extended_v3 trades
trades = []
with open("extended_trades_v3.jsonl") as f:
    for line in f:
        if line.strip():
            trades.append(json.loads(line))

v1 = [t for t in trades if keep_v1(t)]
print(f"V1 filtered trades from extended_v3 (right population): {len(v1)}")

pnl_per_dollar = np.array([net_pnl_per_dollar(t) for t in v1])
pnl_per_stake = pnl_per_dollar * POSITION_USD

print(f"\nPer-trade $ PnL on ${POSITION_USD} stake:")
print(f"  Mean:    ${pnl_per_stake.mean():+.2f}")
print(f"  Median:  ${np.median(pnl_per_stake):+.2f}")
print(f"  Std:     ${pnl_per_stake.std():.2f}")
print(f"  Win rate: {(pnl_per_stake > 0).mean()*100:.1f}%")
print(f"  Worst:   ${pnl_per_stake.min():.2f}")
print(f"  P5/P95:  ${np.percentile(pnl_per_stake, 5):+.2f} / ${np.percentile(pnl_per_stake, 95):+.2f}")

# Trade frequency: uncertain. Test three scenarios.
# - Conservative: extended_v3 rate (~0.46/day on 3000-market universe)
# - Optimistic: original 30d rate (~10.8/day)
# - Middle: 3.0/day
SCENARIOS = [
    ("PESSIMISTIC: extended rate (0.5/day)", 0.5),
    ("MIDDLE: (3/day)", 3.0),
    ("OPTIMISTIC: original rate (10.8/day)", 10.8),
]

rng = np.random.default_rng(42)

print(f"\n{'='*70}")
print(f"MONTE CARLO  ${POSITION_USD} stake on ${BANKROLL_USD:.0f} bankroll  (5000 DKK)")
print(f"{'='*70}")

# For each scenario × horizon
all_results = {}
for label, freq in SCENARIOS:
    print(f"\n--- {label} ---")
    print(f"  {'Days':>5} {'Trades':>7} {'p5':>9} {'p50':>9} {'p95':>9} {'P(>0)':>7} {'P(<-100)':>9} {'P(<-300)':>9}")
    for d in HORIZONS_DAYS:
        n = max(1, int(d * freq))
        sims = np.array([rng.choice(pnl_per_stake, n, replace=True).sum() for _ in range(N_SIM)])
        all_results[(label, d)] = sims
        print(f"  {d:>4}d {n:>7} ${np.percentile(sims, 5):>+7.2f} ${np.percentile(sims, 50):>+7.2f} ${np.percentile(sims, 95):>+7.2f} {100*(sims>0).mean():>5.1f}% {100*(sims<-100).mean():>7.1f}% {100*(sims<-300).mean():>7.1f}%")

# Worst-case max drawdown over 90 days, all scenarios
print(f"\n{'='*70}")
print(f"MAX DRAWDOWN over 90 days (P5/P50/P95 of worst peak-to-trough)")
print(f"{'='*70}")
print(f"  {'Scenario':<35} {'p5 (best)':>12} {'p50':>10} {'p95 (worst)':>14}")
for label, freq in SCENARIOS:
    n = max(1, int(90 * freq))
    dds = []
    for _ in range(N_SIM):
        s = rng.choice(pnl_per_stake, n, replace=True)
        cumsum = np.cumsum(s)
        running_max = np.maximum.accumulate(cumsum)
        max_dd = (cumsum - running_max).min()
        dds.append(max_dd)
    dds = np.array(dds)
    print(f"  {label:<35} ${np.percentile(dds, 5):>+10.2f} ${np.percentile(dds, 50):>+8.2f} ${np.percentile(dds, 95):>+12.2f}")

# Plots
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Per-trade distribution
ax = axes[0, 0]
ax.hist(pnl_per_stake, bins=30, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.3)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
ax.axvline(pnl_per_stake.mean(), color="orange", linewidth=2, label=f"Mean ${pnl_per_stake.mean():+.2f}")
ax.set_xlabel(f"PnL per ${POSITION_USD} trade")
ax.set_ylabel("Count")
ax.set_title(f"V1 historical distribution (n={len(pnl_per_stake)})\n5k DKK bankroll, $30 stake")
ax.legend()

# 30-day PnL distributions across scenarios
ax = axes[0, 1]
for label, _ in SCENARIOS:
    sims = all_results[(label, 30)]
    ax.hist(sims, bins=40, alpha=0.35, label=f"{label[:15]}", density=True)
ax.axvline(0, color="red", linestyle="--")
ax.set_xlabel("30-day total PnL ($)")
ax.set_ylabel("Density")
ax.set_title("30-day PnL across trade-frequency scenarios")
ax.legend(fontsize=8)

# P(profit) curves
ax = axes[0, 2]
for label, _ in SCENARIOS:
    probs = []
    days = HORIZONS_DAYS
    for d in days:
        p = (all_results[(label, d)] > 0).mean() * 100
        probs.append(p)
    ax.plot(days, probs, marker="o", label=label[:15])
ax.set_xlabel("Days")
ax.set_ylabel("P(profitable) %")
ax.set_title("P(net positive) vs horizon")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Median PnL curves
ax = axes[1, 0]
for label, _ in SCENARIOS:
    medians = [np.percentile(all_results[(label, d)], 50) for d in HORIZONS_DAYS]
    ax.plot(HORIZONS_DAYS, medians, marker="o", label=label[:15])
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xlabel("Days")
ax.set_ylabel("Median PnL ($)")
ax.set_title("Median expected PnL vs horizon")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 30-day sample paths for middle scenario
ax = axes[1, 1]
n = int(30 * 3.0)  # middle scenario
paths = []
for _ in range(100):
    s = rng.choice(pnl_per_stake, n, replace=True)
    paths.append(np.cumsum(s))
for p in paths:
    ax.plot(p, alpha=0.15, color="steelblue", linewidth=0.6)
median_path = np.median(np.array(paths), axis=0)
ax.plot(median_path, color="orange", linewidth=2.5, label="Median")
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative PnL ($)")
ax.set_title(f"100 30-day paths (middle scenario, 3 trades/day)")
ax.legend()

# Comparison vs holding cash
ax = axes[1, 2]
labels = ["7d", "14d", "30d", "60d", "90d", "180d"]
mid_p50 = [np.percentile(all_results[("MIDDLE: (3/day)", d)], 50) for d in HORIZONS_DAYS]
pess_p50 = [np.percentile(all_results[("PESSIMISTIC: extended rate (0.5/day)", d)], 50) for d in HORIZONS_DAYS]
opt_p50 = [np.percentile(all_results[("OPTIMISTIC: original rate (10.8/day)", d)], 50) for d in HORIZONS_DAYS]
x = np.arange(len(labels))
w = 0.25
ax.bar(x - w, pess_p50, w, label="Pessimistic", color="orange")
ax.bar(x, mid_p50, w, label="Middle", color="steelblue")
ax.bar(x + w, opt_p50, w, label="Optimistic", color="green")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Median PnL ($)")
ax.set_title(f"Median PnL on ${BANKROLL_USD:.0f} bankroll")
ax.axhline(0, color="black", linewidth=1)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("monte_carlo_v3.png", dpi=100)
print(f"\nPlots saved to monte_carlo_v3.png")

# Bottom line summary
print(f"\n{'='*70}")
print(f"BOTTOM LINE for 5000 DKK ($720) bankroll, $30 trades:")
print(f"{'='*70}")

mid_30 = all_results[("MIDDLE: (3/day)", 30)]
mid_90 = all_results[("MIDDLE: (3/day)", 90)]
opt_30 = all_results[("OPTIMISTIC: original rate (10.8/day)", 30)]
pess_30 = all_results[("PESSIMISTIC: extended rate (0.5/day)", 30)]

print(f"\nAfter 30 DAYS:")
print(f"  Pessimistic (15 trades total):  median ${np.percentile(pess_30, 50):+.0f}, P(>0)={100*(pess_30>0).mean():.0f}%")
print(f"  Middle      (90 trades total):  median ${np.percentile(mid_30, 50):+.0f}, P(>0)={100*(mid_30>0).mean():.0f}%")
print(f"  Optimistic  (324 trades total): median ${np.percentile(opt_30, 50):+.0f}, P(>0)={100*(opt_30>0).mean():.0f}%")
print(f"\nAfter 90 DAYS (middle):")
print(f"  Median: ${np.percentile(mid_90, 50):+.0f}")
print(f"  P5 (10% worst): ${np.percentile(mid_90, 5):+.0f}")
print(f"  P95 (10% best): ${np.percentile(mid_90, 95):+.0f}")
print(f"  P(profitable): {100*(mid_90>0).mean():.0f}%")
print(f"  P(loss >$200): {100*(mid_90<-200).mean():.0f}%")
