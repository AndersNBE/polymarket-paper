"""
Monte Carlo on Strategy B (filtered z>=5 mean reversion).

Bootstrap from the 323 historical filtered trades to simulate many possible
"futures" over different horizons. Output:

  - Distribution of total PnL after 7, 14, 30, 60, 90 days
  - P(profitable), P(beats S&P), P(loss > $X)
  - Max drawdown distribution
  - Sample equity curves
  - Sensitivity: what if signal degrades 50%? What if costs are 2x?

Run:
  python3 monte_carlo.py
"""
import json
import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────
N_SIM            = 10_000
HORIZONS_DAYS    = [7, 14, 30, 60, 90]
BANKROLL         = 5_000.0       # USD
POSITION_USD     = 100.0         # USD per trade
SPREAD_COST      = 0.02          # implicit spread cost per share, ~2¢ median
FEE_RATE_AVG     = 0.025         # average per side (Polymarket is 3-7% per category, we use 5% peak adjusted)
SEED             = 42

# Trade-frequency: from backtest we had 323 filtered trades over ~30 days
# = ~10.8/day. We cap at 30/day max to be realistic about position limits.
HISTORICAL_TRADES_PER_DAY = 10.8

# ────────────────────────────────────────────────────────────────────
# Load + filter historical trades (same logic as paper_trader)
# ────────────────────────────────────────────────────────────────────
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

def keep(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    if t.get("dte") is None: return False
    if t["dte"] >= 365: return False
    if 7 <= t["dte"] < 30: return False
    return True

filtered = [t for t in trades_all if keep(t)]
print(f"Filtered trades available: {len(filtered):,}")

# ────────────────────────────────────────────────────────────────────
# Convert each historical trade to a $ PnL on POSITION_USD stake
# ────────────────────────────────────────────────────────────────────
def trade_dollar_pnl(t, stake_usd, spread=SPREAD_COST, fee_rate=FEE_RATE_AVG):
    """Return $ PnL on stake_usd for a single trade, after spread+fees."""
    if t["direction"] == 1:    # long YES
        shares = stake_usd / t["entry_price"]
    else:                      # short YES (== long NO)
        shares = stake_usd / (1.0 - t["entry_price"])
    gross_per_share = t["ret_per_share"]
    # Polymarket fee per side
    f_entry = fee_rate * t["entry_price"] * (1 - t["entry_price"])
    f_exit  = fee_rate * t["exit_price"] * (1 - t["exit_price"])
    fees = shares * (f_entry + f_exit)
    spread_cost = shares * spread
    return shares * gross_per_share - fees - spread_cost

pnl_per_trade = np.array([trade_dollar_pnl(t, POSITION_USD) for t in filtered])
print(f"\nPer-trade $ PnL on ${POSITION_USD} stake:")
print(f"  Mean:    ${pnl_per_trade.mean():+.3f}")
print(f"  Median:  ${np.median(pnl_per_trade):+.3f}")
print(f"  Std:     ${pnl_per_trade.std():.3f}")
print(f"  Skew:    {((pnl_per_trade - pnl_per_trade.mean())**3).mean() / pnl_per_trade.std()**3:.3f}")
print(f"  P5 / P95: ${np.percentile(pnl_per_trade, 5):+.2f} / ${np.percentile(pnl_per_trade, 95):+.2f}")
print(f"  Worst trade: ${pnl_per_trade.min():.2f}")
print(f"  Best trade:  ${pnl_per_trade.max():.2f}")
print(f"  Win rate (after costs): {(pnl_per_trade > 0).mean()*100:.1f}%")
print(f"  Expected daily ROI: ${pnl_per_trade.mean() * HISTORICAL_TRADES_PER_DAY:.2f} (on $5k bankroll = {100*pnl_per_trade.mean() * HISTORICAL_TRADES_PER_DAY / BANKROLL:.2f}%/day)")

# ────────────────────────────────────────────────────────────────────
# Monte Carlo: bootstrap N trades for each horizon, repeat N_SIM times
# ────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def simulate(horizon_days, trade_freq=HISTORICAL_TRADES_PER_DAY, n_sim=N_SIM, pnl_source=pnl_per_trade):
    n_trades = max(1, int(horizon_days * trade_freq))
    results = []
    paths_sample = []
    for i in range(n_sim):
        sample = rng.choice(pnl_source, size=n_trades, replace=True)
        cumsum = np.cumsum(sample)
        rolling_max = np.maximum.accumulate(cumsum)
        max_dd = (cumsum - rolling_max).min()
        results.append((cumsum[-1], max_dd, (sample > 0).mean()))
        if i < 200:
            paths_sample.append(cumsum)
    return np.array(results), paths_sample

print(f"\n{'='*70}")
print(f"Monte Carlo: {N_SIM:,} simulations × {len(HORIZONS_DAYS)} horizons")
print(f"{'='*70}")

all_results = {}
for d in HORIZONS_DAYS:
    res, paths = simulate(d)
    all_results[d] = {"results": res, "paths": paths}
    pnls = res[:, 0]
    dds = res[:, 1]
    n_trades = int(d * HISTORICAL_TRADES_PER_DAY)
    print(f"\n── Horizon: {d:>3} days  ({n_trades} trades, ${POSITION_USD} each) ──")
    print(f"  Total $ PnL:")
    print(f"    p 5  → ${np.percentile(pnls, 5):+9.2f}")
    print(f"    p25  → ${np.percentile(pnls, 25):+9.2f}")
    print(f"    p50  → ${np.percentile(pnls, 50):+9.2f}   ← median outcome")
    print(f"    p75  → ${np.percentile(pnls, 75):+9.2f}")
    print(f"    p95  → ${np.percentile(pnls, 95):+9.2f}")
    print(f"    mean → ${pnls.mean():+9.2f}")
    print(f"  Probabilities:")
    print(f"    P(profitable):           {(pnls > 0).mean()*100:5.1f}%")
    print(f"    P(>+$500):               {(pnls >  500).mean()*100:5.1f}%")
    print(f"    P(>+$1000):              {(pnls > 1000).mean()*100:5.1f}%")
    print(f"    P(<-$500):               {(pnls < -500).mean()*100:5.1f}%")
    print(f"    P(<-$1000 / bankroll>20% loss):   {(pnls < -1000).mean()*100:5.1f}%")
    print(f"  Max drawdown:")
    print(f"    median:      ${np.percentile(dds, 50):8.2f}")
    print(f"    p25 (typical worst): ${np.percentile(dds, 25):8.2f}")
    print(f"    p5  (bad):           ${np.percentile(dds, 5):8.2f}")
    print(f"  ROI vs $5k bankroll: median={100*np.percentile(pnls,50)/BANKROLL:+.1f}%, mean={100*pnls.mean()/BANKROLL:+.1f}%")

# ────────────────────────────────────────────────────────────────────
# Sensitivity analysis
# ────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("Sensitivity: 30-day horizon, varying assumptions")
print(f"{'='*70}")

# Scenario A: signal degrades 50% (mean halved, std preserved)
degraded_pnl = pnl_per_trade - (pnl_per_trade.mean() * 0.5)  # shift mean down by 50%
# Scenario B: trade frequency halved (e.g. less universe coverage)
half_freq_n = int(30 * HISTORICAL_TRADES_PER_DAY * 0.5)
# Scenario C: 2x spread costs (live trading has more slippage than assumed)
high_cost_pnl = np.array([trade_dollar_pnl(t, POSITION_USD, spread=0.04, fee_rate=0.035) for t in filtered])

scenarios = [
    ("Baseline (backtest assumptions)", pnl_per_trade, int(30 * HISTORICAL_TRADES_PER_DAY)),
    ("Signal degrades 50%", degraded_pnl, int(30 * HISTORICAL_TRADES_PER_DAY)),
    ("Trade frequency halved", pnl_per_trade, half_freq_n),
    ("Spread+fees 2x bigger", high_cost_pnl, int(30 * HISTORICAL_TRADES_PER_DAY)),
    ("All three together (pessimistic)", high_cost_pnl - (high_cost_pnl.mean() * 0.5), half_freq_n),
]

print(f"\n  {'Scenario':<40} {'p5':>9} {'p50':>9} {'p95':>9} {'P(>0)':>8} {'P(<-1k)':>8}")
print(f"  {'-'*84}")
for name, pnl_src, n_t in scenarios:
    sims = []
    for _ in range(N_SIM):
        s = rng.choice(pnl_src, size=n_t, replace=True)
        sims.append(s.sum())
    sims = np.array(sims)
    print(f"  {name:<40} ${np.percentile(sims, 5):+8.2f} ${np.percentile(sims, 50):+8.2f} ${np.percentile(sims, 95):+8.2f} {100*(sims>0).mean():>6.1f}% {100*(sims<-1000).mean():>6.1f}%")

# ────────────────────────────────────────────────────────────────────
# Plots
# ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Distribution of 30-day PnL
ax = axes[0, 0]
pnls_30 = all_results[30]["results"][:, 0]
ax.hist(pnls_30, bins=60, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.3)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Break-even")
ax.axvline(pnls_30.mean(), color="green", linestyle="-", linewidth=1.5, label=f"Mean ${pnls_30.mean():.0f}")
ax.axvline(np.percentile(pnls_30, 50), color="orange", linestyle="-", linewidth=1.5, label=f"Median ${np.percentile(pnls_30, 50):.0f}")
ax.set_xlabel("30-day Total PnL ($)")
ax.set_ylabel("Frequency")
ax.set_title(f"30-day PnL distribution ({N_SIM:,} sims)\n$100 trades on ${BANKROLL:.0f} bankroll")
ax.legend()

# 2. Sample equity curves (30 days)
ax = axes[0, 1]
paths = all_results[30]["paths"]
for p in paths:
    ax.plot(p, alpha=0.08, color="steelblue", linewidth=0.6)
median_path = np.median(np.array(paths), axis=0)
ax.plot(median_path, color="orange", linewidth=2, label="Median path")
ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
ax.set_xlabel("Trades elapsed")
ax.set_ylabel("Cumulative PnL ($)")
ax.set_title("200 simulated equity curves (30 days)")
ax.legend()

# 3. P(profitable) vs horizon
ax = axes[0, 2]
probs = []
medians = []
for d in HORIZONS_DAYS:
    r = all_results[d]["results"][:, 0]
    probs.append((r > 0).mean() * 100)
    medians.append(np.percentile(r, 50))
ax.plot(HORIZONS_DAYS, probs, marker="o", linewidth=2, color="green")
ax.set_xlabel("Horizon (days)")
ax.set_ylabel("Probability profitable (%)", color="green")
ax.tick_params(axis="y", labelcolor="green")
ax2 = ax.twinx()
ax2.plot(HORIZONS_DAYS, medians, marker="s", linewidth=2, color="steelblue", linestyle="--")
ax2.set_ylabel("Median expected PnL ($)", color="steelblue")
ax2.tick_params(axis="y", labelcolor="steelblue")
ax.set_title("P(profitable) and median PnL vs horizon")
ax.grid(True, alpha=0.3)

# 4. Max drawdown distribution
ax = axes[1, 0]
dds = all_results[30]["results"][:, 1]
ax.hist(dds, bins=50, color="crimson", alpha=0.7, edgecolor="black", linewidth=0.3)
ax.axvline(np.percentile(dds, 5), color="black", linestyle="--", label=f"p5: ${np.percentile(dds, 5):.0f}")
ax.set_xlabel("Max drawdown over 30 days ($)")
ax.set_ylabel("Frequency")
ax.set_title("Max drawdown distribution (30d)")
ax.legend()

# 5. Per-trade PnL distribution
ax = axes[1, 1]
ax.hist(pnl_per_trade, bins=50, color="purple", alpha=0.7, edgecolor="black", linewidth=0.3)
ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
ax.axvline(pnl_per_trade.mean(), color="orange", linewidth=1.5, label=f"Mean ${pnl_per_trade.mean():.2f}")
ax.set_xlabel("Per-trade $ PnL")
ax.set_ylabel("Count")
ax.set_title(f"Per-trade PnL (historical, n={len(pnl_per_trade)})")
ax.legend()

# 6. P(drawdown >= X) — Value at Risk curve
ax = axes[1, 2]
dd_thresh = np.linspace(0, -dds.min(), 100)
probs_dd = [100 * (dds <= -t).mean() for t in dd_thresh]
ax.plot(dd_thresh, probs_dd, linewidth=2, color="crimson")
ax.axhline(5, color="black", linestyle="--", linewidth=1, label="5%")
ax.set_xlabel("Drawdown threshold ($)")
ax.set_ylabel("P(max DD >= threshold) %")
ax.set_title("Drawdown VaR (30-day horizon)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path("monte_carlo_results.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"\nPlots saved to {out}")
print(f"Open with: open monte_carlo_results.png")
