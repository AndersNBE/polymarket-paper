"""
backtest_max_positions.py — Compare different max_open_positions values.

Replays the 30-day backtest with realistic costs, but varies the max-open
limit (5, 10, 15, 20, 25). Outputs:
  - Trade count
  - Skipped count
  - Net PnL
  - Max drawdown
  - Average capital deployed
  - Sharpe-equivalent

For each scenario, runs Monte Carlo to project 30/60/90-day distributions.
"""
import json
import datetime
import numpy as np
from pathlib import Path

STAKE = 30.0
GAS_PER_FILL = 0.15
FEE_RATES = {
    "sports_fees_v2": 0.03, "crypto_fees_v2": 0.07, "politics_fees": 0.04,
    "weather_fees": 0.05, "culture_fees": 0.05, "finance_prices_fees": 0.04,
    "tech_fees": 0.04, "economics_fees": 0.05, "mentions_fees": 0.04,
    "general_fees": 0.05, "crypto_15_min": 0.07, None: 0.05,
}

trades_all = json.loads(Path("results_B_v3.json").read_text())
markets = {m["id"]: m for m in json.loads(Path("markets.json").read_text())}

def dte_for(t):
    m = markets.get(t["market_id"])
    if not m or not m.get("endDate"):
        return None
    try:
        end_ts = datetime.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
        return (end_ts - t["entry_ts"]) / 86400
    except (ValueError, AttributeError):
        return None

def slip_for(entry_price, liquidity):
    if liquidity >= 25000: base = 0.005
    elif liquidity >= 10000: base = 0.008
    elif liquidity >= 3000: base = 0.012
    elif liquidity >= 1000: base = 0.020
    else: base = 0.040
    if entry_price < 0.15 or entry_price > 0.85:
        base *= 1.5
    return base

def fee_for(p, sh, ft):
    return sh * FEE_RATES.get(ft, FEE_RATES[None]) * p * (1 - p)

def net_pnl(t):
    if t["direction"] == 1:
        shares = STAKE / t["entry_price"]
    else:
        shares = STAKE / (1 - t["entry_price"])
    gross = shares * t["ret_per_share"]
    slip = slip_for(t["entry_price"], t["liquidity"])
    spread_cost = shares * slip * 2
    f_e = fee_for(t["entry_price"], shares, t["fee_type"])
    f_x = fee_for(t["exit_price"], shares, t["fee_type"])
    gas = GAS_PER_FILL * 2
    return gross - spread_cost - f_e - f_x - gas

# Annotate
for t in trades_all:
    m = markets.get(t["market_id"])
    t["fee_type"] = m.get("feeType") if m else None
    t["liquidity"] = float(m.get("liquidityNum") or 0) if m else 0
    t["dte"] = dte_for(t)

def keep_v1(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    if t["dte"] is None or t["dte"] >= 365 or t["dte"] < 0: return False
    if 7 <= t["dte"] < 30: return False
    return True

filtered = sorted([t for t in trades_all if keep_v1(t)], key=lambda t: t["entry_ts"])
print(f"Filtered trades available: {len(filtered)}")

# Simulate each max_positions scenario
def simulate(max_pos):
    open_pos = []   # list of exit_ts
    accepted = []
    skipped = 0
    max_concurrent = 0
    avg_concurrent = 0
    samples = 0
    for t in filtered:
        # Remove positions that have exited before this entry
        open_pos = [e for e in open_pos if e > t["entry_ts"]]
        if len(open_pos) >= max_pos:
            skipped += 1
            continue
        accepted.append(t)
        open_pos.append(t["exit_ts"])
        max_concurrent = max(max_concurrent, len(open_pos))
        avg_concurrent += len(open_pos)
        samples += 1
    if samples > 0:
        avg_concurrent /= samples

    pnls = np.array([net_pnl(t) for t in accepted])
    cumsum = np.cumsum(pnls) if len(pnls) else np.array([0.0])
    running_max = np.maximum.accumulate(cumsum)
    max_dd = (cumsum - running_max).min()

    return {
        "max_pos": max_pos,
        "n_accepted": len(accepted),
        "n_skipped": skipped,
        "total_pnl": pnls.sum() if len(pnls) else 0,
        "mean_pnl": pnls.mean() if len(pnls) else 0,
        "win_rate": (pnls > 0).mean() * 100 if len(pnls) else 0,
        "sharpe": pnls.mean() / pnls.std() if pnls.std() > 0 else 0,
        "max_dd": max_dd,
        "max_concurrent": max_concurrent,
        "avg_concurrent": avg_concurrent,
        "max_capital_deployed": max_pos * STAKE,
        "pnls": pnls,
    }

scenarios = [3, 5, 8, 10, 12, 15, 18, 20, 24]
results = [simulate(m) for m in scenarios]

# Print comparison table
print(f"\n{'='*100}")
print(f"COMPARISON: Different max_open_positions")
print(f"{'='*100}")
print(f"  {'Max':>4} {'Accept':>7} {'Skip':>5} {'Total $':>10} {'Mean $':>9} {'Win%':>6} {'Sharpe':>8} {'Max DD':>9} {'Peak open':>10} {'Avg open':>9}")
for r in results:
    print(f"  {r['max_pos']:>4} {r['n_accepted']:>7} {r['n_skipped']:>5} ${r['total_pnl']:>+8.2f} ${r['mean_pnl']:>+7.2f} {r['win_rate']:>5.1f}% {r['sharpe']:>+7.3f} ${r['max_dd']:>+7.2f} {r['max_concurrent']:>10} {r['avg_concurrent']:>8.1f}")

# Detailed Monte Carlo for 10 vs 15
print(f"\n{'='*100}")
print(f"MONTE CARLO: 10 vs 15 positions, projected to 30 days")
print(f"{'='*100}")

import random
rng = np.random.default_rng(42)

# Use observed trade rates from 30-day backtest
N_SIM = 10000
for r in results:
    if r["max_pos"] not in (10, 15, 20): continue
    if len(r["pnls"]) == 0: continue
    rate_per_day = r["n_accepted"] / 30.0
    print(f"\n  max={r['max_pos']}: observed {rate_per_day:.1f} trades/day, mean=${r['mean_pnl']:+.2f}")
    for d in [7, 30, 90]:
        n = max(1, int(d * rate_per_day))
        sims = np.array([rng.choice(r["pnls"], n, replace=True).sum() for _ in range(N_SIM)])
        print(f"    {d}d: p5=${np.percentile(sims, 5):>+7.0f}  p50=${np.percentile(sims, 50):>+7.0f}  p95=${np.percentile(sims, 95):>+7.0f}  P(>0)={100*(sims>0).mean():>5.1f}%")

# Find sweet spot
print(f"\n{'='*100}")
print(f"SWEET SPOT ANALYSIS")
print(f"{'='*100}")
best_total = max(results, key=lambda r: r["total_pnl"])
best_sharpe = max(results, key=lambda r: r["sharpe"])
best_capital_eff = max(results, key=lambda r: r["total_pnl"] / r["max_capital_deployed"] if r["max_capital_deployed"] > 0 else 0)
print(f"  Best absolute total: max={best_total['max_pos']}, total=${best_total['total_pnl']:.2f}")
print(f"  Best Sharpe: max={best_sharpe['max_pos']}, sharpe={best_sharpe['sharpe']:.3f}")
print(f"  Best capital efficiency (PnL per $ at risk): max={best_capital_eff['max_pos']}, ratio={best_capital_eff['total_pnl']/best_capital_eff['max_capital_deployed']:.3f}")

# Print recommendation
print(f"\n{'='*100}")
print(f"RECOMMENDATION")
print(f"{'='*100}")
r10 = next(r for r in results if r["max_pos"] == 10)
r15 = next(r for r in results if r["max_pos"] == 15)
r20 = next(r for r in results if r["max_pos"] == 20)
print(f"  max=10: ${r10['total_pnl']:.2f} total, skips {r10['n_skipped']} trades, max DD ${r10['max_dd']:.2f}")
print(f"  max=15: ${r15['total_pnl']:.2f} total, skips {r15['n_skipped']} trades, max DD ${r15['max_dd']:.2f}")
print(f"  max=20: ${r20['total_pnl']:.2f} total, skips {r20['n_skipped']} trades, max DD ${r20['max_dd']:.2f}")
print(f"\n  Delta 10→15: +${r15['total_pnl'] - r10['total_pnl']:.2f} PnL ({r10['n_skipped'] - r15['n_skipped']} fewer skipped), worse DD by ${r10['max_dd'] - r15['max_dd']:.2f}")
print(f"  Delta 15→20: +${r20['total_pnl'] - r15['total_pnl']:.2f} PnL ({r15['n_skipped'] - r20['n_skipped']} fewer skipped), worse DD by ${r15['max_dd'] - r20['max_dd']:.2f}")
