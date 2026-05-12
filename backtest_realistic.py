"""
backtest_realistic.py — Re-evaluate strategy with all paper-trader improvements.

Applies on top of existing trade data (results_B_v3.json):
  1. Realistic slippage (approx of orderbook walk)
  2. CLV computation from historical price bars
  3. Max-open-positions limit (10) — filter trades to respect concurrency
  4. New gas cost ($0.15)

CLV is the gold metric — if avg CLV > 0 we have edge regardless of PnL noise.
"""
import json
import datetime
import time
from pathlib import Path
import numpy as np
import requests

CLOB = "https://clob.polymarket.com"

# Realistic cost model
FEE_RATES = {
    "sports_fees_v2": 0.03, "crypto_fees_v2": 0.07, "politics_fees": 0.04,
    "weather_fees": 0.05, "culture_fees": 0.05, "finance_prices_fees": 0.04,
    "tech_fees": 0.04, "economics_fees": 0.05, "mentions_fees": 0.04,
    "general_fees": 0.05, "crypto_15_min": 0.07, None: 0.05,
}
GAS_PER_FILL = 0.15
MAX_OPEN_POSITIONS = 10
STAKE = 30.0
CLV_LOOKBACK_HOURS = 1  # bars are hourly, so 1 bar later

trades = json.loads(Path("results_B_v3.json").read_text())
markets = {m["id"]: m for m in json.loads(Path("markets.json").read_text())}

def fee_for_fill(price, shares, fee_type):
    rate = FEE_RATES.get(fee_type, FEE_RATES[None])
    return shares * rate * price * (1 - price)

def slippage_estimate(entry_price, liquidity_usd):
    """Estimate cost of walking the book for $30 trade.

    Logic:
      - Tight books (high liquidity): ~1 tick slippage = 1¢
      - Medium liquidity ($2-10k): ~2 ticks = 2¢
      - Low liquidity: ~3-4 ticks = 3-4¢
      - Near boundaries (price near 0/1): worse because thin
    """
    base_slip = 0.01
    if liquidity_usd >= 25000:
        base_slip = 0.005
    elif liquidity_usd >= 10000:
        base_slip = 0.008
    elif liquidity_usd >= 3000:
        base_slip = 0.012
    elif liquidity_usd >= 1000:
        base_slip = 0.020
    else:
        base_slip = 0.040
    # Boundary penalty: prices < 0.15 or > 0.85 have thinner books
    if entry_price < 0.15 or entry_price > 0.85:
        base_slip *= 1.5
    return base_slip

def keep_v1(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    d = t.get("dte")
    if d is None or d >= 365 or d < 0 or 7 <= d < 30: return False
    return True

# Add metadata
import datetime
for t in trades:
    m = markets.get(t["market_id"])
    t["fee_type"] = m.get("feeType") if m else None
    t["liquidity"] = float(m.get("liquidityNum") or 0) if m else 0
    if m and m.get("endDate"):
        try:
            end_ts = datetime.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
            t["dte"] = (end_ts - t["entry_ts"]) / 86400
        except (ValueError, AttributeError):
            t["dte"] = None

filtered = [t for t in trades if keep_v1(t)]
print(f"Filtered (z>=5) trades: {len(filtered)}")

# ─────────────────────────────────────────────────────────────────
# Step 1: Apply realistic costs to existing trades
# ─────────────────────────────────────────────────────────────────
def realistic_pnl(t):
    """$ PnL on $30 stake with realistic costs."""
    if t["direction"] == 1:
        shares = STAKE / t["entry_price"]
    else:
        shares = STAKE / (1 - t["entry_price"])
    gross_per_share = t["ret_per_share"]
    gross = shares * gross_per_share

    # Cost: per-side fee × 2 + slippage × 2 (entry + exit) + gas × 2
    slip = slippage_estimate(t["entry_price"], t["liquidity"])
    spread_cost = shares * slip * 2  # both sides

    fee_type = t["fee_type"]
    f_entry = fee_for_fill(t["entry_price"], shares, fee_type)
    f_exit = fee_for_fill(t["exit_price"], shares, fee_type)
    gas = GAS_PER_FILL * 2

    net = gross - spread_cost - f_entry - f_exit - gas
    return net, gross, spread_cost + f_entry + f_exit + gas

# Compare old vs new cost model
print("\n=== Cost model comparison ===")
old_costs = []
new_costs = []
new_pnls = []
gross_pnls = []
for t in filtered:
    # Old: 2¢ spread + 2.5% fee
    if t["direction"] == 1:
        shares = STAKE / t["entry_price"]
    else:
        shares = STAKE / (1 - t["entry_price"])
    gross = shares * t["ret_per_share"]
    old_spread = shares * 0.02
    old_fee = shares * (0.025 * t["entry_price"] * (1 - t["entry_price"]) +
                        0.025 * t["exit_price"] * (1 - t["exit_price"]))
    old_total = old_spread + old_fee
    old_net = gross - old_total
    old_costs.append(old_total)

    net, _, new_total = realistic_pnl(t)
    new_costs.append(new_total)
    new_pnls.append(net)
    gross_pnls.append(gross)

old_costs = np.array(old_costs)
new_costs = np.array(new_costs)
new_pnls = np.array(new_pnls)
gross_pnls = np.array(gross_pnls)

print(f"  Avg gross PnL/trade: ${gross_pnls.mean():+.2f}")
print(f"  Avg OLD cost: ${old_costs.mean():.2f}")
print(f"  Avg NEW cost: ${new_costs.mean():.2f}  (+${new_costs.mean()-old_costs.mean():.2f} extra)")
print(f"  Avg NEW net PnL: ${new_pnls.mean():+.2f}")
print(f"  Win rate (NEW costs): {(new_pnls > 0).mean()*100:.1f}%")
print(f"  Sharpe (NEW): {new_pnls.mean()/new_pnls.std():.3f}")

# ─────────────────────────────────────────────────────────────────
# Step 2: Apply MAX_OPEN_POSITIONS filter
# ─────────────────────────────────────────────────────────────────
print("\n=== Max positions filter ===")
# Sort trades by entry_ts, simulate sequential opening
sorted_trades = sorted(filtered, key=lambda t: t["entry_ts"])
open_positions = []  # list of (exit_ts, trade)
accepted_trades = []
skipped_count = 0
for t in sorted_trades:
    # Close any positions that ended before this entry
    open_positions = [(ex, tr) for ex, tr in open_positions if ex > t["entry_ts"]]
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        skipped_count += 1
        continue
    accepted_trades.append(t)
    open_positions.append((t["exit_ts"], t))

print(f"  Total filtered trades: {len(filtered)}")
print(f"  Accepted (within 10-position limit): {len(accepted_trades)}")
print(f"  Skipped (would exceed 10): {skipped_count}")

acc_pnls = np.array([realistic_pnl(t)[0] for t in accepted_trades])
if len(acc_pnls) > 0:
    print(f"  Net mean (accepted only): ${acc_pnls.mean():+.2f}")
    print(f"  Total over 30 days: ${acc_pnls.sum():+.2f}")
    print(f"  Win rate: {(acc_pnls > 0).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────
# Step 3: CLV calculation
# ─────────────────────────────────────────────────────────────────
print("\n=== CLV (Closing Line Value) — fetching 1-hour-later prices ===")
print("This is the GOLD metric — positive CLV means strategy has real edge.\n")

def get_yes_token(m):
    tids = m.get("clobTokenIds")
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

# We'll sample ~150 trades to keep API calls manageable
import random
random.seed(42)
clv_sample = random.sample(accepted_trades, min(150, len(accepted_trades)))

clv_values = []
fetched = 0
for i, t in enumerate(clv_sample):
    if i % 20 == 0:
        print(f"  Fetching CLV {i}/{len(clv_sample)}...", flush=True)
    m = markets.get(t["market_id"])
    if not m: continue
    tok = get_yes_token(m)
    if not tok: continue
    r = requests.get(f"{CLOB}/prices-history",
                     params={"market": tok, "fidelity": 60, "interval": "max"},
                     timeout=10)
    time.sleep(0.05)
    if r.status_code != 200: continue
    hist = r.json().get("history", [])
    if not hist: continue
    entry_ts = t["entry_ts"]
    target_ts = entry_ts + CLV_LOOKBACK_HOURS * 3600
    # Find bar closest to target_ts
    later_bar = None
    for h in hist:
        if h["t"] >= target_ts:
            later_bar = h
            break
    if later_bar is None: continue
    fetched += 1
    clv = t["direction"] * (later_bar["p"] - t["entry_price"])
    clv_values.append(clv)

clv_values = np.array(clv_values)
print(f"\n  CLV measurements: {len(clv_values)} / {len(clv_sample)} sampled")
if len(clv_values) > 0:
    print(f"  Mean CLV (in YES price units): {clv_values.mean()*100:+.3f}¢")
    print(f"  Median CLV:                    {np.median(clv_values)*100:+.3f}¢")
    print(f"  % positive CLV:                {(clv_values > 0).mean()*100:.1f}%")
    print(f"  Std:                           {clv_values.std()*100:.3f}¢")
    # Bootstrap
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(clv_values, len(clv_values), replace=True).mean() for _ in range(5000)])
    print(f"  Bootstrap 95% CI:              [{np.percentile(boot, 2.5)*100:+.3f}¢, {np.percentile(boot, 97.5)*100:+.3f}¢]")
    print(f"  P(mean CLV > 0):               {(boot > 0).mean()*100:.1f}%")
    print(f"  P(mean CLV > +0.5¢):           {(boot > 0.005).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────
# Step 4: Final realistic Monte Carlo
# ─────────────────────────────────────────────────────────────────
print("\n=== REALISTIC Monte Carlo: $30 stake, $720 bankroll, 10 max pos ===")
if len(acc_pnls) > 0:
    BANKROLL = 720
    # Trade frequency from accepted_trades over the test period
    test_days = (max(t["entry_ts"] for t in accepted_trades) - min(t["entry_ts"] for t in accepted_trades)) / 86400
    rate_per_day = len(accepted_trades) / test_days if test_days > 0 else 0
    print(f"  Observed trade rate (accepted): {rate_per_day:.1f}/day")

    rng = np.random.default_rng(42)
    print(f"\n  {'Days':>5} {'Trades':>7} {'p5':>9} {'p50':>9} {'p95':>9} {'P(>0)':>7}")
    for d in [7, 14, 30, 60, 90]:
        n = max(1, int(d * rate_per_day))
        sims = np.array([rng.choice(acc_pnls, n, replace=True).sum() for _ in range(10000)])
        print(f"  {d:>4}d {n:>7} ${np.percentile(sims, 5):>+7.2f} ${np.percentile(sims, 50):>+7.2f} ${np.percentile(sims, 95):>+7.2f} {100*(sims>0).mean():>5.1f}%")

print("\n=== SUMMARY ===")
print(f"  Realistic mean PnL/trade: ${acc_pnls.mean():+.2f}")
print(f"  Realistic Sharpe: {acc_pnls.mean()/acc_pnls.std():.3f}" if acc_pnls.std() > 0 else "")
if len(clv_values) > 0:
    print(f"  Realistic CLV mean: {clv_values.mean()*100:+.2f}¢ ({(clv_values > 0).mean()*100:.0f}% positive)")
    if clv_values.mean() > 0.002 and (boot > 0).mean() > 0.95:
        print(f"  ✓ Strategy shows statistically significant edge")
    elif (boot > 0).mean() > 0.80:
        print(f"  ⚠ Strategy MAY have edge but uncertainty is high")
    else:
        print(f"  ✗ Strategy edge not statistically significant")
