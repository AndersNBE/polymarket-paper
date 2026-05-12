"""
Strategy B: Mean-reversion z-score test on long-tail markets.

Methodology:
  1. Sample ~300 long-tail markets (volume24hr $50-$5000, has trading history)
  2. Pull price history (max range, hourly fidelity)
  3. For each market, slide a rolling window:
     - Compute rolling 24-bar mean and std
     - When |z-score| > 2, simulate a counter-trade
     - Hold until |z-score| < 0.5 OR 48 bars elapsed
  4. Tally hit rate, average return, distribution

This is a proper out-of-sample backtest: we use rolling-window stats that only
look BACK, never forward.
"""
import json
import math
import random
import time
from pathlib import Path
import requests
import numpy as np

CLOB = "https://clob.polymarket.com"
markets = json.loads(Path("markets.json").read_text())

def num(m, k, default=0.0):
    v = m.get(k)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

# Filter to long-tail with trading history
candidates = [
    m for m in markets
    if 50 <= num(m, "volume24hr") <= 5000
    and num(m, "liquidityNum") >= 500
    and num(m, "volume1mo") >= 1000
    and m.get("clobTokenIds")
    and m.get("enableOrderBook")
]
print(f"Long-tail candidates: {len(candidates):,}")

random.seed(42)
sample_size = 300
sample = random.sample(candidates, min(sample_size, len(candidates)))
print(f"Sampling {len(sample)} markets for backtest\n")

def get_yes_token(m):
    tids = m.get("clobTokenIds")
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

def fetch_history(token_id, fidelity=60, interval="max", retries=2):
    for i in range(retries + 1):
        try:
            r = requests.get(
                f"{CLOB}/prices-history",
                params={"market": token_id, "fidelity": fidelity, "interval": interval},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("history", [])
        except requests.RequestException:
            pass
        if i < retries:
            time.sleep(0.5)
    return None

# Backtest parameters
WINDOW = 24       # 24 hours rolling window
ENTRY_Z = 2.0     # |z| threshold to enter
EXIT_Z = 0.5      # |z| threshold to exit
MAX_HOLD = 48     # max hours to hold before forced exit
MIN_BARS = 50     # need this much history to test

trades = []   # list of {entry_price, exit_price, direction, hold_hours, market_id, z}
markets_tested = 0
markets_with_signal = 0

for i, m in enumerate(sample, 1):
    if i % 30 == 0:
        print(f"  Progress: {i}/{len(sample)}  trades_so_far={len(trades)}")
    tok = get_yes_token(m)
    if not tok:
        continue
    hist = fetch_history(tok)
    time.sleep(0.12)
    if not hist or len(hist) < MIN_BARS:
        continue
    # Convert to np arrays
    times = np.array([h["t"] for h in hist])
    prices = np.array([h["p"] for h in hist], dtype=float)
    if prices.std() < 0.001:  # flat market, skip
        continue
    markets_tested += 1
    had_signal = False

    # Walk forward
    in_pos = False
    pos_dir = 0
    pos_entry = None
    pos_entry_idx = None
    pos_entry_z = None
    for t in range(WINDOW, len(prices)):
        window = prices[t - WINDOW:t]
        mu = window.mean()
        sd = window.std()
        if sd < 0.005:  # numerical floor
            continue
        z = (prices[t] - mu) / sd

        if not in_pos:
            if abs(z) > ENTRY_Z:
                # Enter counter-trade
                in_pos = True
                pos_dir = -1 if z > 0 else 1  # mean-revert toward mu
                pos_entry = prices[t]
                pos_entry_idx = t
                pos_entry_z = z
                had_signal = True
        else:
            held = t - pos_entry_idx
            if abs(z) < EXIT_Z or held >= MAX_HOLD or t == len(prices) - 1:
                exit_price = prices[t]
                # Direction sign: long (+1) profits when price rises
                ret = pos_dir * (exit_price - pos_entry)
                trades.append({
                    "market_id": m["id"],
                    "question": m["question"][:80],
                    "entry_z": pos_entry_z,
                    "entry_price": pos_entry,
                    "exit_price": exit_price,
                    "direction": pos_dir,
                    "hold_hours": held,
                    "ret_per_share": ret,
                    "forced_exit": (held >= MAX_HOLD),
                })
                in_pos = False
    if had_signal:
        markets_with_signal += 1

print(f"\n=== Results ===")
print(f"Markets tested: {markets_tested}")
print(f"Markets with at least one signal: {markets_with_signal}")
print(f"Total trades simulated: {len(trades)}")

if trades:
    rets = np.array([t["ret_per_share"] for t in trades])
    hits = sum(1 for r in rets if r > 0)
    print(f"\nWin rate: {hits}/{len(rets)} = {100*hits/len(rets):.1f}%")
    print(f"Average return per share (per $1 staked): ${rets.mean():.4f}")
    print(f"Median return per share: ${float(np.median(rets)):.4f}")
    print(f"Std of returns: ${rets.std():.4f}")
    print(f"Sharpe-ish (mean/std): {rets.mean()/rets.std():.3f}")
    print(f"Total cumulative return (sum of all trades, $1 per trade): ${rets.sum():.4f}")
    print(f"Average hold time (hours): {np.mean([t['hold_hours'] for t in trades]):.1f}")
    forced = sum(1 for t in trades if t["forced_exit"])
    print(f"Forced exits (max-hold reached): {forced}/{len(trades)} = {100*forced/len(trades):.1f}%")

    # Realistic-cost adjustment: assume 4% taker fee per round-trip + spread cost
    # Polymarket peak fee is ~4-5% per side at p=0.5
    # Spread costs vary; use 2 cents median on long-tail markets
    SPREAD_COST = 0.02
    FEE_RATE = 0.04  # round-trip approx
    # Net per-share = ret - spread - fee_rate * (entry + exit) * 0.5
    nets = []
    for t in trades:
        avg_p = 0.5 * (t["entry_price"] + t["exit_price"])
        # taker on both sides ≈ fee_rate*p*(1-p) curve; approximate as 0.025*p*(1-p)
        fee = 0.025 * t["entry_price"] * (1 - t["entry_price"]) + 0.025 * t["exit_price"] * (1 - t["exit_price"])
        net = t["ret_per_share"] - SPREAD_COST - fee
        nets.append(net)
    nets = np.array(nets)
    print(f"\n--- After spread (2¢) and approx fees ---")
    net_hits = sum(1 for n in nets if n > 0)
    print(f"Win rate after costs: {net_hits}/{len(nets)} = {100*net_hits/len(nets):.1f}%")
    print(f"Avg return after costs: ${nets.mean():.4f}")
    print(f"Total net return: ${nets.sum():.4f}")

    # Distribution
    print(f"\nReturn percentiles (gross):")
    for p in [5, 25, 50, 75, 95]:
        print(f"  p{p}: ${float(np.percentile(rets, p)):.4f}")

Path("results_B.json").write_text(json.dumps(trades, indent=2, default=str))
print("\nSaved trades to results_B.json")
