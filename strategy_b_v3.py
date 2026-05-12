"""
Strategy B v3: Larger sample (800 markets), per-market spread measurement,
                and more honest net-PnL accounting.

Findings to test:
  - Does the z>=5 edge persist at larger scale?
  - Is the edge concentrated in particular liquidity tiers?
  - What's the temporal distribution of signals (are recent trades still profitable)?
"""
import json
import math
import random
import time
import sys
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

candidates = [
    m for m in markets
    if 50 <= num(m, "volume24hr") <= 10000
    and num(m, "liquidityNum") >= 1000   # bumped up to tighter-spread markets
    and num(m, "volume1mo") >= 2000      # need history
    and m.get("clobTokenIds")
    and m.get("enableOrderBook")
]
print(f"Long-tail candidates (liq>=$1k, v24hr in $50-$10k): {len(candidates):,}")

random.seed(42)
sample = random.sample(candidates, min(800, len(candidates)))
print(f"Sampling {len(sample)} markets for backtest")

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
            time.sleep(0.4)
    return None

WINDOW = 24
EXIT_Z = 0.5
MAX_HOLD = 48
MIN_BARS = 50

# Track trades per market for later analysis
all_trades = []
markets_tested = 0
last_print = 0
for i, m in enumerate(sample, 1):
    if i - last_print >= 50:
        sys.stdout.write(f"  Progress: {i}/{len(sample)}  trades_so_far={len(all_trades)}\n")
        sys.stdout.flush()
        last_print = i
    tok = get_yes_token(m)
    if not tok:
        continue
    hist = fetch_history(tok)
    time.sleep(0.08)
    if not hist or len(hist) < MIN_BARS:
        continue
    times = np.array([h["t"] for h in hist])
    prices = np.array([h["p"] for h in hist], dtype=float)
    if prices.std() < 0.001:
        continue
    markets_tested += 1
    in_pos = False
    pos_dir = 0
    pos_entry = None
    pos_entry_idx = None
    pos_entry_z = None
    for t in range(WINDOW, len(prices)):
        window = prices[t - WINDOW:t]
        mu = window.mean()
        sd = window.std()
        if sd < 0.005:
            continue
        z = (prices[t] - mu) / sd

        if not in_pos:
            # Multiple z entry thresholds: enter for ANY |z|>=2, record entry z
            if abs(z) >= 2.0:
                in_pos = True
                pos_dir = -1 if z > 0 else 1
                pos_entry = prices[t]
                pos_entry_idx = t
                pos_entry_z = z
        else:
            held = t - pos_entry_idx
            # Recompute z at exit time
            if abs(z) < EXIT_Z or held >= MAX_HOLD or t == len(prices) - 1:
                exit_price = prices[t]
                ret = pos_dir * (exit_price - pos_entry)
                all_trades.append({
                    "market_id": m["id"],
                    "liquidity": float(m.get("liquidityNum") or 0),
                    "v24": float(m.get("volume24hr") or 0),
                    "v1mo": float(m.get("volume1mo") or 0),
                    "entry_z": pos_entry_z,
                    "entry_price": pos_entry,
                    "exit_price": exit_price,
                    "direction": pos_dir,
                    "hold_hours": held,
                    "ret_per_share": ret,
                    "forced_exit": (held >= MAX_HOLD),
                    "entry_ts": int(times[pos_entry_idx]),
                    "exit_ts": int(times[t]),
                })
                in_pos = False

print(f"\n=== Total ===")
print(f"Markets tested: {markets_tested}")
print(f"Trades simulated: {len(all_trades)}")

if not all_trades:
    sys.exit(0)

Path("results_B_v3.json").write_text(json.dumps(all_trades, indent=2, default=str))

# Analysis
rets = np.array([t["ret_per_share"] for t in all_trades])
entries = np.array([t["entry_price"] for t in all_trades])
exits = np.array([t["exit_price"] for t in all_trades])
zs = np.array([abs(t["entry_z"]) for t in all_trades])
liqs = np.array([t["liquidity"] for t in all_trades])
entry_ts = np.array([t["entry_ts"] for t in all_trades])

def fee_per_trade(p, fee_rate=0.025):
    return 2 * fee_rate * p * (1 - p)

print(f"\nGross stats:")
print(f"  Win rate: {(rets > 0).mean()*100:.1f}%")
print(f"  Avg return: {rets.mean()*100:+.2f}¢ per share")
print(f"  Median return: {np.median(rets)*100:+.2f}¢")
print(f"  Std: {rets.std()*100:.2f}¢")

# Sweep across (z_min, spread)
print(f"\nNet PnL (cents per $1 trade) across z_min × spread:")
print(f"  {'z>=':>5} | {'n':>5} | {'spread=0.5':>10} | {'spread=1':>9} | {'spread=2':>9} | {'spread=3':>9} | {'spread=5':>9}")
for z_min in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
    mask = zs >= z_min
    if mask.sum() < 5:
        continue
    sel_rets = rets[mask]
    sel_entries = entries[mask]
    sel_exits = exits[mask]
    fees = np.array([fee_per_trade(p) for p in 0.5*(sel_entries+sel_exits)])
    row = [f"  {z_min:>4.1f}", f"{mask.sum():>5d}"]
    for sp in [0.005, 0.01, 0.02, 0.03, 0.05]:
        net = (sel_rets - sp - fees) * 100
        row.append(f"{net.mean():>+8.2f}")
    print(" | ".join(row))

# Liquidity-tier analysis
print(f"\nBy liquidity tier (z>=4):")
for lo, hi in [(1000, 3000), (3000, 10000), (10000, 30000), (30000, 1e9)]:
    mask = (zs >= 4.0) & (liqs >= lo) & (liqs < hi)
    if mask.sum() < 3:
        continue
    sel = rets[mask]
    fees = np.array([fee_per_trade(p) for p in 0.5*(entries[mask]+exits[mask])])
    net1c = sel - 0.01 - fees
    net2c = sel - 0.02 - fees
    print(f"  ${lo:>5}-${hi:<7}: n={mask.sum():4d}  gross_avg={sel.mean()*100:+.2f}¢  net@1¢={net1c.mean()*100:+.2f}¢  net@2¢={net2c.mean()*100:+.2f}¢  win%={100*(sel>0).mean():.1f}")

# Temporal robustness: is the signal still there in recent months?
print(f"\nTemporal stability (z>=4):")
import datetime
mask = zs >= 4.0
if mask.sum() > 30:
    sel_ts = entry_ts[mask]
    sel_ret = rets[mask]
    sel_entries = entries[mask]
    sel_exits = exits[mask]
    sel_fees = np.array([fee_per_trade(p) for p in 0.5*(sel_entries+sel_exits)])
    # 4 quartiles by time
    q = np.percentile(sel_ts, [0, 25, 50, 75, 100])
    for i in range(4):
        m2 = (sel_ts >= q[i]) & (sel_ts < q[i+1] + (1 if i==3 else 0))
        if m2.sum() < 3:
            continue
        period_start = datetime.datetime.fromtimestamp(q[i]).strftime("%Y-%m-%d")
        period_end = datetime.datetime.fromtimestamp(q[i+1]).strftime("%Y-%m-%d")
        net1c = (sel_ret[m2] - 0.01 - sel_fees[m2]) * 100
        print(f"  {period_start} → {period_end}: n={m2.sum():3d}  gross_avg={sel_ret[m2].mean()*100:+.2f}¢  net@1¢={net1c.mean():+.2f}¢")
