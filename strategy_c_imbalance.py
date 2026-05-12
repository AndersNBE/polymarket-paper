"""
Strategy C: Trade-flow imbalance signal test.

Hypothesis: when recent trades are heavily one-sided (e.g., 80%+ taker BUYs),
this predicts subsequent price movement. Two regimes to test:
  - Momentum: imbalance predicts CONTINUED move in that direction
  - Mean-reversion: imbalance predicts REVERSAL (panic/overshoot)

Methodology:
  1. Sample markets with sufficient activity
  2. For each market, pull recent trades (up to 1000)
  3. In sliding 1-hour windows, compute buy-side dollar share
  4. Pair each window with the price 1, 4, 12 hours later (from price-history)
  5. Bin by imbalance level and compute average forward return
"""
import json
import time
import random
import sys
from pathlib import Path
import requests
import numpy as np

CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
markets_list = json.loads(Path("markets.json").read_text())
markets = {m["id"]: m for m in markets_list}

def num(m, k, default=0.0):
    v = m.get(k)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

# Markets with enough trade activity
candidates = [
    m for m in markets_list
    if num(m, "volume24hr") >= 1000
    and num(m, "liquidityNum") >= 2000
    and num(m, "volume1mo") >= 5000
    and m.get("conditionId")
]
print(f"Candidates with >=$1k v24hr: {len(candidates):,}")

random.seed(7)
sample = random.sample(candidates, min(200, len(candidates)))

def get_yes_token(m):
    tids = m.get("clobTokenIds")
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

def fetch_trades(condition_id, retries=2):
    """Fetch recent trades (up to 1000) for a condition."""
    for i in range(retries + 1):
        try:
            r = requests.get(
                f"{DATA}/trades",
                params={"market": condition_id, "limit": 1000, "offset": 0},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        if i < retries:
            time.sleep(0.4)
    return None

def fetch_history(token_id, retries=2):
    for i in range(retries + 1):
        try:
            r = requests.get(
                f"{CLOB}/prices-history",
                params={"market": token_id, "fidelity": 60, "interval": "max"},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json().get("history", [])
        except requests.RequestException:
            pass
        if i < retries:
            time.sleep(0.4)
    return None

# Forward-return horizons (hours)
HORIZONS = [1, 4, 12]
WINDOW_SECS = 3600  # 1-hour windows

observations = []
markets_done = 0
last_print = 0
for i, m in enumerate(sample, 1):
    if i - last_print >= 25:
        sys.stdout.write(f"  {i}/{len(sample)}  obs={len(observations)}\n")
        sys.stdout.flush()
        last_print = i
    cond = m.get("conditionId")
    tok = get_yes_token(m)
    if not cond or not tok:
        continue
    trades = fetch_trades(cond)
    time.sleep(0.10)
    if not trades or len(trades) < 30:
        continue
    hist = fetch_history(tok)
    time.sleep(0.10)
    if not hist or len(hist) < 30:
        continue
    markets_done += 1

    # Build trade arrays - each trade is on the YES side or NO side
    # We want "buy YES pressure" - this means buying YES OR selling NO
    # Convention: trade with outcomeIndex==0 is on YES side; ==1 is on NO side
    # side="BUY" + outcomeIndex==0 → buy YES → upward pressure
    # side="SELL" + outcomeIndex==0 → sell YES → downward
    # side="BUY"  + outcomeIndex==1 → buy NO  → downward (equiv to selling YES)
    # side="SELL" + outcomeIndex==1 → sell NO → upward
    tr_ts = []
    tr_signed_dollar = []
    for t in trades:
        try:
            ts = int(t["timestamp"])
            side = t["side"]
            size = float(t["size"])
            price = float(t["price"])
            oi = int(t.get("outcomeIndex", 0))
        except (KeyError, ValueError, TypeError):
            continue
        # Compute YES-pressure sign
        if oi == 0:
            sign = 1 if side == "BUY" else -1
        else:  # NO side
            sign = -1 if side == "BUY" else 1
        # Dollar value
        dollars = size * price if oi == 0 else size * (1 - price)
        tr_ts.append(ts)
        tr_signed_dollar.append(sign * dollars)

    if len(tr_ts) < 30:
        continue
    tr_ts = np.array(tr_ts)
    tr_dollars = np.array(tr_signed_dollar)
    # Total absolute dollar volume per trade
    tr_abs = np.abs(tr_dollars)

    # Build price-history lookup
    h_ts = np.array([h["t"] for h in hist])
    h_p = np.array([h["p"] for h in hist], dtype=float)

    # Slide hourly windows
    if tr_ts.max() <= tr_ts.min():
        continue
    t_start = int(tr_ts.min())
    t_end = int(tr_ts.max())
    # Snap to hour boundaries
    step = WINDOW_SECS
    for win_end in range(t_start + step, t_end - 12*3600, step):
        win_start = win_end - step
        mask = (tr_ts > win_start) & (tr_ts <= win_end)
        if mask.sum() < 5:
            continue
        win_dollars = tr_dollars[mask]
        win_abs = tr_abs[mask]
        total_abs = win_abs.sum()
        if total_abs < 50:  # require some volume
            continue
        # YES-buy share = (signed sum / total) — ranges -1 to +1
        imbalance = win_dollars.sum() / total_abs

        # Find spot price at win_end
        idx = np.searchsorted(h_ts, win_end)
        if idx >= len(h_p) or idx == 0:
            continue
        # take closest bar
        p_now = h_p[idx]

        # Get forward returns
        fwd_rets = {}
        for hours in HORIZONS:
            tgt_ts = win_end + hours * 3600
            j = np.searchsorted(h_ts, tgt_ts)
            if j >= len(h_p):
                continue
            fwd_rets[hours] = float(h_p[j] - p_now)

        if not fwd_rets:
            continue

        observations.append({
            "market_id": m["id"],
            "ts": win_end,
            "imbalance": float(imbalance),
            "volume": float(total_abs),
            "n_trades": int(mask.sum()),
            "price": float(p_now),
            **{f"ret_{h}h": fwd_rets.get(h) for h in HORIZONS},
        })

print(f"\nMarkets analyzed: {markets_done}")
print(f"Total (window, forward-return) observations: {len(observations):,}")

if observations:
    Path("results_C.json").write_text(json.dumps(observations, indent=2, default=str))

    imb = np.array([o["imbalance"] for o in observations])
    print(f"\nImbalance distribution:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  p{p}: {np.percentile(imb, p):+.3f}")

    # Bin by imbalance level, compute average forward return for each horizon
    print(f"\nAverage forward returns by imbalance bin (cents/share):")
    bins = [(-1.01, -0.7), (-0.7, -0.4), (-0.4, -0.1), (-0.1, 0.1), (0.1, 0.4), (0.4, 0.7), (0.7, 1.01)]
    print(f"  {'imbalance range':>15} | {'n':>5} | {'ret_1h':>8} | {'ret_4h':>8} | {'ret_12h':>9}")
    for lo, hi in bins:
        mask = [(o["imbalance"] >= lo) and (o["imbalance"] < hi) for o in observations]
        n = sum(mask)
        if n < 5:
            continue
        sel = [o for o, k in zip(observations, mask) if k]
        r1 = np.mean([o["ret_1h"] for o in sel if o["ret_1h"] is not None]) * 100
        r4 = np.mean([o["ret_4h"] for o in sel if o["ret_4h"] is not None]) * 100
        r12 = np.mean([o["ret_12h"] for o in sel if o["ret_12h"] is not None]) * 100
        print(f"  {lo:+.2f} to {hi:+.2f} | {n:>5} | {r1:>+6.2f}¢ | {r4:>+6.2f}¢ | {r12:>+7.2f}¢")

    # Top-level: correlation
    for h in HORIZONS:
        rets = np.array([o[f"ret_{h}h"] for o in observations if o.get(f"ret_{h}h") is not None])
        imbs = np.array([o["imbalance"] for o in observations if o.get(f"ret_{h}h") is not None])
        if len(rets) > 30:
            corr = np.corrcoef(imbs, rets)[0, 1]
            print(f"\n{h}h horizon: corr(imbalance, return) = {corr:+.3f}  (n={len(rets)})")
