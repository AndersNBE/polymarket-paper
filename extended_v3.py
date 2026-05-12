#!/usr/bin/env python3
"""
extended_v3.py — Extended backtest on LOW-VOLUME resolved markets ($5k-$50k).

This matches our live paper trader's universe ($1k-$50k liquidity), which is
the actual population we trade. Previous extended_v2 tested $100k+ markets
(big news events) which behave differently than long-tail retail markets.

Same trade-reconstruction approach: pull all trades from data-api/trades,
forward-fill into hourly bars, run z-score strategy.
"""
import json
import time
import sys
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np

DATA_API = "https://data-api.polymarket.com"

# Files (separate from v2)
RESOLVED_MARKETS = Path("resolved_markets.json")
TRADES_FILE = Path("extended_trades_v3.jsonl")
PROGRESS_FILE = Path("extended_progress_v3.json")
ERRORS_FILE = Path("extended_errors_v3.jsonl")

# Strategy params (same)
ROLLING_WINDOW = 24
ENTRY_Z = 2.0
EXIT_Z = 0.5
MAX_HOLD = 48

# Universe filter: LOWER volume range matching live universe
MIN_VOLUME = 5_000     # $5k+
MAX_VOLUME = 50_000    # capped at $50k
MAX_UNIVERSE = 1500    # bigger sample since signals are sparser
MAX_WORKERS = 10       # bumped up for I/O
API_DELAY = 0.02

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def num(d, k, default=0.0):
    v = d.get(k)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def fetch_trades(condition_id, max_pages=200):
    out = []
    offset = 0
    PAGE = 500
    while offset < max_pages * PAGE:
        try:
            r = requests.get(
                f"{DATA_API}/trades",
                params={"market": condition_id, "limit": PAGE, "offset": offset},
                timeout=15,
            )
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            out.extend(batch)
            if len(batch) < PAGE:
                break
            offset += PAGE
        except requests.RequestException:
            break
    return out

def reconstruct_bars(trades):
    if not trades:
        return []
    bars_by_hour = {}
    for tr in trades:
        try:
            ts = int(tr["timestamp"])
            price = float(tr["price"])
            oi = int(tr.get("outcomeIndex", 0))
        except (KeyError, ValueError, TypeError):
            continue
        yes_price = price if oi == 0 else 1.0 - price
        hour = (ts // 3600) * 3600
        if hour not in bars_by_hour or ts > bars_by_hour[hour][0]:
            bars_by_hour[hour] = (ts, yes_price)
    if not bars_by_hour:
        return []
    min_h = min(bars_by_hour.keys())
    max_h = max(bars_by_hour.keys())
    out = []
    last_p = None
    for h in range(min_h, max_h + 3600, 3600):
        if h in bars_by_hour:
            last_p = bars_by_hour[h][1]
        if last_p is not None:
            out.append((h, last_p))
    return out

def process_market(m):
    cond = m.get("conditionId")
    if not cond:
        return []
    raw = fetch_trades(cond)
    if not raw:
        return []
    bars = reconstruct_bars(raw)
    if len(bars) < ROLLING_WINDOW + 5:
        return []
    times = np.array([b[0] for b in bars])
    prices = np.array([b[1] for b in bars], dtype=float)
    if prices.std() < 0.001:
        return []
    end_ts = None
    try:
        end_str = m.get("endDate")
        if end_str:
            end_ts = datetime.datetime.fromisoformat(end_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        pass
    out_trades = []
    in_pos = False
    pos = None
    for t in range(ROLLING_WINDOW, len(prices)):
        window = prices[t - ROLLING_WINDOW:t]
        mu = window.mean()
        sd = window.std()
        if sd < 0.005:
            continue
        z = (prices[t] - mu) / sd
        if not in_pos:
            if abs(z) >= ENTRY_Z:
                in_pos = True
                pos = {
                    "market_id": m["id"],
                    "fee_type": m.get("feeType"),
                    "entry_z": float(z),
                    "entry_price": float(prices[t]),
                    "entry_idx": int(t),
                    "entry_ts": int(times[t]),
                    "direction": -1 if z > 0 else 1,
                }
        else:
            held = t - pos["entry_idx"]
            if abs(z) < EXIT_Z or held >= MAX_HOLD or t == len(prices) - 1:
                exit_price = float(prices[t])
                ret = pos["direction"] * (exit_price - pos["entry_price"])
                trade = {
                    **pos,
                    "exit_price": exit_price,
                    "exit_ts": int(times[t]),
                    "hold_hours": int(held),
                    "ret_per_share": float(ret),
                    "forced_exit": held >= MAX_HOLD,
                    "end_ts": end_ts,
                    "lifetime_volume": num(m, "volumeNum"),
                    "history_bars": len(prices),
                    "n_raw_trades": len(raw),
                }
                if end_ts:
                    trade["dte_days_at_entry"] = (end_ts - trade["entry_ts"]) / 86400
                out_trades.append(trade)
                in_pos = False
    return out_trades

def load_progress():
    if PROGRESS_FILE.exists():
        return set(str(x) for x in json.loads(PROGRESS_FILE.read_text()))
    return set()

def save_progress(done):
    PROGRESS_FILE.write_text(json.dumps(sorted(done)))

def main():
    all_markets = json.loads(RESOLVED_MARKETS.read_text())
    filt = [m for m in all_markets
            if MIN_VOLUME <= num(m, "volumeNum") <= MAX_VOLUME
            and m.get("conditionId")]
    # Random shuffle so we don't bias by volume order
    import random
    random.seed(42)
    random.shuffle(filt)
    filt = filt[:MAX_UNIVERSE]
    print(f"[{now()}] Low-volume resolved universe (${MIN_VOLUME:,}-${MAX_VOLUME:,}): {len(filt):,}", flush=True)

    done = load_progress()
    todo = [m for m in filt if str(m["id"]) not in done]
    print(f"[{now()}] To process: {len(todo):,} (skipping {len(done):,})\n", flush=True)
    if not todo:
        print("Done already")
        return

    total_trades = 0
    total_filtered = 0
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            for line in f:
                if line.strip():
                    total_trades += 1
                    try:
                        tr = json.loads(line)
                        if abs(tr["entry_z"]) >= 5.0 and 0.10 <= tr["entry_price"] <= 0.90:
                            total_filtered += 1
                    except (json.JSONDecodeError, KeyError):
                        pass

    print(f"[{now()}] Spawning {MAX_WORKERS} parallel workers...\n", flush=True)
    start = time.time()
    errors = 0

    def worker(m):
        try:
            return ("ok", m, process_market(m))
        except Exception as e:
            return ("err", m, str(e))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, m): m for m in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                status, m, payload = fut.result()
                if status == "ok":
                    trades = payload
                    if trades:
                        with open(TRADES_FILE, "a") as f:
                            for tr in trades:
                                f.write(json.dumps(tr, default=str) + "\n")
                        total_trades += len(trades)
                        for tr in trades:
                            if abs(tr["entry_z"]) >= 5.0 and 0.10 <= tr["entry_price"] <= 0.90:
                                total_filtered += 1
                    done.add(str(m["id"]))
                else:
                    errors += 1
                    with open(ERRORS_FILE, "a") as f:
                        f.write(json.dumps({"market_id": m["id"], "error": payload, "ts": now()}) + "\n")
            except Exception:
                errors += 1
            if i % 50 == 0 or i == len(todo):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (len(todo) - i) / rate / 60 if rate > 0 else 0
                pct = 100 * i / len(todo)
                print(
                    f"[{now()}]  {i:>4}/{len(todo):>4} ({pct:>5.1f}%)  "
                    f"rate={rate:>5.2f}/s  trades={total_trades:>6,}  "
                    f"filtered={total_filtered:>5,}  errors={errors:>3}  "
                    f"eta={eta_min:>5.1f}min",
                    flush=True,
                )
                save_progress(done)
            time.sleep(API_DELAY)

    save_progress(done)
    print(f"\n[{now()}] ✓ DONE  Total trades: {total_trades:,}  Filtered: {total_filtered:,}", flush=True)

if __name__ == "__main__":
    main()
