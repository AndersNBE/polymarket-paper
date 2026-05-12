#!/usr/bin/env python3
"""
extended_backtest.py — Long-horizon backtest on RESOLVED Polymarket markets.

Pulls resolved markets (closed=true) that ended within the last 12 months,
fetches their FULL price history (often multi-month per market), runs the
same z-score strategy used by paper_trader.

Output:
  - extended_trades.jsonl  — append-only log of all trades found
  - extended_progress.json — set of processed market IDs (for resume)
  - extended_errors.jsonl  — markets that errored
  - extended_backtest.log  — stdout log

Live progress:
  Prints every 50 markets with: progress %, rate, trades found, ETA,
  and running filtered (z>=5 + filters) trade count.

Resume after interrupt:
  Just re-run the script; it skips completed markets.

Run:
  python3 -u extended_backtest.py 2>&1 | tee extended_backtest.log
"""
import json
import time
import sys
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
import numpy as np

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"

# Parallelism: 8 concurrent HTTP requests
MAX_WORKERS = 8

# Files
RESOLVED_MARKETS = Path("resolved_markets.json")
TRADES_FILE      = Path("extended_trades.jsonl")
PROGRESS_FILE    = Path("extended_progress.json")
ERRORS_FILE      = Path("extended_errors.jsonl")

# Strategy params (same as paper_trader)
ROLLING_WINDOW = 24
ENTRY_Z = 2.0          # capture all; we filter to >=5 in analysis
EXIT_Z  = 0.5
MAX_HOLD = 48

# Universe filter for resolved markets
MIN_LIFETIME_VOLUME = 5000
MAX_AGE_DAYS = 365     # ended within last 12 months
MAX_UNIVERSE_SIZE = 20000  # cap to keep total time reasonable
API_DELAY = 0.02       # small per-worker delay; aggregate ~30 req/s with 8 workers

# Filter (matches paper_trader exact rules) used for inline progress count
def is_filtered_signal(trade):
    if abs(trade["entry_z"]) < 5.0: return False
    if not (0.10 <= trade["entry_price"] <= 0.90): return False
    return True  # dte filter is harder to compute inline w/o end_ts; we don't filter on it here

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def num(d, k, default=0.0):
    v = d.get(k)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def fetch_json(url, params=None, retries=3, timeout=15):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(0.5 * (attempt + 1))
    return None

# ──────────────────────────────────────────────────────────────────────
# Phase 1: build resolved-markets universe
# ──────────────────────────────────────────────────────────────────────
def pull_resolved_markets():
    if RESOLVED_MARKETS.exists():
        existing = json.loads(RESOLVED_MARKETS.read_text())
        print(f"[{now()}] Loaded {len(existing):,} cached resolved markets from {RESOLVED_MARKETS}", flush=True)
        return existing
    print(f"[{now()}] Pulling resolved-markets list (closed=true)...", flush=True)
    out = []
    offset = 0
    PAGE = 500
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(days=MAX_AGE_DAYS)).timestamp()
    while True:
        batch = fetch_json(
            f"{GAMMA}/markets",
            params={
                "limit": PAGE,
                "offset": offset,
                "closed": "true",
                "archived": "false",
            },
        )
        if not batch:
            break
        kept = 0
        for m in batch:
            if num(m, "volumeNum") < MIN_LIFETIME_VOLUME:
                continue
            # End date within MAX_AGE_DAYS
            end = m.get("endDate") or m.get("endDateIso")
            end_ts = None
            if end:
                try:
                    end_ts = datetime.datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp()
                except (ValueError, AttributeError):
                    end_ts = None
            if end_ts and end_ts < cutoff:
                continue
            if not m.get("clobTokenIds"):
                continue
            out.append(m)
            kept += 1
        print(f"  offset={offset:>6}  page-kept={kept:>4}  total-kept={len(out):>6,}", flush=True)
        if len(batch) < PAGE or len(out) >= MAX_UNIVERSE_SIZE:
            break
        offset += PAGE
        time.sleep(0.25)
    if len(out) > MAX_UNIVERSE_SIZE:
        # Keep highest-volume ones
        out.sort(key=lambda m: -num(m, "volumeNum"))
        out = out[:MAX_UNIVERSE_SIZE]
    RESOLVED_MARKETS.write_text(json.dumps(out))
    print(f"[{now()}] Saved {len(out):,} resolved markets to {RESOLVED_MARKETS}", flush=True)
    return out

# ──────────────────────────────────────────────────────────────────────
# Phase 2: process each market
# ──────────────────────────────────────────────────────────────────────
def get_yes_token(m):
    tids = m.get("clobTokenIds")
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

def fetch_history(token_id):
    data = fetch_json(
        f"{CLOB}/prices-history",
        params={"market": token_id, "fidelity": 60, "interval": "max"},
    )
    if data:
        return data.get("history", [])
    return None

def market_end_ts(m):
    end = m.get("endDate") or m.get("endDateIso")
    if not end:
        return None
    try:
        return datetime.datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None

def process_market(m):
    tok = get_yes_token(m)
    if not tok:
        return []
    hist = fetch_history(tok)
    if not hist or len(hist) < ROLLING_WINDOW + 5:
        return []
    times = np.array([h["t"] for h in hist])
    prices = np.array([h["p"] for h in hist], dtype=float)
    if prices.std() < 0.001:
        return []
    end_ts = market_end_ts(m)
    trades = []
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
                }
                # Compute days-to-end at entry (only valid for filtering)
                if end_ts:
                    trade["dte_days_at_entry"] = (end_ts - trade["entry_ts"]) / 86400
                trades.append(trade)
                in_pos = False
    return trades

# ──────────────────────────────────────────────────────────────────────
# Progress persistence
# ──────────────────────────────────────────────────────────────────────
def load_progress():
    if PROGRESS_FILE.exists():
        return set(str(x) for x in json.loads(PROGRESS_FILE.read_text()))
    return set()

def save_progress(done):
    PROGRESS_FILE.write_text(json.dumps(sorted(done)))

# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────
def main():
    markets = pull_resolved_markets()
    done = load_progress()
    todo = [m for m in markets if str(m["id"]) not in done]

    print(f"\n[{now()}] Universe: {len(markets):,} markets  (already done: {len(done):,}, to do: {len(todo):,})\n", flush=True)
    if not todo:
        print("All done already!", flush=True)
        return

    start = time.time()
    total_trades = 0
    total_filtered = 0   # z>=5 + price-filter
    errors = 0

    # Count trades in existing file (for accurate resume tally)
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            for line in f:
                if line.strip():
                    total_trades += 1
                    try:
                        trade = json.loads(line)
                        if is_filtered_signal(trade):
                            total_filtered += 1
                    except json.JSONDecodeError:
                        pass
        print(f"[{now()}] Existing trades in log: {total_trades:,}  (filtered: {total_filtered:,})\n", flush=True)

    # Wrapper that adds the API delay per worker
    def worker(m):
        try:
            trades = process_market(m)
            time.sleep(API_DELAY)  # gentle rate-limit per worker
            return ("ok", m, trades)
        except Exception as e:
            return ("err", m, str(e))

    # Parallel execution
    print(f"[{now()}] Spawning {MAX_WORKERS} parallel workers...\n", flush=True)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(worker, m): m for m in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                status, m, payload = fut.result()
                if status == "ok":
                    trades = payload
                    if trades:
                        # Single-threaded file append (main thread does I/O)
                        with open(TRADES_FILE, "a") as f:
                            for tr in trades:
                                f.write(json.dumps(tr, default=str) + "\n")
                        total_trades += len(trades)
                        for tr in trades:
                            if is_filtered_signal(tr):
                                total_filtered += 1
                    done.add(str(m["id"]))
                else:
                    errors += 1
                    with open(ERRORS_FILE, "a") as f:
                        f.write(json.dumps({"market_id": m["id"], "error": payload, "ts": now()}) + "\n")
            except Exception as e:
                errors += 1

            # Periodic progress + save
            if i % 100 == 0 or i == len(todo):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                remaining = len(todo) - i
                eta_min = (remaining / rate / 60) if rate > 0 else 0
                pct = 100 * i / len(todo)
                print(
                    f"[{now()}]  {i:>5}/{len(todo):>5} ({pct:>5.1f}%)  "
                    f"rate={rate:>5.1f}/s  trades={total_trades:>6,}  "
                    f"filtered(z>=5)={total_filtered:>5,}  errors={errors:>3}  "
                    f"eta={eta_min:>5.1f}min",
                    flush=True,
                )
                save_progress(done)

    save_progress(done)
    print(f"\n[{now()}] ✓ DONE", flush=True)
    print(f"  Total trades written: {total_trades:,}", flush=True)
    print(f"  Filtered trades (z>=5 + price): {total_filtered:,}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  Output: {TRADES_FILE}", flush=True)

if __name__ == "__main__":
    main()
