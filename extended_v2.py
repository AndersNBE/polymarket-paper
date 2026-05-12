"""
extended_v2.py — REAL extended backtest using trade reconstruction.

The CLOB prices-history endpoint returns 0 bars for resolved markets at hourly
fidelity. But the trades endpoint at data-api.polymarket.com returns ALL
trades for ALL markets (active or resolved). We can reconstruct hourly OHLC
bars from those trades.

Strategy:
  1. Pull resolved markets (high volume, last 12 months)
  2. For each, paginate through all trades (data-api/trades)
  3. Bucket trades into hourly bins → reconstruct close prices
  4. Run same z-score backtest as before
  5. Compare results

Parallelized with ThreadPoolExecutor.
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

# Files
RESOLVED_MARKETS = Path("resolved_markets.json")
TRADES_FILE_V2 = Path("extended_trades_v2.jsonl")
PROGRESS_FILE_V2 = Path("extended_progress_v2.json")
ERRORS_FILE_V2 = Path("extended_errors_v2.jsonl")

# Strategy params (same as paper_trader)
ROLLING_WINDOW = 24
ENTRY_Z = 2.0
EXIT_Z = 0.5
MAX_HOLD = 48

# Universe filter — sample HIGH-VOLUME resolved markets first
MIN_LIFETIME_VOLUME = 100_000   # $100k+ (smaller sample, better quality)
MAX_UNIVERSE_SIZE = 800
MAX_WORKERS = 8
API_DELAY = 0.03

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def num(d, k, default=0.0):
    v = d.get(k)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default

def fetch_trades_all(condition_id, max_pages=200):
    """Pull ALL trades for a market, paginated. Returns list of dicts with ts, price, side, size."""
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

def reconstruct_hourly_bars(trades, yes_token):
    """Convert raw trades into hourly close-price bars on YES side.
       Forward-fills missing hours with last known price (matches the
       original prices-history endpoint's behavior for empty hours).
       Returns sorted list of {t, p}."""
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
        if oi == 0:
            yes_price = price
        else:
            yes_price = 1.0 - price
        hour = (ts // 3600) * 3600
        if hour not in bars_by_hour or ts > bars_by_hour[hour][0]:
            bars_by_hour[hour] = (ts, yes_price)
    if not bars_by_hour:
        return []
    # Forward-fill: produce a bar for every hour between first and last trade
    min_h = min(bars_by_hour.keys())
    max_h = max(bars_by_hour.keys())
    out = []
    last_p = None
    for h in range(min_h, max_h + 3600, 3600):
        if h in bars_by_hour:
            last_p = bars_by_hour[h][1]
        if last_p is not None:
            out.append({"t": h, "p": last_p})
    return out

def process_market(m):
    """Returns list of trades from running z-score strategy on reconstructed bars."""
    cond = m.get("conditionId")
    if not cond:
        return []
    raw_trades = fetch_trades_all(cond)
    if not raw_trades:
        return []
    tids = m.get("clobTokenIds")
    try:
        yes_token = json.loads(tids)[0] if isinstance(tids, str) else tids[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        yes_token = None
    hist = reconstruct_hourly_bars(raw_trades, yes_token)
    if len(hist) < ROLLING_WINDOW + 5:
        return []
    times = np.array([h["t"] for h in hist])
    prices = np.array([h["p"] for h in hist], dtype=float)
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
                    "n_raw_trades": len(raw_trades),
                }
                if end_ts:
                    trade["dte_days_at_entry"] = (end_ts - trade["entry_ts"]) / 86400
                out_trades.append(trade)
                in_pos = False
    return out_trades

def load_progress():
    if PROGRESS_FILE_V2.exists():
        return set(str(x) for x in json.loads(PROGRESS_FILE_V2.read_text()))
    return set()

def save_progress(done):
    PROGRESS_FILE_V2.write_text(json.dumps(sorted(done)))

def main():
    # Load resolved markets and filter to high-volume
    all_markets = json.loads(RESOLVED_MARKETS.read_text())
    high_vol = [m for m in all_markets if num(m, "volumeNum") >= MIN_LIFETIME_VOLUME and m.get("conditionId")]
    high_vol.sort(key=lambda m: -num(m, "volumeNum"))
    high_vol = high_vol[:MAX_UNIVERSE_SIZE]
    print(f"[{now()}] Universe of high-volume resolved markets: {len(high_vol):,}", flush=True)

    done = load_progress()
    todo = [m for m in high_vol if str(m["id"]) not in done]
    print(f"[{now()}] To process: {len(todo):,} (skipping {len(done):,} already done)\n", flush=True)

    if not todo:
        print("All done!")
        return

    # Existing trade count
    total_trades = 0
    total_filtered = 0
    if TRADES_FILE_V2.exists():
        with open(TRADES_FILE_V2) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                total_trades += 1
                try:
                    t = json.loads(line)
                    if abs(t["entry_z"]) >= 5.0 and 0.10 <= t["entry_price"] <= 0.90:
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
                        with open(TRADES_FILE_V2, "a") as f:
                            for tr in trades:
                                f.write(json.dumps(tr, default=str) + "\n")
                        total_trades += len(trades)
                        for tr in trades:
                            if abs(tr["entry_z"]) >= 5.0 and 0.10 <= tr["entry_price"] <= 0.90:
                                total_filtered += 1
                    done.add(str(m["id"]))
                else:
                    errors += 1
                    with open(ERRORS_FILE_V2, "a") as f:
                        f.write(json.dumps({"market_id": m["id"], "error": payload, "ts": now()}) + "\n")
            except Exception:
                errors += 1
            if i % 20 == 0 or i == len(todo):
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

    save_progress(done)
    print(f"\n[{now()}] ✓ DONE  Total trades: {total_trades:,}  Filtered: {total_filtered:,}", flush=True)

if __name__ == "__main__":
    main()
