#!/usr/bin/env python3
"""
paper_trader.py — Live paper-trading the filtered mean-reversion strategy.

Strategy (locked from backtest):
  ENTRY: |z-score(1h close vs 24h rolling mean)| >= 5
         AND  0.10 <= entry_price <= 0.90
         AND  days-to-resolution < 365 and NOT in [7, 30]
         AND  bestBid/bestAsk both present and spread reasonable
  EXIT:  |z| < 0.5  OR  48h held  OR  resolution imminent

Execution model:
  - BUY (long YES):  pay bestAsk + slippage_ticks*tickSize
  - SELL (short via NO):  receive bestBid - slippage_ticks*tickSize
  - Polymarket fee:  C * feeRate * p * (1-p) per side
  - Gas: $0.05 per fill (conservative for Polygon)
  - Slippage: 1 tick per side (conservative)

Run:
  python3 paper_trader.py                # main loop, 15-min polling
  python3 paper_trader.py --single       # one iteration then exit
  python3 paper_trader.py --status       # print state summary
"""
import json
import math
import os
import random
import signal as _signal
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

# ────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

# Version is set via --version flag (v1 = current/baseline, v2 = improved)
# Default is v1 for backward compatibility.
VERSION = "v1"
for _a in sys.argv:
    if _a.startswith("--version="):
        VERSION = _a.split("=", 1)[1]
    elif _a == "--version" and sys.argv.index(_a) + 1 < len(sys.argv):
        VERSION = sys.argv[sys.argv.index(_a) + 1]
if "--v2" in sys.argv: VERSION = "v2"
assert VERSION in ("v1", "v2"), f"Unknown version: {VERSION}"

_suffix = "" if VERSION == "v1" else "_" + VERSION
STATE = HERE / f"paper_state{_suffix}.json"
TRADE_LOG = HERE / f"paper_trades{_suffix}.jsonl"
SIGNAL_LOG = HERE / f"paper_signals{_suffix}.jsonl"
CYCLE_LOG = HERE / f"paper_cycles{_suffix}.jsonl"
DAILY_LOG = HERE / f"paper_daily{_suffix}.csv"

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# Base CFG; per-version overrides applied below
CFG = {
    "poll_interval_sec": 15 * 60,
    "universe_refresh_sec": 6 * 3600,
    "max_open_positions": 10,   # realistic for $720 bankroll with $30 stake
    # universe filters
    "min_liquidity": 1000,
    "max_liquidity": 50000,
    "min_v24": 50,
    "max_v24": 10000,
    "min_v1mo": 2000,
    # signal filters — V1 defaults
    "rolling_window": 24,    # bars
    "entry_z": 5.0,
    "exit_z": 0.5,
    "max_hold_hours": 48,
    "price_min": 0.10,
    "price_max": 0.90,
    "dte_max": 365,
    "dte_block_lo": 7,       # block trades with this many days to end
    "dte_block_hi": 30,
    "direction_filter": None,   # None = both; "short" = only z>0; "long" = only z<0
    # execution model
    "trade_size_usd": 100.0,
    "fee_rates": {
        "sports_fees_v2": 0.03,
        "crypto_fees_v2": 0.07,
        "politics_fees": 0.04,
        "weather_fees": 0.05,
        "culture_fees": 0.05,
        "finance_prices_fees": 0.04,
        "tech_fees": 0.04,
        "economics_fees": 0.05,
        "mentions_fees": 0.04,
        "general_fees": 0.05,
        "crypto_15_min": 0.07,
        "_default": 0.05,
    },
    "gas_per_fill_usd": 0.15,    # conservative; can spike to $2 under Polygon congestion
    "slippage_ticks": 1,         # additional 1-tick "latency cost" beyond orderbook walk
    "clv_lookback_sec": 30 * 60, # measure CLV 30 min after entry
    # require both sides quoted with at least this many tick widths
    "max_spread_for_entry": 0.05,   # don't enter if spread > 5¢
    # rate limiting
    "api_delay_sec": 0.04,
    # price-history cache: hourly bars only update once an hour, so cache aggressively
    "history_cache_sec": 50 * 60,
    # scan limit per cycle (0 = no limit)
    "scan_limit": 0,
}

# V2 strategy overrides (improved version from strategy_improvements.py)
if VERSION == "v2":
    CFG["entry_z"] = 7.0          # was 5.0
    CFG["max_hold_hours"] = 12    # was 48
    CFG["direction_filter"] = "short"   # only short YES (z>0)

# ────────────────────────────────────────────────────────────────────────
# UTILITIES
# ────────────────────────────────────────────────────────────────────────
def now_ts() -> int:
    return int(time.time())

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def jget(o, k, default=None):
    try:
        v = o.get(k)
        return v if v is not None else default
    except AttributeError:
        return default

def to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def fee_for_fill(price, n_shares, fee_type):
    """Polymarket taker fee = n_shares × rate × p × (1-p)."""
    rate = CFG["fee_rates"].get(fee_type, CFG["fee_rates"]["_default"])
    return n_shares * rate * price * (1 - price)

def days_to_end_ts(end_date_str, now_ts_val):
    if not end_date_str:
        return None
    try:
        end_ts = datetime.fromisoformat(end_date_str.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None
    return (end_ts - now_ts_val) / 86400

def fetch_json(url, params=None, retries=2, timeout=15):
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        if i < retries:
            time.sleep(0.4 * (i + 1))
    return None

def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")

# ────────────────────────────────────────────────────────────────────────
# STATE
# ────────────────────────────────────────────────────────────────────────
def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {
        "universe_ts": 0,
        "universe": [],     # list of market_id
        "positions": {},    # market_id -> position dict
        "history_cache": {},  # market_id -> {"ts": int, "history": [...]}
        "cycle": 0,
        "started_at": now_iso(),
    }

def save_state(state):
    STATE.write_text(json.dumps(state, indent=2, default=str))

# ────────────────────────────────────────────────────────────────────────
# UNIVERSE
# ────────────────────────────────────────────────────────────────────────
def fetch_universe():
    """Pull all markets matching filter criteria. Returns dict by id."""
    print(f"[{now_iso()}] Refreshing universe...")
    out = {}
    offset = 0
    PAGE = 500
    while True:
        batch = fetch_json(
            f"{GAMMA}/markets",
            params={"limit": PAGE, "offset": offset,
                    "active": "true", "closed": "false", "archived": "false"},
        )
        if not batch:
            break
        for m in batch:
            liq = to_float(m.get("liquidityNum"), 0.0)
            v24 = to_float(m.get("volume24hr"), 0.0)
            v1mo = to_float(m.get("volume1mo"), 0.0)
            if not (CFG["min_liquidity"] <= liq <= CFG["max_liquidity"]):
                continue
            if not (CFG["min_v24"] <= v24 <= CFG["max_v24"]):
                continue
            if v1mo < CFG["min_v1mo"]:
                continue
            if not m.get("enableOrderBook"):
                continue
            if not m.get("clobTokenIds"):
                continue
            out[str(m["id"])] = m
        if len(batch) < PAGE:
            break
        offset += PAGE
        time.sleep(0.3)
    print(f"[{now_iso()}] Universe size: {len(out):,}")
    return out

def fetch_market_snapshot(market_id):
    """Get current state of a single market for live execution data."""
    return fetch_json(f"{GAMMA}/markets/{market_id}")

def fetch_price_history(token_id):
    data = fetch_json(
        f"{CLOB}/prices-history",
        params={"market": token_id, "fidelity": 60, "interval": "max"},
    )
    if data:
        return data.get("history", [])
    return None

def fetch_orderbook(token_id):
    """Returns dict with 'bids' (list[{price,size}]) and 'asks'."""
    return fetch_json(f"{CLOB}/book", params={"token_id": token_id})

def walk_orderbook(book, side, target_dollars):
    """Walk through orderbook levels for a buy (LONG) or sell (SHORT) of target_dollars.

    For LONG: walk asks ascending; we BUY YES shares at increasing prices.
    For SHORT: walk bids descending; we conceptually SELL YES shares (== buy NO).
      For each YES bid at price P, equivalent NO ask is (1-P).
      Capital cost per share when shorting = (1 - P).

    Returns: (avg_yes_price, shares_filled, dollars_spent) or (None, 0, 0) if no fill.
    """
    if not book:
        return None, 0, 0
    if side == "long":
        orders = book.get("asks") or []
        try:
            orders = sorted(orders, key=lambda o: float(o["price"]))
        except (KeyError, ValueError):
            return None, 0, 0
    else:  # short
        orders = book.get("bids") or []
        try:
            orders = sorted(orders, key=lambda o: -float(o["price"]))
        except (KeyError, ValueError):
            return None, 0, 0

    total_shares = 0.0
    total_cost = 0.0
    for o in orders:
        try:
            p = float(o["price"])
            s = float(o["size"])
        except (KeyError, ValueError, TypeError):
            continue
        cost_per_share = p if side == "long" else (1.0 - p)
        if cost_per_share <= 0:
            continue
        level_cost = cost_per_share * s
        if total_cost + level_cost >= target_dollars:
            remaining = target_dollars - total_cost
            sh = remaining / cost_per_share
            total_shares += sh
            total_cost = target_dollars
            break
        else:
            total_shares += s
            total_cost += level_cost
    if total_shares <= 0 or total_cost <= 0:
        return None, 0, 0
    if side == "long":
        avg_yes_price = total_cost / total_shares
    else:
        # avg NO price = total_cost / total_shares  →  avg YES price = 1 - avg NO price
        avg_yes_price = 1.0 - (total_cost / total_shares)
    return avg_yes_price, total_shares, total_cost

def get_history_cached(state, market_id, token_id):
    """Use cached history when fresh; otherwise refetch.
       Trim cached history to the most recent 100 bars to keep state file small."""
    cache = state.setdefault("history_cache", {})
    entry = cache.get(market_id)
    if entry and now_ts() - entry.get("ts", 0) < CFG["history_cache_sec"]:
        return entry.get("history")
    hist = fetch_price_history(token_id)
    time.sleep(CFG["api_delay_sec"])
    if hist is not None:
        hist_trimmed = hist[-100:] if len(hist) > 100 else hist
        cache[market_id] = {"ts": now_ts(), "history": hist_trimmed}
    return hist

def prune_history_cache(state, keep_ids):
    cache = state.get("history_cache", {})
    keep = set(str(x) for x in keep_ids)
    state["history_cache"] = {k: v for k, v in cache.items() if k in keep}

def get_yes_token(market):
    tids = market.get("clobTokenIds")
    try:
        arr = json.loads(tids) if isinstance(tids, str) else tids
        return arr[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None

# ────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
# ────────────────────────────────────────────────────────────────────────
def compute_signal(prices, window):
    if len(prices) < window + 1:
        return None
    win = prices[-window-1:-1]   # exclude current bar
    mu = float(np.mean(win))
    sd = float(np.std(win))
    if sd < 0.005:
        return None
    current = float(prices[-1])
    z = (current - mu) / sd
    return {"z": z, "mu": mu, "sd": sd, "current": current}

# ────────────────────────────────────────────────────────────────────────
# EXECUTION (paper)
# ────────────────────────────────────────────────────────────────────────
def simulate_entry(market, direction, signal_data, token_id=None):
    """direction=-1 means we're short (sell YES at bid); +1 means long (buy YES at ask).
       Uses real orderbook (walks the book) to compute fill price for our trade size.
       Returns position dict or None."""
    bid = to_float(market.get("bestBid"))
    ask = to_float(market.get("bestAsk"))
    tick = to_float(market.get("orderPriceMinTickSize"), 0.01)
    if bid is None or ask is None:
        return None
    spread = ask - bid
    if spread > CFG["max_spread_for_entry"]:
        return None
    slip = CFG["slippage_ticks"] * tick

    # Walk the orderbook for realistic fill price
    book = fetch_orderbook(token_id) if token_id else None
    time.sleep(CFG["api_delay_sec"])
    if book:
        side = "long" if direction == 1 else "short"
        avg_yes_price, shares, dollars_spent = walk_orderbook(book, side, CFG["trade_size_usd"])
        if avg_yes_price is None:
            return None  # not enough depth
        # Apply additional latency-slippage (1 tick beyond walked price)
        if direction == 1:
            exec_price = min(0.999, avg_yes_price + slip)
        else:
            exec_price = max(0.001, avg_yes_price - slip)
    else:
        # Fallback: use bestBid/Ask
        if direction == 1:
            exec_price = ask + slip
        else:
            exec_price = bid - slip
        shares = CFG["trade_size_usd"] / exec_price if direction == 1 else CFG["trade_size_usd"] / (1 - exec_price)

    if exec_price <= 0.001 or exec_price >= 0.999:
        return None
    fee_type = market.get("feeType")
    entry_fee = fee_for_fill(exec_price, shares, fee_type)
    gas = CFG["gas_per_fill_usd"]
    return {
        "market_id": str(market["id"]),
        "question": (market.get("question") or "")[:140],
        "fee_type": fee_type,
        "direction": direction,
        "entry_ts": now_ts(),
        "entry_price_mid": signal_data["current"],
        "entry_exec_price": exec_price,
        "entry_bid": bid,
        "entry_ask": ask,
        "entry_spread": spread,
        "entry_z": signal_data["z"],
        "entry_mu": signal_data["mu"],
        "entry_sd": signal_data["sd"],
        "shares": shares,
        "entry_fee": entry_fee,
        "entry_gas": gas,
        "tick_size": tick,
        "end_date": market.get("endDate"),
        "clv_check_at": now_ts() + CFG["clv_lookback_sec"],
        "clv_price": None,
        "clv_value": None,
        "yes_token_id": token_id,
    }

def simulate_exit(position, market, reason, exit_price_mid):
    bid = to_float(market.get("bestBid"))
    ask = to_float(market.get("bestAsk"))
    tick = position.get("tick_size", 0.01)
    if bid is None or ask is None:
        bid = exit_price_mid
        ask = exit_price_mid
    slip = CFG["slippage_ticks"] * tick
    direction = position["direction"]
    shares = position["shares"]
    fee_type = position.get("fee_type")

    # Walk orderbook for exit (opposite side of entry)
    yes_token = position.get("yes_token_id")
    book = fetch_orderbook(yes_token) if yes_token else None
    time.sleep(CFG["api_delay_sec"])
    if book:
        # Exit side: long-exit = sell YES (walk bids); short-exit = buy YES (walk asks)
        side = "short" if direction == 1 else "long"
        # Target $ we receive (long exit) or pay (short exit) ≈ shares * avg_price
        # Walk by target_dollars = shares * current best_quote as estimate
        est_price = bid if direction == 1 else ask
        target_dollars = shares * (est_price if direction == 1 else (1 - est_price))
        avg_yes_price, shares_filled, _ = walk_orderbook(book, side, target_dollars)
        if avg_yes_price is not None:
            if direction == 1:
                exec_price = max(0.001, avg_yes_price - slip)
            else:
                exec_price = min(0.999, avg_yes_price + slip)
        else:
            exec_price = max(0.001, bid - slip) if direction == 1 else min(0.999, ask + slip)
    else:
        if direction == 1:
            exec_price = max(0.001, bid - slip)
        else:
            exec_price = min(0.999, ask + slip)

    exit_fee = fee_for_fill(exec_price, shares, fee_type)
    gas = CFG["gas_per_fill_usd"]
    # PnL
    if direction == 1:
        gross_pnl = shares * (exec_price - position["entry_exec_price"])
    else:
        # short YES: profit when YES price falls
        gross_pnl = shares * (position["entry_exec_price"] - exec_price)
    net_pnl = gross_pnl - position["entry_fee"] - exit_fee - position["entry_gas"] - gas
    return {
        **position,
        "exit_ts": now_ts(),
        "exit_reason": reason,
        "exit_price_mid": exit_price_mid,
        "exit_exec_price": exec_price,
        "exit_bid": bid,
        "exit_ask": ask,
        "exit_fee": exit_fee,
        "exit_gas": gas,
        "hold_hours": (now_ts() - position["entry_ts"]) / 3600,
        "gross_pnl_usd": gross_pnl,
        "net_pnl_usd": net_pnl,
        "total_fees_usd": position["entry_fee"] + exit_fee + position["entry_gas"] + gas,
    }

# ────────────────────────────────────────────────────────────────────────
# MAIN CYCLE
# ────────────────────────────────────────────────────────────────────────
def run_cycle(state):
    state["cycle"] += 1
    cycle_no = state["cycle"]
    print(f"\n========== CYCLE #{cycle_no} [{VERSION.upper()}] @ {now_iso()} ==========")
    print(f"   strategy: entry_z>={CFG['entry_z']}, max_hold={CFG['max_hold_hours']}h, dir={CFG.get('direction_filter') or 'both'}")

    # Refresh universe if stale
    if now_ts() - state["universe_ts"] > CFG["universe_refresh_sec"]:
        uni = fetch_universe()
        if uni:
            state["universe"] = list(uni.keys())
            state["universe_ts"] = now_ts()
            state["_last_universe_data"] = uni  # cache for this cycle
    else:
        # Need fresh market data — refetch all universe markets in batches
        # Easiest: re-fetch full markets endpoint and filter to our universe
        print(f"[{now_iso()}] Re-fetching market quotes...")
        uni = fetch_universe()  # same filters, gets fresh bestBid/bestAsk
        if uni:
            state["_last_universe_data"] = uni
    uni = state.get("_last_universe_data", {})
    if not uni:
        print("No universe loaded, skipping cycle")
        return

    prune_history_cache(state, uni.keys())
    print(f"Tracking universe: {len(uni)} markets   (history cache size: {len(state.get('history_cache', {}))})")
    print(f"Open positions: {len(state['positions'])}")

    # ─── Step 1: Check exits for open positions ───
    closed_count = 0
    for mid in list(state["positions"].keys()):
        pos = state["positions"][mid]
        market = uni.get(mid)
        if market is None:
            # Market disappeared from universe (maybe resolved or filtered out)
            market = fetch_market_snapshot(mid)
            time.sleep(CFG["api_delay_sec"])
        if market is None:
            continue
        # If closed/archived, force exit at mid
        if market.get("closed") or market.get("archived"):
            tok = get_yes_token(market)
            hist = get_history_cached(state, mid, tok) if tok else None
            last = hist[-1]["p"] if hist else (
                to_float(market.get("lastTradePrice")) or
                0.5 * (to_float(market.get("bestBid"), 0.5) + to_float(market.get("bestAsk"), 0.5))
            )
            closed = simulate_exit(pos, market, "market_resolved", last)
            append_jsonl(TRADE_LOG, closed)
            del state["positions"][mid]
            closed_count += 1
            print(f"  ✗ Closed (resolved): {pos['question'][:60]}  pnl=${closed['net_pnl_usd']:+.2f}")
            continue
        # Fetch latest price history (cached)
        tok = get_yes_token(market)
        if not tok:
            continue
        hist = get_history_cached(state, mid, tok)
        if not hist or len(hist) < CFG["rolling_window"] + 1:
            continue
        prices = np.array([h["p"] for h in hist], dtype=float)
        sig = compute_signal(prices, CFG["rolling_window"])
        if sig is None:
            continue
        # CLV check: if past clv_check_at and not yet measured, capture mid-price NOW
        if pos.get("clv_price") is None and now_ts() >= pos.get("clv_check_at", float("inf")):
            cur_bid = to_float(market.get("bestBid"))
            cur_ask = to_float(market.get("bestAsk"))
            if cur_bid is not None and cur_ask is not None:
                clv_midprice = 0.5 * (cur_bid + cur_ask)
                pos["clv_price"] = clv_midprice
                # CLV positive = we entered at a better price than where market is now
                # Long: bought low; market moved up → mid > entry_mid → positive CLV
                # Short: sold high; market moved down → mid < entry_mid → positive CLV
                pos["clv_value"] = pos["direction"] * (clv_midprice - pos.get("entry_price_mid", clv_midprice))
        # Check exit
        held_hours = (now_ts() - pos["entry_ts"]) / 3600
        reason = None
        if abs(sig["z"]) < CFG["exit_z"]:
            reason = "z_revert"
        elif held_hours >= CFG["max_hold_hours"]:
            reason = "max_hold"
        if reason:
            closed = simulate_exit(pos, market, reason, sig["current"])
            append_jsonl(TRADE_LOG, closed)
            del state["positions"][mid]
            closed_count += 1
            print(f"  ✗ Closed ({reason}): {pos['question'][:60]}  pnl=${closed['net_pnl_usd']:+.2f}  z_at_exit={sig['z']:+.2f}  CLV={closed.get('clv_value') or 'n/a'}")

    # ─── Step 2: Detect new entries ───
    entered_count = 0
    rejected = {"no_token": 0, "no_history": 0, "no_signal": 0, "z_too_low": 0,
                "price_boundary": 0, "dte_filter": 0, "direction_filter": 0,
                "entry_failed_book": 0, "max_positions": 0}
    scanned = 0
    signals_seen = 0
    if len(state["positions"]) >= CFG["max_open_positions"]:
        print("Max positions reached, not opening new")
        rejected["max_positions"] = len(uni)
    else:
        total = len(uni)
        for mid, market in uni.items():
            if mid in state["positions"]:
                continue
            scanned += 1
            if CFG["scan_limit"] and scanned > CFG["scan_limit"]:
                print(f"  scan_limit reached ({CFG['scan_limit']})")
                break
            if scanned % 250 == 0:
                print(f"  ... scan progress {scanned}/{total}  signals_so_far={signals_seen}  opens_so_far={entered_count}")
            tok = get_yes_token(market)
            if not tok:
                rejected["no_token"] += 1
                continue
            hist = get_history_cached(state, mid, tok)
            if not hist or len(hist) < CFG["rolling_window"] + 1:
                rejected["no_history"] += 1
                continue
            prices = np.array([h["p"] for h in hist], dtype=float)
            sig = compute_signal(prices, CFG["rolling_window"])
            if sig is None:
                rejected["no_signal"] += 1
                continue
            if abs(sig["z"]) < CFG["entry_z"]:
                rejected["z_too_low"] += 1
                continue
            signals_seen += 1
            # Apply filters
            if not (CFG["price_min"] <= sig["current"] <= CFG["price_max"]):
                rejected["price_boundary"] += 1
                continue
            dte = days_to_end_ts(market.get("endDate"), now_ts())
            if dte is None or dte > CFG["dte_max"] or CFG["dte_block_lo"] <= dte < CFG["dte_block_hi"] or dte < 0:
                rejected["dte_filter"] += 1
                continue
            direction = -1 if sig["z"] > 0 else 1
            if CFG.get("direction_filter") == "short" and direction != -1:
                rejected["direction_filter"] += 1
                continue
            if CFG.get("direction_filter") == "long" and direction != 1:
                rejected["direction_filter"] += 1
                continue
            pos = simulate_entry(market, direction, sig, token_id=tok)
            if pos is None:
                rejected["entry_failed_book"] += 1
                continue
            pos["dte"] = dte
            state["positions"][mid] = pos
            entered_count += 1
            append_jsonl(SIGNAL_LOG, {
                "ts": now_ts(),
                "market_id": mid,
                "question": pos["question"],
                "z": sig["z"],
                "current_price": sig["current"],
                "direction": direction,
                "dte_days": dte,
                "entry_exec_price": pos.get("entry_exec_price"),
                "entry_spread": pos.get("entry_spread"),
                "shares": pos.get("shares"),
                "stake_usd": CFG["trade_size_usd"],
                "action": "OPENED",
            })
            print(f"  ✓ OPEN  {pos['question'][:60]}  z={sig['z']:+.2f}  px={sig['current']:.3f}  dir={direction:+d}")
            if len(state["positions"]) >= CFG["max_open_positions"]:
                break

        print(f"Scanned: {scanned}, signals seen: {signals_seen}, entered: {entered_count}")
        print(f"Filter rejections: {rejected}")

    # Persist cycle activity for the dashboard
    append_jsonl(CYCLE_LOG, {
        "ts": now_ts(),
        "cycle": cycle_no,
        "universe_size": len(uni),
        "history_cache_size": len(state.get("history_cache", {})),
        "open_positions_before": len(state["positions"]) - entered_count + closed_count,
        "open_positions_after": len(state["positions"]),
        "scanned": scanned,
        "signals_at_z5": signals_seen,
        "rejected": rejected,
        "opened": entered_count,
        "closed": closed_count,
    })

    print(f"\n[{now_iso()}] Cycle #{cycle_no} done. Open positions: {len(state['positions'])}, closed: {closed_count}, opened: {entered_count}")

    # Drop transient cache before save
    state.pop("_last_universe_data", None)
    save_state(state)
    print_pnl_summary()

# ────────────────────────────────────────────────────────────────────────
# REPORTING
# ────────────────────────────────────────────────────────────────────────
def print_pnl_summary():
    if not TRADE_LOG.exists():
        print("No closed trades yet")
        return
    trades = []
    with open(TRADE_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not trades:
        print("No closed trades yet")
        return
    pnl = sum(t["net_pnl_usd"] for t in trades)
    fees = sum(t["total_fees_usd"] for t in trades)
    wins = sum(1 for t in trades if t["net_pnl_usd"] > 0)
    print(f"  PAPER PnL so far: ${pnl:+.2f}   trades={len(trades)}   win%={100*wins/len(trades):.1f}%   total_fees=${fees:.2f}")

def print_status():
    state = load_state()
    print(f"Started: {state.get('started_at')}")
    print(f"Cycles run: {state.get('cycle', 0)}")
    print(f"Universe size: {len(state.get('universe', [])):,}")
    print(f"Open positions: {len(state.get('positions', {}))}")
    if state.get("positions"):
        print("Currently held:")
        for mid, pos in state["positions"].items():
            held_h = (now_ts() - pos["entry_ts"]) / 3600
            print(f"  [{held_h:5.1f}h]  z={pos['entry_z']:+.2f}  dir={pos['direction']:+d}  exec=${pos['entry_exec_price']:.3f}  {pos['question'][:60]}")
    print()
    print_pnl_summary()

# ────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ────────────────────────────────────────────────────────────────────────
_stop = False
def _on_signal(signum, frame):
    global _stop
    _stop = True
    print(f"\nReceived signal {signum}, finishing current cycle then stopping...")

def main():
    if "--status" in sys.argv:
        print_status()
        return
    _signal.signal(_signal.SIGINT, _on_signal)
    _signal.signal(_signal.SIGTERM, _on_signal)
    state = load_state()
    single = "--single" in sys.argv
    while not _stop:
        try:
            run_cycle(state)
        except Exception as e:
            print(f"ERROR in cycle: {e}")
            traceback.print_exc()
            time.sleep(60)
            continue
        if single:
            break
        # Sleep with periodic stop checks
        slept = 0
        while slept < CFG["poll_interval_sec"] and not _stop:
            time.sleep(5)
            slept += 5
    save_state(state)
    print("Stopped cleanly.")

if __name__ == "__main__":
    main()
