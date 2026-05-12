#!/usr/bin/env python3
"""
analyze_slippage_filter.py — Estimate impact of max_slippage_for_entry filter.

Backtest reconstructs prices from trade tape (data-api/trades), so it has no
orderbook depth. We synthesize per-trade slippage from observable market
features and compare:
  A) baseline: charge each trade its synthetic spread cost
  B) with filter: drop trades whose synthetic spread > 2¢, charge rest

Synthetic spread heuristic uses trades-per-bar as a liquidity proxy.
Calibrated so it predicts ~7-14¢ slippage for Lithuania-grade trades
(0.2 trades/bar, $20k lifetime volume) and ~1-2¢ for top markets.

Run as `analyze_slippage_filter.py [v1|v3]` (defaults to v3).
"""
import json
import sys
from pathlib import Path
import numpy as np

VARIANT = (sys.argv[1] if len(sys.argv) > 1 else "v3").lower()

# Both strategies are evaluated against the LOW-VOLUME resolved-market universe
# ($5k-$50k lifetime volume) since that's what the live trader actually sees.
# extended_trades.jsonl is the earlier HIGH-volume run and has only 184 trades —
# unrepresentative of where we trade.
TRADES_FILE = "extended_trades_v3.jsonl"

if VARIANT == "v1":
    LABEL = "V1 (dte<365 NOT in [7,30)) on low-vol universe"
elif VARIANT == "v3":
    LABEL = "V3 (dte>=30) on low-vol universe"
else:
    print(f"Unknown variant: {VARIANT}"); sys.exit(1)

FEE_RATE = 0.025
STAKE = 30.0           # paper trader trade size
MAX_SLIPPAGE = 0.02    # the filter threshold

def keep_v1(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    dte = t.get("dte_days_at_entry")
    if dte is None or dte < 0 or dte >= 365: return False
    if 7 <= dte < 30: return False
    return True

def keep_v3(t):
    if abs(t["entry_z"]) < 5.0: return False
    if not (0.10 <= t["entry_price"] <= 0.90): return False
    dte = t.get("dte_days_at_entry")
    if dte is None or dte < 30 or dte >= 365: return False
    return True

KEEP = keep_v1 if VARIANT == "v1" else keep_v3

# ── Synthetic per-share slippage model ──
# Liquidity proxy = trades per bar (history_bars is hourly bar count over market lifetime).
# Calibration:
#   Lithuania live trade had 7.2¢ adverse slippage. Its market params (approx):
#     lifetime_volume=$22k, n_raw_trades=170, history_bars=900 → 0.19 trades/bar.
#   Bigger markets ($100k+ vol with ~5+ trades/bar) we observe ~1-2¢ spreads.
# Step function chosen for transparency over a fitted curve.
def synthetic_slippage(t):
    bars = max(int(t.get("history_bars", 1)), 1)
    nt = int(t.get("n_raw_trades", 0))
    tpb = nt / bars
    if tpb >= 5:
        base = 0.005
    elif tpb >= 2:
        base = 0.010
    elif tpb >= 1:
        base = 0.020
    elif tpb >= 0.5:
        base = 0.035
    elif tpb >= 0.2:
        base = 0.060      # Lithuania regime
    else:
        base = 0.100
    # Endpoints (close to 0/1) widen spreads slightly
    p = t["entry_price"]
    edge_penalty = 0.005 if p < 0.15 or p > 0.85 else 0.0
    return base + edge_penalty

def pnl_with_slippage(t, slip_per_share):
    """Net $ PnL on $30 stake using realistic per-share slippage."""
    direction = t["direction"]
    entry_p = t["entry_price"]
    exit_p = t["exit_price"]
    # Adverse fill at entry: LONG pays mid+slip, SHORT sells at mid-slip
    eff_entry = entry_p + direction * slip_per_share
    eff_exit  = exit_p  - direction * slip_per_share
    eff_entry = min(max(eff_entry, 0.001), 0.999)
    eff_exit  = min(max(eff_exit,  0.001), 0.999)
    shares = STAKE / eff_entry if direction == 1 else STAKE / (1 - eff_entry)
    gross_per_share = direction * (eff_exit - eff_entry)
    gross = shares * gross_per_share
    f_entry = FEE_RATE * eff_entry * (1 - eff_entry)
    f_exit  = FEE_RATE * eff_exit  * (1 - eff_exit)
    return gross - shares * (f_entry + f_exit)

def load():
    rows = []
    with open(TRADES_FILE) as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows

def main():
    raw = load()
    kept = [t for t in raw if KEEP(t)]
    print(f"=== Slippage filter impact — {LABEL} ===")
    print(f"Source file:  {TRADES_FILE}")
    print(f"Raw trades:   {len(raw):,}")
    print(f"Strat-kept:   {len(kept):,}")
    if not kept:
        print("No trades after strat filter.")
        return

    slip = np.array([synthetic_slippage(t) for t in kept])
    pnl_baseline = np.array([pnl_with_slippage(t, s) for t, s in zip(kept, slip)])
    keep_mask = slip <= MAX_SLIPPAGE
    pnl_filtered_only = pnl_baseline[keep_mask]

    print(f"\nSynthetic slippage distribution (¢/share):")
    print(f"  p10={np.percentile(slip,10)*100:.2f}¢  p50={np.percentile(slip,50)*100:.2f}¢  p90={np.percentile(slip,90)*100:.2f}¢")
    print(f"  trades with slip > {MAX_SLIPPAGE*100:.0f}¢: {(~keep_mask).sum():,} of {len(kept):,} ({(~keep_mask).mean()*100:.1f}%)")

    print(f"\n--- A) Baseline (no filter, variable slippage) ---")
    print(f"  N:           {len(kept):,}")
    print(f"  Mean $/trade: ${pnl_baseline.mean():+.3f}")
    print(f"  Median:      ${np.median(pnl_baseline):+.3f}")
    print(f"  Win rate:    {(pnl_baseline>0).mean()*100:.1f}%")
    print(f"  Total $:     ${pnl_baseline.sum():+.2f}")
    print(f"  Sharpe:      {pnl_baseline.mean()/pnl_baseline.std() if pnl_baseline.std() > 0 else 0:.3f}")

    print(f"\n--- B) With slippage filter (drop synth-spread > {MAX_SLIPPAGE*100:.0f}¢) ---")
    if len(pnl_filtered_only):
        print(f"  N:           {len(pnl_filtered_only):,}  ({len(pnl_filtered_only)/len(kept)*100:.1f}% of strat-kept)")
        print(f"  Mean $/trade: ${pnl_filtered_only.mean():+.3f}")
        print(f"  Median:      ${np.median(pnl_filtered_only):+.3f}")
        print(f"  Win rate:    {(pnl_filtered_only>0).mean()*100:.1f}%")
        print(f"  Total $:     ${pnl_filtered_only.sum():+.2f}")
        print(f"  Sharpe:      {pnl_filtered_only.mean()/pnl_filtered_only.std() if pnl_filtered_only.std() > 0 else 0:.3f}")
    else:
        print("  N: 0  (filter ate every trade — threshold too tight)")

    print(f"\n--- Delta ---")
    if len(pnl_filtered_only):
        d_mean = pnl_filtered_only.mean() - pnl_baseline.mean()
        d_total = pnl_filtered_only.sum() - pnl_baseline.sum()
        d_win   = (pnl_filtered_only>0).mean() - (pnl_baseline>0).mean()
        print(f"  Mean $/trade:  {d_mean:+.3f}   ({'↑ filter helps per-trade' if d_mean > 0 else '↓ filter hurts per-trade'})")
        print(f"  Win rate:      {d_win*100:+.1f}pp")
        print(f"  Total $:       {d_total:+.2f}   ({'↑ filter helps total $' if d_total > 0 else '↓ filter hurts total $'})")

    # ── Threshold sweep — where is the profitability inflection? ──
    print(f"\n--- Threshold sweep (drop trades with synth-slip > T) ---")
    print(f"  {'T (¢)':>6}  {'kept':>5}  {'%kept':>6}  {'mean $':>8}  {'win%':>6}  {'total $':>10}  {'Sharpe':>7}")
    for t_pct in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 99.0):
        T = t_pct / 100
        mask = slip <= T
        if mask.sum() == 0:
            print(f"  {t_pct:>6.1f}  {0:>5}  {'-':>6}  {'-':>8}  {'-':>6}  {'-':>10}  {'-':>7}")
            continue
        sel = pnl_baseline[mask]
        sh = sel.mean()/sel.std() if sel.std() > 0 else 0
        print(f"  {t_pct:>6.1f}  {mask.sum():>5}  {mask.mean()*100:>5.1f}%  {sel.mean():>+8.3f}  {(sel>0).mean()*100:>5.1f}%  {sel.sum():>+10.2f}  {sh:>+7.3f}")

    # ── PnL by slippage bucket — shows whether high-slip trades are net-losers ──
    print(f"\n--- PnL by synthetic-slippage bucket ---")
    buckets = [(0, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, 1)]
    print(f"  {'bucket (¢)':>14}  {'n':>5}  {'mean $':>8}  {'win%':>6}  {'total $':>10}")
    for lo, hi in buckets:
        mask = (slip >= lo) & (slip < hi)
        if mask.sum() == 0: continue
        sel = pnl_baseline[mask]
        print(f"  {f'[{lo*100:.0f},{hi*100:.0f})':>14}  {mask.sum():>5}  {sel.mean():>+8.3f}  {(sel>0).mean()*100:>5.1f}%  {sel.sum():>+10.2f}")

if __name__ == "__main__":
    main()
