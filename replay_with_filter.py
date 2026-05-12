#!/usr/bin/env python3
"""
replay_with_filter.py — Show which existing positions would have been rejected
by the new max_slippage_for_entry filter, and the PnL impact.

Reads paper_trades.jsonl + paper_state.json and computes for each entry:
    signed_slip = (exec - mid) * direction
If > 2¢, filter would have rejected.

For closed trades we know the actual PnL — easy.
For open positions we can't know final PnL, but we report current state.
"""
import json
from pathlib import Path

THRESHOLD = 0.02  # 2¢

HERE = Path(__file__).parent
closed = [json.loads(l) for l in open(HERE / "paper_trades.jsonl") if l.strip()]
state = json.load(open(HERE / "paper_state.json"))
open_pos = list(state.get("positions", {}).values())

def slip(p):
    em = p.get("entry_price_mid")
    ee = p.get("entry_exec_price")
    d  = p.get("direction")
    if em is None or ee is None or d is None:
        return None
    return (ee - em) * d  # >0 = adverse

print("=" * 78)
print(f"REPLAY: max_slippage_for_entry = {THRESHOLD*100:.0f}¢")
print("=" * 78)

# ── Closed trades ──
print(f"\n── CLOSED TRADES ({len(closed)}) ──")
print(f"{'q':30s}  {'dir':>4}  {'mid':>6}  {'exec':>6}  {'slip¢':>6}  {'PnL':>8}  filter")
kept_pnl = 0.0
saved_pnl = 0.0
for t in closed:
    s = slip(t)
    if s is None: continue
    pnl = t.get("net_pnl_usd", 0)
    keep = s <= THRESHOLD
    if keep:
        kept_pnl += pnl
    else:
        saved_pnl -= pnl  # rejecting a losing trade = positive saving
    q = (t.get("question") or "")[:30]
    em = t.get("entry_price_mid", 0)
    ee = t.get("entry_exec_price", 0)
    d  = t.get("direction", 0)
    dl = "SHORT" if d == -1 else "LONG"
    marker = "✓ KEEP" if keep else "✗ REJECT"
    print(f"{q:30s}  {dl:>5}  {em:>6.3f}  {ee:>6.3f}  {s*100:>+5.1f}  ${pnl:>+7.2f}  {marker}")

# ── Open positions ──
print(f"\n── OPEN POSITIONS ({len(open_pos)}) ──")
print(f"{'q':30s}  {'dir':>4}  {'mid':>6}  {'exec':>6}  {'slip¢':>6}  {'stake':>6}  filter")
n_kept = n_rejected = 0
deployed_kept = deployed_rejected = 0.0
for p in open_pos:
    s = slip(p)
    if s is None: continue
    keep = s <= THRESHOLD
    em = p.get("entry_price_mid", 0)
    ee = p.get("entry_exec_price", 0)
    d  = p.get("direction", 0)
    dl = "SHORT" if d == -1 else "LONG"
    # Deployed capital approximation
    shares = p.get("shares", 0)
    if d == 1:
        deployed = shares * ee
    else:
        deployed = shares * (1 - ee)
    if keep:
        n_kept += 1
        deployed_kept += deployed
    else:
        n_rejected += 1
        deployed_rejected += deployed
    q = (p.get("question") or "")[:30]
    marker = "✓ KEEP" if keep else "✗ REJECT"
    print(f"{q:30s}  {dl:>5}  {em:>6.3f}  {ee:>6.3f}  {s*100:>+5.1f}  ${deployed:>5.2f}  {marker}")

# ── Summary ──
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"\nClosed trades (definitive PnL impact):")
print(f"  Filter would have kept:     ${kept_pnl:+.2f}")
print(f"  Filter would have saved:    ${saved_pnl:+.2f}  (rejected losing trades)")
total_actual = sum(t.get('net_pnl_usd', 0) for t in closed)
total_with_filter = kept_pnl
print(f"  Actual realized PnL:        ${total_actual:+.2f}")
print(f"  With filter (counterfactual): ${total_with_filter:+.2f}")
print(f"  Δ from filter:               ${total_with_filter - total_actual:+.2f}")

print(f"\nOpen positions:")
print(f"  Filter would have kept:     {n_kept}/{len(open_pos)}  (${deployed_kept:.2f} deployed)")
print(f"  Filter would have rejected: {n_rejected}/{len(open_pos)}  (${deployed_rejected:.2f} that would have stayed un-deployed)")
print(f"  Total deployed without filter: ${deployed_kept + deployed_rejected:.2f}")
print(f"  Note: open PnL is unknown until close — these positions still live.")
