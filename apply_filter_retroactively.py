#!/usr/bin/env python3
"""
apply_filter_retroactively.py — Remove positions/trades whose entry slippage
exceeded max_slippage_for_entry, as if the filter had been active from day 1.

Rejected entries are MOVED (not deleted) to archive files so audit trail is
preserved:
  paper_trades.jsonl  -> rejects go to paper_trades_prefilter.jsonl
  paper_state.json    -> rejected positions go to paper_positions_prefilter.jsonl

After this runs, dashboard will show only positions/trades that would have
passed the 2¢ slippage filter — clean restart with the new model.

Run once. Idempotent: re-running won't move anything (because everything in
main files already passes filter).
"""
import json
import time
from pathlib import Path

THRESHOLD = 0.02  # 2¢ — must match max_slippage_for_entry in paper_trader.py

HERE = Path(__file__).parent
TRADES_FILE = HERE / "paper_trades.jsonl"
STATE_FILE = HERE / "paper_state.json"
TRADES_ARCHIVE = HERE / "paper_trades_prefilter.jsonl"
POSITIONS_ARCHIVE = HERE / "paper_positions_prefilter.jsonl"

def slip(p):
    em = p.get("entry_price_mid")
    ee = p.get("entry_exec_price")
    d  = p.get("direction")
    if em is None or ee is None or d is None:
        return None
    return (ee - em) * d

def main():
    # ── Closed trades ──
    closed = []
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            closed = [json.loads(l) for l in f if l.strip()]

    keep_closed = []
    archive_closed = []
    for t in closed:
        s = slip(t)
        if s is None:
            keep_closed.append(t)
            continue
        if s > THRESHOLD:
            t["_archived_reason"] = f"slip {s*100:+.1f}¢ exceeded threshold {THRESHOLD*100:.0f}¢"
            t["_archived_at"] = int(time.time())
            archive_closed.append(t)
        else:
            keep_closed.append(t)

    # ── Open positions ──
    state = json.load(open(STATE_FILE)) if STATE_FILE.exists() else {}
    positions = state.get("positions", {})

    keep_positions = {}
    archive_positions = []
    for mid, p in positions.items():
        s = slip(p)
        if s is None:
            keep_positions[mid] = p
            continue
        if s > THRESHOLD:
            p["_archived_reason"] = f"slip {s*100:+.1f}¢ exceeded threshold {THRESHOLD*100:.0f}¢"
            p["_archived_at"] = int(time.time())
            archive_positions.append(p)
        else:
            keep_positions[mid] = p

    # ── Report ──
    print(f"Closed trades: {len(closed)} total → keep {len(keep_closed)}, archive {len(archive_closed)}")
    print(f"Open positions: {len(positions)} total → keep {len(keep_positions)}, archive {len(archive_positions)}")
    if archive_closed:
        for t in archive_closed:
            print(f"  Closed archived: {t.get('question','')[:50]}  slip={slip(t)*100:+.1f}¢  pnl=${t.get('net_pnl_usd', 0):+.2f}")
    if archive_positions:
        for p in archive_positions:
            print(f"  Open archived:   {p.get('question','')[:50]}  slip={slip(p)*100:+.1f}¢")

    # ── Write archives (append, in case re-run) ──
    if archive_closed:
        with open(TRADES_ARCHIVE, "a") as f:
            for t in archive_closed:
                f.write(json.dumps(t, default=str) + "\n")
        print(f"  → archived {len(archive_closed)} closed trades to {TRADES_ARCHIVE.name}")
    if archive_positions:
        with open(POSITIONS_ARCHIVE, "a") as f:
            for p in archive_positions:
                f.write(json.dumps(p, default=str) + "\n")
        print(f"  → archived {len(archive_positions)} open positions to {POSITIONS_ARCHIVE.name}")

    # ── Rewrite main files ──
    if archive_closed:
        with open(TRADES_FILE, "w") as f:
            for t in keep_closed:
                f.write(json.dumps(t, default=str) + "\n")
        print(f"  → rewrote {TRADES_FILE.name} with {len(keep_closed)} kept trades")

    if archive_positions:
        state["positions"] = keep_positions
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, default=str, indent=2)
        print(f"  → rewrote {STATE_FILE.name} with {len(keep_positions)} kept positions")

    if not archive_closed and not archive_positions:
        print("\nNothing to archive — everything already passes filter.")

if __name__ == "__main__":
    main()
