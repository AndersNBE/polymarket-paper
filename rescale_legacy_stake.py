#!/usr/bin/env python3
"""
rescale_legacy_stake.py — Scale down legacy $100-stake positions to the
current $30 config. Treats it as if the position had always been opened
at $30: shares and entry_fee shrink proportionally, gas stays fixed
(it's per-transaction, not per-share).

Only positions whose effective stake > TARGET_STAKE * 1.5 are rescaled,
so book-walk variance around $30 isn't touched.

Run once. Idempotent.
"""
import json
from pathlib import Path

TARGET_STAKE = 30.0
RESCALE_IF_OVER = TARGET_STAKE * 1.5  # $45

HERE = Path(__file__).parent
STATE_FILE = HERE / "paper_state.json"

def stake_of(p):
    d = p.get("direction")
    ee = p.get("entry_exec_price", 0)
    s = p.get("shares", 0)
    if d == 1:
        return s * ee
    else:
        return s * (1 - ee)

def main():
    state = json.load(open(STATE_FILE))
    positions = state.get("positions", {})
    n_changed = 0
    for mid, p in positions.items():
        old_stake = stake_of(p)
        if old_stake <= RESCALE_IF_OVER:
            continue
        ratio = TARGET_STAKE / old_stake
        p["_legacy_shares"] = p.get("shares")
        p["_legacy_entry_fee"] = p.get("entry_fee")
        p["_legacy_stake"] = old_stake
        p["shares"] = p["shares"] * ratio
        p["entry_fee"] = p.get("entry_fee", 0) * ratio
        # gas is per-fill flat cost — keep as-is
        new_stake = stake_of(p)
        n_changed += 1
        print(f"  {p.get('question','')[:50]}: ${old_stake:.2f} → ${new_stake:.2f}  (ratio {ratio:.3f})")
    if n_changed:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, default=str, indent=2)
        print(f"\nRescaled {n_changed} positions; saved to {STATE_FILE.name}.")
    else:
        print("Nothing to rescale (all positions already at target stake).")

if __name__ == "__main__":
    main()
