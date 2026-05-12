"""Quick profile of the market dataset."""
import json
from pathlib import Path
import numpy as np

markets = json.loads(Path("markets.json").read_text())
print(f"Total active markets: {len(markets):,}\n")

def num(m, k):
    v = m.get(k)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None

vols = [v for m in markets if (v := num(m, "volumeNum")) is not None]
liqs = [v for m in markets if (v := num(m, "liquidityNum")) is not None]
v24 = [v for m in markets if (v := num(m, "volume24hr")) is not None]

def pct(arr, ps=(0, 25, 50, 75, 90, 95, 99, 100)):
    a = np.array(arr)
    return {f"p{p}": round(float(np.percentile(a, p)), 2) for p in ps}

print("Volume (lifetime):", pct(vols))
print("Liquidity (now):  ", pct(liqs))
print("Volume 24h:       ", pct(v24))

# How many markets have meaningful trading?
v_thresh = 1000
l_thresh = 100
v24_thresh = 100
print(f"\nMarkets with lifetime volume >${v_thresh}: {sum(1 for v in vols if v >= v_thresh):,}")
print(f"Markets with liquidity >${l_thresh}: {sum(1 for v in liqs if v >= l_thresh):,}")
print(f"Markets with 24h volume >${v24_thresh}: {sum(1 for v in v24 if v >= v24_thresh):,}")

# How many have orderbook enabled?
ob = sum(1 for m in markets if m.get("enableOrderBook"))
print(f"\nMarkets with orderbook enabled: {ob:,}")

# Group analysis: how many distinct groups (groupItemTitle)?
from collections import Counter
groups = Counter()
for m in markets:
    g = m.get("groupItemTitle")
    if g:
        groups[g] += 1

print(f"\nMarkets with groupItemTitle: {sum(groups.values()):,}")
print(f"Distinct groups: {len(groups):,}")
print(f"Top 10 groups by member count:")
for name, n in groups.most_common(10):
    print(f"  {n:4d}  {name[:80]}")
