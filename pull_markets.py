"""Pull all active Polymarket markets from Gamma API, paginated."""
import json
import time
import requests
from pathlib import Path

OUT = Path(__file__).parent / "markets.json"
URL = "https://gamma-api.polymarket.com/markets"
PAGE = 500

all_markets = []
offset = 0
while True:
    params = {
        "limit": PAGE,
        "offset": offset,
        "active": "true",
        "closed": "false",
        "archived": "false",
    }
    r = requests.get(URL, params=params, timeout=30)
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    all_markets.extend(batch)
    print(f"  offset={offset} got={len(batch)} total={len(all_markets)}")
    if len(batch) < PAGE:
        break
    offset += PAGE
    time.sleep(0.3)

OUT.write_text(json.dumps(all_markets))
print(f"Saved {len(all_markets)} markets to {OUT}")
