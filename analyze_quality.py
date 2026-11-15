"""Measure how badly the collector's order-book bands and trade-flow window were miscalibrated.

Reads micro/collect_20261014T175001Z.json and, for the 10 symbols whose raw books were
stored, checks whether the +/-0.5% and +/-1% bands actually cut the book at all.
"""

from __future__ import annotations

import json
from pathlib import Path

data = json.loads(
    Path("micro/collect_20261014T175001Z.json").read_text(encoding="utf-8")
)
aggs = data["aggregates"]
raw = data["raw"]

same = [a["symbol"] for a in aggs if a.get("imb_05") == a.get("imb_1")]
print(f"symbols where imb_05 == imb_1 exactly : {len(same)} / {len(aggs)}")
print("  -> in those cases the +/-0.5% and +/-1% bands never bound; the metric")
print("     is really 'imbalance of the visible top-50 levels', not a band imbalance.")
print(f"  example: {', '.join(same[:12])}")

zero_depth = [a["symbol"] for a in aggs if (a.get("depth_bid_usd") or 0) < 500_000]
print(
    f"\nsymbols with 'depth within +/-1%' under 0.5M USD : {len(zero_depth)} / {len(aggs)}"
)
print("  -> the band is not being measured; the book simply ends before it.")

print("\nactual price coverage of a limit=50 book (from the stored raw books):")
print(
    f"  {'symbol':<12}{'levels':>7}{'down %':>10}{'up %':>10}   band asked = +/-1.000%"
)
for sym, book in raw.items():
    bids = book["bids"]
    asks = book["asks"]
    mid = (bids[0][0] + asks[0][0]) / 2
    down = (mid - bids[-1][0]) / mid * 100
    up = (asks[-1][0] - mid) / mid * 100
    flag = "TRUNCATED" if (down < 1.0 or up < 1.0) else "ok"
    print(f"  {sym:<12}{len(bids):>7}{down:>10.3f}{up:>10.3f}   {flag}")
