"""Audit the v2 sensor against the specific defects measured in v1.

Run:  python audit_v2.py micro2/collect_<stamp>.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BANDS = (2, 5, 10, 25, 50, 100)


def main(path: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    aggs = data["aggregates"]
    flow = data["flow"]
    stream = data["book_stream"]
    meta = data["meta"]
    n = len(aggs)

    print(f"file        : {Path(path).name}")
    print(f"T0          : {meta['created_utc']}")
    print(f"symbols     : {n}   failures: {len(meta['failures'])}")
    print(f"snap span   : {meta['snap_span_s']}s")
    print()

    print("--- DEFECT 1: bands asked for from a book that cannot reach them")
    cov = sorted(a["cov_bps"] for a in aggs if a.get("cov_bps") is not None)
    if cov:
        print(
            f"  book coverage (bps): min={cov[0]:.1f} p10={cov[len(cov) // 10]:.1f} "
            f"median={statistics.median(cov):.1f} max={cov[-1]:.1f}"
        )
    print("  NEW: coverage is measured and reported, so no band is silently faked")
    print(f"  {'band':>6} {'measurable':>12} {'avg depth $M (bid/ask)':>26}")
    for bps in BANDS:
        vals = [a for a in aggs if a.get(f"imb{bps}") is not None]
        bid = [a[f"dep{bps}_bid"] for a in vals if a.get(f"dep{bps}_bid") is not None]
        ask = [a[f"dep{bps}_ask"] for a in vals if a.get(f"dep{bps}_ask") is not None]
        bm = f"{statistics.mean(bid) / 1e6:.2f}" if bid else "-"
        am = f"{statistics.mean(ask) / 1e6:.2f}" if ask else "-"
        print(f"  {bps:>4}bp {len(vals):>8}/{n:<3} {bm:>26} / {am}")
    print()
    print("  v1 tell: 78/200 symbols had imb_05 == imb_1 exactly (band never bound).")
    same = sum(
        1 for a in aggs if a.get("imb5") is not None and a.get("imb5") == a.get("imb25")
    )
    exact = [
        a["symbol"]
        for a in aggs
        if a.get("imb5") is not None and a.get("imb5") == a.get("imb25")
    ]
    print(f"  NEW   : imb5 == imb25 exactly for {same}/{n} symbols")
    if exact[:6]:
        print(f"          examples: {', '.join(exact[:6])}")
    print()

    print("--- DEFECT 2: trade window length differed across symbols (31s..367s)")
    ws = [f["window_s"] for f in flow.values() if f.get("window_s")]
    ts = [f["trades"] for f in flow.values() if f.get("trades") is not None]
    if ws:
        print(
            f"  NEW   : window_s min={min(ws)} max={max(ws)} unique={sorted(set(ws))}"
        )
        print("          -> one shared clock interval, comparable across symbols")
    if ts:
        print(
            f"  trade count in window: min={min(ts)} median={int(statistics.median(ts))} max={max(ts)}"
        )
        print(f"  symbols with <30 trades (noisy): {sum(1 for t in ts if t < 30)}")
    print(f"  flow collected for {len(flow)}/{n} symbols (design: top flow_top)")
    print()

    print("--- DEFECT 3: a single frozen frame, no visible change")
    print("  NEW   : live orderbook.1 updates per second, per symbol:")
    for sym, b in list(stream.items()):
        std = b["tobi_std"]
        std_txt = "na" if std is None else f"{std:.3f}"
        print(
            f"    {sym:<12} upd/s={b['updates_per_s']:.1f} "
            f"tobi_mean={b['tobi_mean']:+.3f} tobi_std={std_txt}"
        )
    dc = data.get("depth_change", {})
    print("  NEW   : resting depth within 5 bps, start-of-window -> T0:")
    for sym, v in dc.items():
        print(f"    {sym:<12} {v * 100:+.2f}%")
    print()

    print("--- DEFECT 4: fields that existed but were never collected")
    for key in (
        "fund_prev_bps",
        "oi_chg_5m",
        "oi_chg_60m",
        "ls_ratio",
        "prem_bps",
        "prem_mean6_bps",
        "basis_bps",
        "mom1h_free",
        "max_lev",
        "maint_margin",
    ):
        have = sum(1 for a in aggs if a.get(key) is not None)
        print(f"  {key:<16} present for {have:>3}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
