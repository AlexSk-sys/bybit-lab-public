"""Honest read of the v2 horizon curve.

The only question that matters: at any horizon, does the agent beat the BEST
constant rule (always-up or always-down)? A curve that hovers near 50% while both
constants sit far away means zero skill, no matter how nice the middle looks.

Also checks whether confidence was informative (it can be worse than useless) and
whether the Brier score beats the optimal constant forecast, which has a known
floor of up_rate*(1-up_rate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    rep = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    curve = {int(k): v for k, v in rep["horizon_curve"].items()}
    rows = rep["rows"]

    print(
        f"origin {rep['origin_utc']}  snapshot {rep['snapshot_utc']}  stale {rep['stale_min']:.1f} min"
    )
    print()

    edges = []
    print("min   n   mine  bestconst  which      edge     brier  brier_opt")
    for m in sorted(curve):
        c = curve[m]
        best = max(c["always_up"], c["always_down"])
        which = "up" if c["always_up"] >= c["always_down"] else "down"
        edge = c["mine"] - best
        edges.append(edge)
        bopt = c["up_rate"] * (1 - c["up_rate"])
        print(
            f"{m:3d} {c['n']:4d}  {c['mine']:5.1%}  {best:8.1%}  {which:>4}  "
            f"{edge:+7.1%}  {c['brier']:.4f}  {bopt:.4f}"
        )

    print()
    print(
        f"edge  min {min(edges):+.1%}  max {max(edges):+.1%}  mean {sum(edges) / len(edges):+.1%}"
    )
    print(
        f"horizons where I beat the best constant: {sum(1 for e in edges if e > 0)}/{len(edges)}"
    )

    # how often did I say up, and how did each direction do
    calls = [r for r in rows if r.get("call_15m")]
    up_calls = sum(1 for r in calls if r["call_15m"] == "up")
    print()
    print(
        f"calls: up {up_calls}/{len(calls)} ({up_calls / len(calls):.1%})  down {len(calls) - up_calls}"
    )
    for h in ("15m", "60m"):
        act_up = [r for r in calls if r[f"actual_{h}"] == "up"]
        act_dn = [r for r in calls if r[f"actual_{h}"] == "down"]
        mine_up = [r for r in calls if r["call_15m"] == "up"]
        mine_dn = [r for r in calls if r["call_15m"] == "down"]
        co = sum(1 for r in mine_up if r[f"hit_{h}"])
        cd = sum(1 for r in mine_dn if r[f"hit_{h}"])
        print(
            f"{h}: actual up {len(act_up)}/{len(calls)} ({len(act_up) / len(calls):.1%})  |  "
            f"my up-calls correct {co}/{len(mine_up)} ({co / len(mine_up):.1%})  "
            f"my down-calls correct {cd}/{len(mine_dn)} ({cd / len(mine_dn):.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
