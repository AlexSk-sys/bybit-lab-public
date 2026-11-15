"""Freeze the agent's TSV decisions into the calls JSON the evaluator parses.

The point of this script is *validation and a timestamp*, not analysis:
  * every symbol in the collector's universe must have exactly one call
  * no call may reference a symbol that is not in the universe
  * created_utc is stamped at the moment of freezing, so the very act of
    freezing is auditable and the calls cannot be back-fitted to outcomes.

Usage:  python freeze_calls.py micro2/collect_<stamp>.json micro2/<calls>.tsv
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

CALLS = ("up", "down")
CONF = ("high", "medium", "low")


def main() -> int:
    collect_path, tsv_path = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads(collect_path.read_text(encoding="utf-8"))
    universe = [a["symbol"] for a in data["aggregates"]]

    lines = tsv_path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    if header[:4] != ["symbol", "call", "confidence", "p_up"]:
        print(f"bad header: {header!r}", file=sys.stderr)
        return 2

    calls: list[dict] = []
    seen: set[str] = set()
    bad: list[str] = []
    for n, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) < 5:
            bad.append(f"line {n}: only {len(parts)} fields")
            continue
        symbol, call, conf, p_up, rationale = (
            parts[0],
            parts[1],
            parts[2],
            parts[3],
            parts[4],
        )
        if call not in CALLS:
            bad.append(f"line {n}: call={call!r}")
        if conf not in CONF:
            bad.append(f"line {n}: confidence={conf!r}")
        try:
            pu = float(p_up)
        except ValueError:
            bad.append(f"line {n}: p_up={p_up!r}")
            pu = 0.5
        if symbol in seen:
            bad.append(f"line {n}: duplicate {symbol}")
        seen.add(symbol)
        calls.append(
            {
                "symbol": symbol,
                "call": call,
                "confidence": conf,
                "p_up": pu,
                "rationale": rationale,
            }
        )

    missing = [s for s in universe if s not in seen]
    extra = [s for s in seen if s not in universe]
    if bad or missing or extra:
        print("VALIDATION FAILED", file=sys.stderr)
        for b in bad[:40]:
            print(f"  {b}", file=sys.stderr)
        if missing:
            print(f"  missing {len(missing)}: {missing[:40]}", file=sys.stderr)
        if extra:
            print(f"  extra {len(extra)}: {extra[:40]}", file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc)
    out = {
        "created_utc": stamp.isoformat(),
        "snapshot_utc": data["meta"]["created_utc"],
        "source_tsv": tsv_path.name,
        "collect": collect_path.name,
        "n": len(calls),
        "calls": calls,
    }
    dest = tsv_path.with_suffix(".json")
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"frozen {len(calls)} calls -> {dest}")
    print(f"created_utc = {stamp.isoformat()}")
    print(f"snapshot    = {data['meta']['created_utc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
