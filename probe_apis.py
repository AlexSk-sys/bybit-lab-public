"""Probe which Bybit V5 market endpoints exist and what the collector did NOT use."""

from __future__ import annotations

import json

import requests

BASE = "https://api.bybit.com"
S = requests.Session()
S.headers.update({"Accept": "application/json"})


def probe(label: str, path: str, params: dict, show_keys: bool = True) -> None:
    try:
        r = S.get(BASE + path, params=params, timeout=25)
        p = r.json()
    except Exception as exc:  # noqa: BLE001
        print(f"[----] {label:<34} EXC {exc}")
        return
    code = p.get("retCode")
    if code != 0:
        print(f"[{code:>4}] {label:<34} {p.get('retMsg')}")
        return
    res = p["result"]
    if isinstance(res, list):
        n = len(res)
        keys = sorted(res[0].keys()) if res and isinstance(res[0], dict) else []
        print(f"[{code:>4}] {label:<34} list n={n} keys={keys if show_keys else ''}")
        return
    if isinstance(res, dict):
        keys = sorted(res.keys())
        print(f"[{code:>4}] {label:<34} dict keys={keys}")
        for k in ("list", "b", "a"):
            v = res.get(k)
            if isinstance(v, list):
                print(f"         .{k}: n={len(v)} sample={v[0] if v else None}")
        return
    print(f"[{code:>4}] {label:<34} {type(res).__name__} {str(res)[:120]}")


print("=== A. endpoints the collector does NOT use ===")
probe(
    "instruments-info (linear)",
    "/v5/market/instruments-info",
    {"category": "linear", "symbol": "BTCUSDT"},
)
probe(
    "mark-price-kline",
    "/v5/market/mark-price-kline",
    {"category": "linear", "symbol": "BTCUSDT", "interval": "5", "limit": 2},
)
probe(
    "index-price-kline",
    "/v5/market/index-price-kline",
    {"category": "linear", "symbol": "BTCUSDT", "interval": "5", "limit": 2},
)
probe(
    "premium-index-kline",
    "/v5/market/premium-index-price-kline",
    {"category": "linear", "symbol": "BTCUSDT", "interval": "5", "limit": 2},
)
probe(
    "funding/history",
    "/v5/market/funding/history",
    {"category": "linear", "symbol": "BTCUSDT", "limit": 2},
)
probe(
    "risk-limit", "/v5/market/risk-limit", {"category": "linear", "symbol": "BTCUSDT"}
)
probe("insurance", "/v5/market/insurance", {"coin": "BTC"})
probe("time", "/v5/market/time", {})
probe(
    "long-short-ratio (top tra.)",
    "/v5/market/long-short-ratio",
    {"category": "linear", "symbol": "BTCUSDT", "period": "5min", "limit": 2},
)
probe(
    "liquidation (public)",
    "/v5/market/liquidation",
    {"category": "linear", "symbol": "BTCUSDT", "limit": 5},
)
probe(
    "historical-volatility",
    "/v5/market/historical-volatility",
    {"category": "option", "symbol": "BTCUSDT"},
)
probe(
    "position-info (private)",
    "/v5/position/list",
    {"category": "linear", "symbol": "BTCUSDT"},
)

print()
print("=== B. options universe (category=option) ===")
probe(
    "tickers option (all)",
    "/v5/market/tickers",
    {"category": "option"},
    show_keys=False,
)

print()
print("=== C. how much time does a 500-trade window cover? ===")
for sym in ("BTCUSDT", "ETHUSDT", "ENAUSDT", "1000PEPEUSDT"):
    try:
        res = S.get(
            BASE + "/v5/market/recent-trade",
            params={"category": "linear", "symbol": sym, "limit": 1000},
            timeout=25,
        ).json()["result"]
        rows = res["list"]
        if not rows:
            print(f"  {sym:<14} no trades")
            continue
        ts = [int(t["time"]) for t in rows]
        span_ms = max(ts) - min(ts)
        span_s = span_ms / 1000
        rate = len(rows) / span_s if span_s > 0 else float("inf")
        print(
            f"  {sym:<14} n={len(rows):<5} span={span_s:>8.2f}s  ~{rate:>9.1f} trades/s  -> 500 trades = {500 / rate if rate else 0:>7.2f}s"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {sym:<14} EXC {exc}")

print()
print("=== D. orderbook depth: does limit=50 truncate the +/-1% band? ===")
for sym in ("BTCUSDT", "SOLUSDT", "ENAUSDT"):
    for lim in (50, 200, 500):
        try:
            res = S.get(
                BASE + "/v5/market/orderbook",
                params={"category": "linear", "symbol": sym, "limit": lim},
                timeout=25,
            ).json()["result"]
            bids = [(float(p), float(q)) for p, q in res["b"]]
            asks = [(float(p), float(q)) for p, q in res["a"]]
            mid = (bids[0][0] + asks[0][0]) / 2
            cov_b = (mid - bids[-1][0]) / mid * 100
            cov_a = (asks[-1][0] - mid) / mid * 100
            print(
                f"  {sym:<10} limit={lim:<4} levels b/a={len(bids)}/{len(asks)}  covers -{cov_b:.3f}% / +{cov_a:.3f}%"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {sym:<10} limit={lim:<4} EXC {exc}")

print()
print("=== E. fields available in bulk linear tickers (1 call, all symbols) ===")
res = S.get(
    BASE + "/v5/market/tickers",
    params={"category": "linear", "symbol": "BTCUSDT"},
    timeout=25,
).json()["result"]["list"][0]
print(json.dumps(res, indent=1)[:1200])
