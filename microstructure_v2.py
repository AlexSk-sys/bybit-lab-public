"""Microstructure sensor v2 -- calibrated bands, real flow window, basis and funding term structure.

What v1 got wrong, and how v2 fixes it:

  1. BANDS      v1 asked for a +/-1% depth band from a 50-level book. For BTCUSDT those
                50 levels reach only +/-0.01%, so the band never bound and 78/200 symbols
                produced identical +/-0.5% and +/-1% numbers. v2 fetches 500 levels, measures
                depth at fixed offsets in *basis points*, and records the book's actual
                coverage so every band can be marked valid or "not measurable".
  2. FLOW       v1 used /v5/market/recent-trade?limit=500, which spans 31 seconds for BTCUSDT
                and 367 seconds for 1000PEPEUSDT -- not comparable across symbols, and only
                collected for 30 of 200 symbols. v2 subscribes to the public WebSocket and
                measures taker flow over one fixed clock window for every symbol.
  3. SNAPSHOT   v1 saw a single frozen frame. v2 watches the top of book live (update count,
                top-of-book imbalance mean/sigma) and compares resting depth before and after
                the window, so liquidity *change* is visible, not just its level.
  4. MISSING    v2 adds the index/mark basis, the premium-index term structure, funding-rate
                history, longer open-interest and long/short history, and risk-limit tiers.

Ordering matters: the flow window runs FIRST, the structure snapshot runs SECOND, and T0 is
the moment of that snapshot. Everything the analyst sees is therefore history relative to T0
-- no look-ahead.

Usage:
    python microstructure_v2.py collect [--top 200] [--flow-top 40] [--book-top 5] [--window 120]
    python microstructure_v2.py evaluate <collect.json> <calls.json> [--desktop DIR]

Research instrument. Not investment advice.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from websockets.sync.client import connect

BASE_URL = "https://api.bybit.com"
WS_URL = "wss://stream.bybit.com/v5/public/linear"

ORDERBOOK = "/v5/market/orderbook"
KLINE = "/v5/market/kline"
TICKERS = "/v5/market/tickers"
ACCOUNT_RATIO = "/v5/market/account-ratio"
OPEN_INTEREST = "/v5/market/open-interest"
FUNDING_HISTORY = "/v5/market/funding/history"
PREMIUM_KLINE = "/v5/market/premium-index-price-kline"
RISK_LIMIT = "/v5/market/risk-limit"
RECENT_TRADE = "/v5/market/recent-trade"

TIMEOUT = 25
MAX_RETRIES = 6
WORKERS = 10
BOOK_LEVELS = 500
BANDS_BPS = (2, 5, 10, 25, 50, 100)
DEPTH_CHANGE_BPS = 5.0
WALL_BPS = 25.0

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"Accept": "application/json"})
        _local.session = s
    return s


def api(path: str, params: dict) -> dict:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = _session().get(BASE_URL + path, params=params, timeout=TIMEOUT)
            if r.status_code == 429:
                time.sleep(0.8 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            code = payload.get("retCode")
            if code == 10006:
                time.sleep(0.8 * (attempt + 1))
                continue
            if code != 0:
                raise RuntimeError(f"{code}: {payload.get('retMsg')}")
            return payload["result"]
        except Exception as exc:  # noqa: BLE001 - retry, then surface
            last = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"failed {path} {params}: {last}")


def _f(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def chunked(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# --------------------------------------------------------------------------- flow window


def run_flow_window(
    trade_syms: list[str], book_syms: list[str], seconds: int
) -> tuple[dict, dict, float]:
    """Subscribe to publicTrade (all trade_syms) and orderbook.1 (book_syms) for `seconds`."""
    flow = {
        s: {
            "n": 0,
            "buy_n": 0,
            "sell_n": 0,
            "buy_v": 0.0,
            "sell_v": 0.0,
            "notional": 0.0,
            "max_usd": 0.0,
        }
        for s in trade_syms
    }
    book = {s: {"updates": 0, "tobi_sum": 0.0, "tobi_sq": 0.0} for s in book_syms}
    args = [f"publicTrade.{s}" for s in trade_syms] + [
        f"orderbook.1.{s}" for s in book_syms
    ]

    print(
        f"  flow window: {seconds}s, {len(trade_syms)} trade topics, {len(book_syms)} book topics",
        flush=True,
    )
    with connect(WS_URL, open_timeout=20, close_timeout=5, max_size=2**24) as ws:
        for chunk in chunked(args, 10):
            ws.send(json.dumps({"op": "subscribe", "args": chunk}))
        # Start the clock only after subscriptions are in place, so the reported window is the
        # interval actually observed and updates/s is not diluted by connection setup.
        started = time.time()
        deadline = started + seconds
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                raw = ws.recv(timeout=min(1.0, remaining))
            except TimeoutError:
                continue
            msg = json.loads(raw)
            op = msg.get("op")
            if op == "ping":
                ws.send(json.dumps({"op": "pong"}))
                continue
            if op == "subscribe" or "topic" not in msg:
                continue
            topic = msg["topic"]
            data = msg.get("data")
            if topic.startswith("publicTrade."):
                sym = topic.split(".", 1)[1]
                rec = flow.get(sym)
                if rec is None or not isinstance(data, list):
                    continue
                for t in data:
                    side = t.get("S")
                    size = _f(t.get("v")) or 0.0
                    price = _f(t.get("p")) or 0.0
                    usd = size * price
                    rec["n"] += 1
                    rec["notional"] += usd
                    rec["max_usd"] = max(rec["max_usd"], usd)
                    if side == "Buy":
                        rec["buy_n"] += 1
                        rec["buy_v"] += size
                    else:
                        rec["sell_n"] += 1
                        rec["sell_v"] += size
            elif topic.startswith("orderbook.1."):
                sym = topic.rsplit(".", 1)[1]
                rec = book.get(sym)
                if rec is None or not isinstance(data, dict):
                    continue
                b = data.get("b") or []
                a = data.get("a") or []
                if not b or not a:
                    continue
                bq, aq = _f(b[0][1]) or 0.0, _f(a[0][1]) or 0.0
                if bq + aq <= 0:
                    continue
                tobi = (bq - aq) / (bq + aq)
                rec["updates"] += 1
                rec["tobi_sum"] += tobi
                rec["tobi_sq"] += tobi * tobi
        # Measured before leaving the context manager: closing the socket performs a
        # handshake that can block for close_timeout and would inflate the window.
        elapsed = time.time() - started
    return flow, book, elapsed


def summarise_flow(flow: dict, book: dict, elapsed: float) -> tuple[dict, dict]:
    flow_out: dict[str, dict] = {}
    for sym, r in flow.items():
        tot_v = r["buy_v"] + r["sell_v"]
        tot_n = r["buy_n"] + r["sell_n"]
        flow_out[sym] = {
            "window_s": round(elapsed, 1),
            "trades": r["n"],
            "buy_vol_ratio": (r["buy_v"] / tot_v) if tot_v > 0 else None,
            "buy_count_ratio": (r["buy_n"] / tot_n) if tot_n > 0 else None,
            "avg_trade_usd": (r["notional"] / r["n"]) if r["n"] else None,
            "max_trade_usd": r["max_usd"] or None,
            "notional_usd": r["notional"],
        }
    book_out: dict[str, dict] = {}
    for sym, r in book.items():
        n = r["updates"]
        mean = r["tobi_sum"] / n if n else None
        var = (r["tobi_sq"] / n - mean * mean) if n and mean is not None else None
        book_out[sym] = {
            "updates": n,
            "updates_per_s": (n / elapsed) if elapsed > 0 else None,
            "tobi_mean": mean,
            "tobi_std": math.sqrt(var) if var is not None and var > 0 else None,
        }
    return flow_out, book_out


# --------------------------------------------------------------------------- snapshot layer


def depth_within(
    levels: list[tuple[float, float]], mid: float, bps: float, upper: bool
) -> float:
    thr = mid * (1 + bps / 1e4) if upper else mid * (1 - bps / 1e4)
    return sum(p * q for p, q in levels if (p <= thr if upper else p >= thr))


def largest_level(
    levels: list[tuple[float, float]], mid: float, within_bps: float, upper: bool
) -> tuple[float | None, float | None]:
    thr = mid * (1 + within_bps / 1e4) if upper else mid * (1 - within_bps / 1e4)
    sel = [(p, q) for p, q in levels if (p <= thr if upper else p >= thr)]
    if not sel:
        return None, None
    p, q = max(sel, key=lambda x: x[1])
    dist = ((p - mid) if upper else (mid - p)) / mid * 1e4
    return dist, p * q


def compute_snapshot(
    sym: dict,
    ob: dict,
    kl: list,
    fund: list,
    oi: list,
    ls: list,
    prem: list,
    risk: list,
) -> dict | None:
    bids = [(float(p), float(q)) for p, q in ob.get("b", [])]
    asks = [(float(p), float(q)) for p, q in ob.get("a", [])]
    if not bids or not asks:
        return None
    last_price = _f(sym.get("lastPrice"))
    bb, ba = bids[0][0], asks[0][0]
    mid = (bb + ba) / 2
    if mid <= 0:
        return None

    cov_bps = min(
        (mid - bids[-1][0]) / mid * 1e4,
        (asks[-1][0] - mid) / mid * 1e4,
    )

    rec: dict = {
        "symbol": sym["symbol"],
        "last_price": last_price,
        "mid": mid,
        "spread_bps": (ba - bb) / mid * 1e4,
        "cov_bps": cov_bps,
        "turnover24h": _f(sym.get("turnover24h")),
    }

    for bps in BANDS_BPS:
        ok = cov_bps >= bps
        bd = depth_within(bids, mid, bps, upper=False) if ok else None
        ad = depth_within(asks, mid, bps, upper=True) if ok else None
        rec[f"dep{bps}_bid"] = bd
        rec[f"dep{bps}_ask"] = ad
        rec[f"imb{bps}"] = (
            ((bd - ad) / (bd + ad)) if (ok and bd and ad and bd + ad > 0) else None
        )

    wb_dist, wb_usd = largest_level(bids, mid, WALL_BPS, upper=False)
    wa_dist, wa_usd = largest_level(asks, mid, WALL_BPS, upper=True)
    rec["wall_bid_bps"] = wb_dist
    rec["wall_bid_usd"] = wb_usd
    rec["wall_ask_bps"] = wa_dist
    rec["wall_ask_usd"] = wa_usd

    b1, a1 = _f(sym.get("bid1Size")), _f(sym.get("ask1Size"))
    rec["tobi"] = (
        ((b1 - a1) / (b1 + a1))
        if (b1 is not None and a1 is not None and b1 + a1 > 0)
        else None
    )

    close = np.array([float(r[4]) for r in reversed(kl)], dtype=float)
    high = np.array([float(r[2]) for r in reversed(kl)], dtype=float)
    low = np.array([float(r[3]) for r in reversed(kl)], dtype=float)
    vol = np.array([float(r[5]) for r in reversed(kl)], dtype=float)

    def ret(bars: int) -> float | None:
        return float(close[-1] / close[-1 - bars] - 1) if len(close) > bars else None

    rec["ret_15m"] = ret(3)
    rec["ret_1h"] = ret(12)
    rec["ret_3h"] = ret(36)

    if len(close) > 15:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        rec["atr_pct"] = float(tr[-14:].mean() / close[-1] * 100)
    else:
        rec["atr_pct"] = None

    if len(close) > 13:
        logret = np.diff(np.log(close))
        rec["rvol_1h_pct"] = float(logret[-12:].std() * math.sqrt(12) * 100)
    else:
        rec["rvol_1h_pct"] = None

    rec["vol_ratio"] = (
        float(vol[-1] / vol[-20:].mean())
        if len(vol) >= 20 and vol[-20:].mean() > 0
        else None
    )

    prev1h = _f(sym.get("prevPrice1h"))
    rec["mom1h_free"] = (last_price / prev1h - 1) if (last_price and prev1h) else None

    mark, index = _f(sym.get("markPrice")), _f(sym.get("indexPrice"))
    rec["basis_bps"] = ((mark / index - 1) * 1e4) if (mark and index) else None

    fr = _f(sym.get("fundingRate"))
    rec["fund_bps"] = fr * 1e4 if fr is not None else None
    rates = [_f(r.get("fundingRate")) for r in fund]
    rates = [r for r in rates if r is not None]
    if rates:
        rec["fund_prev_bps"] = rates[0] * 1e4
        rec["fund_mean3_bps"] = float(np.mean(rates[:3])) * 1e4
    rec["fund_next_ms"] = _f(sym.get("nextFundingTime"))

    oi_vals = [_f(r.get("openInterest")) for r in oi]
    oi_vals = [v for v in oi_vals if v]
    rec["oi_usd"] = _f(sym.get("openInterestValue"))
    if len(oi_vals) >= 2:
        rec["oi_chg_5m"] = (oi_vals[0] / oi_vals[1] - 1) * 100
    if len(oi_vals) >= 13:
        rec["oi_chg_60m"] = (oi_vals[0] / oi_vals[12] - 1) * 100

    if ls:
        ratios = []
        for row in ls:
            br, sr = _f(row.get("buyRatio")), _f(row.get("sellRatio"))
            if br and sr:
                ratios.append(br / sr)
        if ratios:
            rec["ls_ratio"] = ratios[0]
            rec["ls_mean3"] = float(np.mean(ratios[:3]))
            rec["ls_chg"] = ratios[0] - rec["ls_mean3"]

    prem_vals = [_f(r[4]) for r in prem]
    prem_vals = [v for v in prem_vals if v is not None]
    if prem_vals:
        rec["prem_bps"] = prem_vals[-1] * 1e4
        rec["prem_mean6_bps"] = float(np.mean(prem_vals[-6:])) * 1e4

    if risk:
        rec["max_lev"] = _f(risk[0].get("maxLeverage"))
        rec["maint_margin"] = _f(risk[0].get("maintenanceMargin"))
    return rec


def snapshot_symbol(sym: dict, want_risk: bool = False) -> dict | None:
    s = sym["symbol"]
    snap_ms = int(time.time() * 1000)
    ob = api(ORDERBOOK, {"category": "linear", "symbol": s, "limit": BOOK_LEVELS})
    kl = api(KLINE, {"category": "linear", "symbol": s, "interval": "5", "limit": 60})[
        "list"
    ]
    fund = api(FUNDING_HISTORY, {"category": "linear", "symbol": s, "limit": 8})["list"]
    oi = api(
        OPEN_INTEREST,
        {"category": "linear", "symbol": s, "intervalTime": "5min", "limit": 13},
    )["list"]
    ls = api(
        ACCOUNT_RATIO,
        {"category": "linear", "symbol": s, "period": "5min", "limit": 13},
    )["list"]
    prem = api(
        PREMIUM_KLINE, {"category": "linear", "symbol": s, "interval": "5", "limit": 12}
    )["list"]
    risk = (
        api(RISK_LIMIT, {"category": "linear", "symbol": s})["list"]
        if want_risk
        else []
    )
    rec = compute_snapshot(sym, ob, kl, fund, oi, ls, prem, risk)
    if rec is not None:
        rec["snap_ms"] = snap_ms
    return rec


# --------------------------------------------------------------------------- collect


def _fmt(v, digits=2, sign=False) -> str:
    if v is None:
        return "na"
    return f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"


def render_table(
    aggs: list[dict], flow: dict, book: dict, depth_change: dict, meta: dict
) -> str:
    lines = [
        "MICROSTRUCTURE SENSOR v2 -- calibrated",
        f"snapshot_utc (T0): {meta['created_utc']}",
        f"flow window: {meta['window_s']}s ending at T0  |  book levels: {meta['book_levels']}",
        f"universe: top {meta['top']} USDT perps by 24h turnover",
        "",
        "columns:",
        "  cov    : how far the fetched book actually reaches, in bps (if < band, the band is 'na')",
        "  imb5/25: (bid-ask)/(bid+ask) of resting USD within 5 / 25 bps of mid, -1..1",
        "  dep5M  : resting USD within 5 bps, millions, bid/ask",
        "  wallB/A: distance(bps) and size($M) of the single largest level within 25 bps",
        "  tobi   : top-of-book size imbalance from the bulk ticker call, -1..1",
        "  fund   : funding bps/8h | mean of last 3 settlements",
        "  oi5/60 : open-interest change over 5 and 60 minutes, %",
        "  L/S    : long/short account ratio | change vs its own 3-period mean",
        "  prem   : premium-index (basis) bps | basis from mark vs index bps",
        "  atr/rv : 5m ATR% | 1h realised vol %",
        "  r15/1h/3h : returns over 15m / 60m / 180m, %",
        "",
    ]
    header = (
        f"{'symbol':<14}{'mid':>11}{'sprd':>6}{'cov':>7}{'imb5':>7}{'imb25':>7}"
        f"{'dep5M b/a':>13}{'wallB':>13}{'wallA':>13}{'tobi':>7}"
        f"{'fund':>7}{'f3m':>7}{'oi5%':>7}{'oi60%':>8}{'L/S':>6}{'lsD':>6}"
        f"{'prem':>7}{'basis':>7}{'atr%':>6}{'rv%':>6}{'r15%':>7}{'r1h%':>7}{'r3h%':>7}"
    )
    lines.append(header)

    def key(a: dict):
        return -(a.get("turnover24h") or 0)

    for a in sorted(aggs, key=key):
        dep = f"{_fmt((a.get('dep5_bid') or 0) / 1e6, 1)}/{_fmt((a.get('dep5_ask') or 0) / 1e6, 1)}"
        wb = f"{_fmt(a.get('wall_bid_bps'), 1)}/{_fmt((a.get('wall_bid_usd') or 0) / 1e6, 1)}"
        wa = f"{_fmt(a.get('wall_ask_bps'), 1)}/{_fmt((a.get('wall_ask_usd') or 0) / 1e6, 1)}"
        lines.append(
            f"{a['symbol']:<14}{_fmt(a['mid'], 6 if a['mid'] < 5 else 2):>11}{_fmt(a['spread_bps'], 1):>6}"
            f"{_fmt(a['cov_bps'], 0):>7}{_fmt(a.get('imb5'), 3, True):>7}{_fmt(a.get('imb25'), 3, True):>7}"
            f"{dep:>13}{wb:>13}{wa:>13}{_fmt(a.get('tobi'), 3, True):>7}"
            f"{_fmt(a.get('fund_bps'), 2, True):>7}{_fmt(a.get('fund_mean3_bps'), 2, True):>7}"
            f"{_fmt(a.get('oi_chg_5m'), 2, True):>7}{_fmt(a.get('oi_chg_60m'), 2, True):>8}"
            f"{_fmt(a.get('ls_ratio'), 2):>6}{_fmt(a.get('ls_chg'), 2, True):>6}"
            f"{_fmt(a.get('prem_bps'), 1, True):>7}{_fmt(a.get('basis_bps'), 1, True):>7}"
            f"{_fmt(a.get('atr_pct'), 2):>6}{_fmt(a.get('rvol_1h_pct'), 2):>6}"
            f"{_fmt((a.get('ret_15m') or 0) * 100, 2, True):>7}{_fmt((a.get('ret_1h') or 0) * 100, 2, True):>7}"
            f"{_fmt((a.get('ret_3h') or 0) * 100, 2, True):>7}"
        )

    lines += [
        "",
        "=" * 118,
        f"FLOW TABLE (live WebSocket, one fixed {meta['window_s']}s window ending at T0)",
        "=" * 118,
        "  buy% : share of taker volume that was BUY (side=S), comparable across symbols because the window is the same clock interval",
        "  n    : trades observed in the window    avg$/max$ : mean and largest trade notional",
        "  upd/s: orderbook.1 updates per second (quote churn)   tobi_m/s : top-of-book imbalance mean and sigma over the window",
        "  d5%  : change in resting USD within 5 bps between window start and T0, %",
        "",
        f"{'symbol':<14}{'buy%':>7}{'n':>8}{'avg$':>10}{'max$':>12}{'notM$':>9}{'upd/s':>8}{'tobi_m':>8}{'tobi_s':>8}{'d5%':>8}",
    ]
    for sym, f in sorted(
        flow.items(), key=lambda kv: -(kv[1].get("notional_usd") or 0)
    ):
        b = book.get(sym, {})
        dc = depth_change.get(sym)
        lines.append(
            f"{sym:<14}{_fmt((f.get('buy_vol_ratio') or 0) * 100, 1):>7}{f.get('trades', 0):>8}"
            f"{_fmt(f.get('avg_trade_usd'), 0):>10}{_fmt(f.get('max_trade_usd'), 0):>12}"
            f"{_fmt((f.get('notional_usd') or 0) / 1e6, 2):>9}"
            f"{_fmt(b.get('updates_per_s'), 1):>8}{_fmt(b.get('tobi_mean'), 3, True):>8}{_fmt(b.get('tobi_std'), 3):>8}"
            f"{_fmt((dc or 0) * 100, 2, True) if dc is not None else 'na':>8}"
        )
    return "\n".join(lines)


def cmd_collect(args: argparse.Namespace) -> int:
    all_rows = api(TICKERS, {"category": "linear"})["list"]
    liquid = [
        x
        for x in all_rows
        if x["symbol"].endswith("USDT") and (_f(x.get("turnover24h")) or 0) > 0
    ]
    liquid.sort(key=lambda x: _f(x.get("turnover24h")) or 0, reverse=True)
    universe = liquid[: args.top]
    symbols = [x["symbol"] for x in universe]
    trade_syms = symbols[: args.flow_top]
    book_syms = symbols[: args.book_top]
    print(f"universe: {len(universe)} of {len(liquid)} liquid USDT perps", flush=True)

    # --- step 1: pre-window book snapshot, to be able to measure depth CHANGE across the window
    pre_depth: dict[str, float] = {}
    for s in book_syms:
        try:
            ob = api(
                ORDERBOOK, {"category": "linear", "symbol": s, "limit": BOOK_LEVELS}
            )
            bids = [(float(p), float(q)) for p, q in ob["b"]]
            asks = [(float(p), float(q)) for p, q in ob["a"]]
            mid = (bids[0][0] + asks[0][0]) / 2
            cov = min((mid - bids[-1][0]) / mid * 1e4, (asks[-1][0] - mid) / mid * 1e4)
            if cov >= DEPTH_CHANGE_BPS:
                pre_depth[s] = depth_within(
                    bids, mid, DEPTH_CHANGE_BPS, upper=False
                ) + depth_within(asks, mid, DEPTH_CHANGE_BPS, upper=True)
            else:
                print(
                    f"  pre-snapshot {s}: book covers only {cov:.0f} bps, "
                    f"depth change not measurable",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  pre-snapshot {s} failed: {exc}", file=sys.stderr)

    # --- step 2: flow window (runs BEFORE the snapshot, so it is history relative to T0)
    flow_raw, book_raw, elapsed = run_flow_window(trade_syms, book_syms, args.window)
    flow, book = summarise_flow(flow_raw, book_raw, elapsed)

    # --- step 3: structure snapshot.
    tickers = api(TICKERS, {"category": "linear"})["list"]
    by_symbol = {x["symbol"]: x for x in tickers if x["symbol"] in set(symbols)}
    risk_syms = set(symbols[: args.risk_top])

    aggregates: list[dict] = []
    post_depth: dict[str, float] = {}
    failures: list[str] = []
    snaps: list[int] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(snapshot_symbol, sym, sym["symbol"] in risk_syms): sym["symbol"]
            for sym in universe
            if sym["symbol"] in by_symbol
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            sym_name = futures[fut]
            try:
                rec = fut.result()
                if rec:
                    aggregates.append(rec)
                    if rec.get("snap_ms"):
                        snaps.append(rec["snap_ms"])
                    if sym_name in set(book_syms):
                        b5 = rec.get(f"dep{int(DEPTH_CHANGE_BPS)}_bid")
                        a5 = rec.get(f"dep{int(DEPTH_CHANGE_BPS)}_ask")
                        if b5 is not None and a5 is not None:
                            post_depth[sym_name] = b5 + a5
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{sym_name}: {exc}")
            if done % 25 == 0:
                print(f"  snapshot {done}/{len(futures)}", flush=True)

    # T0 is defined AFTER the snapshot layer, so every structure reading is history relative
    # to T0 and the calls cannot see the future. The price paid is staleness, which is
    # reported as snap_span_s rather than hidden.
    baseline_rows = api(TICKERS, {"category": "linear"})["list"]
    baseline = {
        x["symbol"]: _f(x.get("lastPrice"))
        for x in baseline_rows
        if x["symbol"] in set(symbols) and _f(x.get("lastPrice"))
    }
    snap_span = (max(snaps) - min(snaps)) / 1000 if len(snaps) > 1 else None
    created = datetime.now(timezone.utc)
    span_txt = f"  (structure span {snap_span:.0f}s, stale by that much at most)"
    print(f"T0 = {created.isoformat()}{span_txt if snap_span else ''}", flush=True)

    depth_change = {
        s: (post_depth[s] / pre_depth[s] - 1) for s in post_depth if pre_depth.get(s)
    }

    meta = {
        "created_utc": created.isoformat(),
        "top": args.top,
        "flow_top": args.flow_top,
        "book_top": args.book_top,
        "risk_top": args.risk_top,
        "window_s": args.window,
        "book_levels": BOOK_LEVELS,
        "bands_bps": list(BANDS_BPS),
        "snap_span_s": round(snap_span, 1) if snap_span is not None else None,
        "universe_total": len(liquid),
        "collected": len(aggregates),
        "failures": failures,
    }
    payload = {
        "meta": meta,
        "aggregates": aggregates,
        "baseline": baseline,
        "flow": flow,
        "book_stream": book,
        "depth_change": depth_change,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"collect_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    table = render_table(aggregates, flow, book, depth_change, meta)
    table_path = out_dir / f"table_{stamp}.txt"
    table_path.write_text(table, encoding="utf-8")

    n_trunc = sum(1 for a in aggregates if (a.get("cov_bps") or 0) < 25)
    print(
        f"collected: {len(aggregates)} symbols, {len(flow)} flow, {len(book)} stream books, {len(failures)} failures"
    )
    print(f"books not reaching 25 bps: {n_trunc}")
    print(f"json : {json_path}")
    print(f"table: {table_path}")
    for f in failures[:8]:
        print(f"  failure: {f}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- evaluate


def _price_at(rows: list[list], target_ms: int) -> float | None:
    """Open of the 1-minute bar that contains `target_ms` (rows ascending by openTime).

    The bar's OPEN is used, not its close, so a horizon of N minutes can never borrow price
    action from minute N+1. Time-based rather than position-based, so a late or early
    evaluation run cannot shift the horizon. Returns None while the bar does not exist yet,
    which scores as 'pend' instead of dropping the symbol from the denominator.
    """
    for row in rows:
        t = int(row[0])
        if t > target_ms:
            return None
        if t + 60_000 > target_ms:
            return float(row[1])
    return None


def _direction(change: float | None) -> str:
    if change is None:
        return "pend"
    if change > 0:
        return "up"
    if change < 0:
        return "down"
    return "flat"


def cmd_evaluate(args: argparse.Namespace) -> int:
    data = json.loads(Path(args.collect).read_text(encoding="utf-8"))
    meta = data["meta"]
    snap = datetime.fromisoformat(meta["created_utc"])
    aggregates = {a["symbol"]: a for a in data["aggregates"]}
    flow = data.get("flow", {})

    calls: dict[str, dict] = {}
    call_meta: dict = {}
    if Path(args.calls).exists():
        cd = json.loads(Path(args.calls).read_text(encoding="utf-8"))
        calls = {c["symbol"]: c for c in cd.get("calls", [])}
        call_meta = cd

    # Horizons are measured from the moment the calls were frozen, not from the snapshot.
    # The snapshot only informs the call. If the origin were the snapshot time, every minute
    # between snapshot and freeze would leak already-realised price action into the short
    # horizons and flatter the agent.
    created = datetime.fromisoformat(
        call_meta.get("created_utc") or meta["created_utc"]
    )
    created_ms = int(created.timestamp() * 1000)
    now = datetime.now(timezone.utc)
    elapsed_min = (now - created).total_seconds() / 60
    stale_min = (created - snap).total_seconds() / 60

    def work(symbol: str):
        kl = api(
            KLINE,
            {
                "category": "linear",
                "symbol": symbol,
                "interval": "1",
                "start": created_ms - 3_720_000,
                "limit": 150,
            },
        )["list"]
        return symbol, list(reversed(kl))

    series: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(work, s) for s in aggregates]):
            try:
                symbol, rows = fut.result()
                series[symbol] = rows
            except Exception as exc:  # noqa: BLE001
                print(f"  series failed: {exc}", file=sys.stderr)

    # Per-symbol context, resolved once from the same 1m series that scores every horizon.
    pre: dict[str, dict] = {}
    for symbol, agg in aggregates.items():
        bars = series.get(symbol, [])
        p0 = _price_at(bars, created_ms)
        if p0 is None:
            continue
        pm = _price_at(bars, created_ms - 3_600_000)
        call = calls.get(symbol) or {}
        pre[symbol] = {
            "base": p0,
            "bars": bars,
            "call": call.get("call"),
            "p_up": call.get("p_up"),
            "spread": (agg.get("spread_bps") or 0.0) / 1e4,
            "mom": None if pm is None else ("up" if p0 > pm else "down"),
            "ret_1h_signal": agg.get("ret_1h"),
            "flow_buy_ratio": (flow.get(symbol) or {}).get("buy_vol_ratio"),
        }

    rows_out: list[dict] = []
    for symbol, ctx in pre.items():
        rec = {
            "symbol": symbol,
            "base_price": ctx["base"],
            "ret_1h_signal": ctx["ret_1h_signal"],
            "flow_buy_ratio": ctx["flow_buy_ratio"],
        }
        for horizon, minutes in (("15m", 15), ("60m", 60)):
            px = _price_at(ctx["bars"], created_ms + minutes * 60_000)
            chg = None if px is None else (px / ctx["base"] - 1) * 100
            actual = _direction(chg)
            call = ctx["call"]
            decided = actual not in ("flat", "pend")
            rec[f"chg_{horizon}"] = chg
            rec[f"actual_{horizon}"] = actual
            rec[f"call_{horizon}"] = call
            rec[f"hit_{horizon}"] = bool(call and decided and call == actual)
            rec[f"call_{horizon}_defined"] = bool(call and decided)
            if ctx["p_up"] is not None and decided:
                rec[f"brier_{horizon}"] = (
                    ctx["p_up"] - (1.0 if actual == "up" else 0.0)
                ) ** 2
            mom = ctx["mom"]
            rec[f"momentum_{horizon}"] = mom
            rec[f"momentum_hit_{horizon}"] = bool(mom and decided and mom == actual)
            rec[f"reversion_hit_{horizon}"] = bool(mom and decided and mom != actual)
        rows_out.append(rec)

    # ---- the requested curve: accuracy at every minute-horizon 1..H for the same frozen calls
    curve: dict[int, dict] = {}
    for minutes in range(1, args.horizon_max + 1):
        target = created_ms + minutes * 60_000
        hit: list[bool] = []
        hit_strict: list[bool] = []
        briers: list[float] = []
        up = down = mom_ok = 0
        for ctx in pre.values():
            if not ctx["call"]:
                continue
            px = _price_at(ctx["bars"], target)
            if px is None:
                continue
            chg = px / ctx["base"] - 1
            actual = "up" if chg > 0 else "down"
            hit.append(ctx["call"] == actual)
            if abs(chg) > ctx["spread"]:
                hit_strict.append(ctx["call"] == actual)
            up += actual == "up"
            down += actual == "down"
            mom_ok += (ctx["mom"] or "down") == actual
            if ctx["p_up"] is not None:
                briers.append((ctx["p_up"] - (1.0 if actual == "up" else 0.0)) ** 2)
        n = len(hit)
        if not n:
            continue
        curve[minutes] = {
            "n": n,
            "mine": sum(hit) / n,
            "mine_strict": (sum(hit_strict) / len(hit_strict)) if hit_strict else None,
            "n_strict": len(hit_strict),
            "up_rate": up / n,
            "always_down": down / n,
            "always_up": up / n,
            "momentum": mom_ok / n,
            "brier": (sum(briers) / len(briers)) if briers else None,
            "se": (0.25 / n) ** 0.5,
        }

    def rate(horizon: str, key: str) -> tuple[int, float | None]:
        vals = [
            r[key]
            for r in rows_out
            if r.get(f"call_{horizon}_defined") and r.get(key) is not None
        ]
        if not vals:
            return 0, None
        return len(vals), sum(1 for v in vals if v) / len(vals)

    md: list[str] = [
        "# Эксперимент v2: прогноз по калиброванной микроструктуре",
        "",
        f"- Снимок структуры: {snap.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Фиксация вызовов (отсчёт): {created.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Сверка: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Прошло от фиксации: **{elapsed_min:.1f} мин** "
        f"(структура старше вызовов на {stale_min:.1f} мин)",
        f"- Универсум: топ-{meta['top']} USDT-перпетуалов по обороту",
        f"- Поток: фиксированное окно {meta['window_s']}с, топ-{meta['flow_top']}",
        f"- Стакан: {meta['book_levels']} уровней, полосы в bps с проверкой покрытия",
        f"- Разброс времени снимка структуры: {meta.get('snap_span_s')}с — "
        "структура могла быть прочитана на столько раньше T0 (устаревание, не подглядывание)",
        f"- Оценено символов: **{len(rows_out)}**",
        "",
        "> Исследовательский эксперимент, не инвестиционный совет.",
        "",
        "## Итог",
        "",
        "| Горизонт | Символов | Моя точность | «Моментум» | «Откат» | Монетка |",
        "|---|---|---|---|---|---|",
    ]
    for horizon in ("15m", "60m"):
        n, acc = rate(horizon, f"hit_{horizon}")
        _, mom = rate(horizon, f"momentum_hit_{horizon}")
        _, rev = rate(horizon, f"reversion_hit_{horizon}")
        md.append(
            f"| {horizon} | {n} | {'—' if acc is None else f'**{acc:.1%}**'} | "
            f"{'—' if mom is None else f'{mom:.1%}'} | {'—' if rev is None else f'{rev:.1%}'} | 50.0% |"
        )

    for horizon in ("15m", "60m"):
        briers = [r[f"brier_{horizon}"] for r in rows_out if f"brier_{horizon}" in r]
        if briers:
            md += [
                "",
                f"Brier score ({horizon}): **{sum(briers) / len(briers):.4f}** (0.25 = бесполезно)",
            ]

    md += [
        "",
        "## Уверенность",
        "",
        "| Уверенность | Символов | 15м | 60м |",
        "|---|---|---|---|",
    ]
    for level in ("high", "medium", "low"):
        syms = [s for s, c in calls.items() if c.get("confidence") == level]
        if not syms:
            continue
        cells = []
        for horizon in ("15m", "60m"):
            sub = [
                r
                for r in rows_out
                if r["symbol"] in syms and r.get(f"call_{horizon}_defined")
            ]
            ok = [r for r in sub if r.get(f"hit_{horizon}")]
            cells.append(
                "—" if not sub else f"{len(ok) / len(sub):.1%} ({len(ok)}/{len(sub)})"
            )
        md.append(f"| {level} | {len(syms)} | {cells[0]} | {cells[1]} |")

    md += [
        "",
        "## Бета против анализа",
        "",
        "| Подмножество | Символов | Точность 15м | Точность 60м |",
        "|---|---|---|---|",
    ]
    for label, same in (
        ("Совпал с дрейфом (бета)", True),
        ("Отклонился от дрейфа (анализ)", False),
    ):
        items = [
            r
            for r in rows_out
            if r.get("momentum_15m") is not None
            and r.get("call_15m") is not None
            and ((r["call_15m"] == r["momentum_15m"]) == same)
        ]
        cells = []
        for horizon in ("15m", "60m"):
            sub = [r for r in items if r.get(f"call_{horizon}_defined")]
            ok = [r for r in sub if r.get(f"hit_{horizon}")]
            cells.append(
                "—" if not sub else f"{len(ok) / len(sub):.1%} ({len(ok)}/{len(sub)})"
            )
        md.append(f"| {label} | {len(items)} | {cells[0]} | {cells[1]} |")

    for horizon in ("15m", "60m"):
        dec = [
            r
            for r in rows_out
            if r.get(f"actual_{horizon}") not in (None, "pend", "flat")
        ]
        up = [r for r in dec if r[f"actual_{horizon}"] == "up"]
        if dec:
            md += [
                "",
                f"Доля реально выросших ({horizon}): **{len(up) / len(dec):.1%}** ({len(up)}/{len(dec)}). "
                f"Правило «всегда вниз» дало бы {1 - len(up) / len(dec):.1%}.",
            ]

    md += [
        "",
        "## Где я был прав (60 мин)",
        "",
        "| Символ | Мой вызов | Факт 60м | Обоснование |",
        "|---|---|---|---|",
    ]
    hits = [r for r in rows_out if r.get("hit_60m")]
    for r in sorted(hits, key=lambda r: -(r.get("chg_60m") or 0))[:20]:
        md.append(
            f"| {r['symbol']} | {calls[r['symbol']].get('call')} | {r['chg_60m']:+.2f}% | "
            f"{(calls[r['symbol']].get('rationale') or '—').replace('|', '/')} |"
        )
    if not hits:
        md.append("| — | — | — | — |")

    md += [
        "",
        "## Где я ошибся (60 мин)",
        "",
        "| Символ | Мой вызов | Факт 60м | Обоснование |",
        "|---|---|---|---|",
    ]

    def mag(r: dict) -> float:
        return abs(r.get("chg_60m") or 0)

    misses = [
        r
        for r in rows_out
        if r.get("actual_60m") not in (None, "pend", "flat") and not r.get("hit_60m")
    ]
    for r in sorted(misses, key=mag, reverse=True)[:20]:
        md.append(
            f"| {r['symbol']} | {calls.get(r['symbol'], {}).get('call', '—')} | {r['chg_60m']:+.2f}% | "
            f"{(calls.get(r['symbol'], {}).get('rationale') or '—').replace('|', '/')} |"
        )
    if not misses:
        md.append("| — | — | — | — |")

    md += [
        "",
        f"## Точность по минутным горизонтам (1…{args.horizon_max} мин от фиксации вызовов)",
        "",
        f"Отсчёт — от момента фиксации вызовов, а не от снимка: снимок был на "
        f"{stale_min:.1f} мин раньше. Иначе в короткие горизонты утекло бы уже случившееся.",
        "",
        "«Я строго» — засчитаны только ходы, превысившие спред на момент снимка,",
        "то есть отскок в пределах спреда не считается попаданием.",
        "",
        "| Мин | Символов | Я | Я строго | Всегда вниз | Всегда вверх | Моментум 60м | Brier | шум ± |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for minutes, c in curve.items():
        srt = (
            "—"
            if c["mine_strict"] is None
            else f"{c['mine_strict']:.1%} ({c['n_strict']})"
        )
        br = "—" if c["brier"] is None else f"{c['brier']:.4f}"
        md.append(
            f"| {minutes} | {c['n']} | **{c['mine']:.1%}** | {srt} | "
            f"{c['always_down']:.1%} | {c['always_up']:.1%} | {c['momentum']:.1%} | "
            f"{br} | {c['se'] * 100:.1f} п.п. |"
        )

    if curve:
        best_min = max(curve, key=lambda m: curve[m]["mine"])
        best = curve[best_min]
        md += [
            "",
            f"Лучший горизонт по точности: **{best_min} мин — {best['mine']:.1%}** "
            f"(«всегда вниз» там {best['always_down']:.1%}, «моментум» {best['momentum']:.1%}).",
            "",
            f"> Осторожно: это {len(curve)} горизонтов на одной и той же выборке. Даже при "
            "полном отсутствии сигнала максимум из стольких точек систематически выше "
            "среднего, поэтому «лучший горизонт» сам по себе ничего не доказывает. "
            f"Ориентир — шум ±{best['se'] * 100:.1f} п.п. и разрыв с колонкой «всегда вниз».",
        ]

    md += [
        "",
        "## Что исправлено против v1",
        "",
        "- полосы глубины считаются в bps, и покрытие книги проверяется: если книга не",
        "  достаёт до полосы, значение помечается `na`, а не выдаётся за измерение;",
        "- поток сделок измеряется по единому окну одинаковой длительности для всех символов;",
        "- видно изменение ликвидности за окно, а не только один кадр;",
        "- добавлены базис, история фандинга, тренды OI и L/S, уровни риск-лимитов.",
        "",
        "---",
        f"Сенсор: `microstructure_v2.py` | данные: `{Path(args.collect).name}` | прогнозы: `{Path(args.calls).name}`",
    ]

    dest_dir = Path(args.desktop) if args.desktop else Path.cwd()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"bybit-micro-v2-{now.strftime('%Y-%m-%d_%H-%M')}.md"
    dest.write_text("\n".join(md), encoding="utf-8")
    dest.with_suffix(".json").write_text(
        json.dumps(
            {
                "snapshot_utc": snap.isoformat(),
                "origin_utc": created.isoformat(),
                "stale_min": stale_min,
                "horizon_max": args.horizon_max,
                "horizon_curve": {str(k): v for k, v in curve.items()},
                "rows": rows_out,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print(f"elapsed_min={elapsed_min:.1f} symbols={len(rows_out)}")
    for horizon in ("15m", "60m"):
        n, acc = rate(horizon, f"hit_{horizon}")
        print(f"  mine {horizon}: {'n/a' if acc is None else f'{acc:.1%}'} over {n}")
    print(f"written: {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Microstructure sensor v2 (read-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="flow window first, then a calibrated snapshot")
    c.add_argument("--top", type=int, default=200)
    c.add_argument("--flow-top", type=int, default=40)
    c.add_argument("--book-top", type=int, default=5)
    c.add_argument(
        "--risk-top", type=int, default=50, help="symbols to fetch risk-limit for"
    )
    c.add_argument("--window", type=int, default=120, help="flow window seconds")
    c.add_argument("--out-dir", default="micro2")
    c.set_defaults(func=cmd_collect)

    e = sub.add_parser("evaluate", help="score the frozen calls")
    e.add_argument("collect")
    e.add_argument("calls")
    e.add_argument("--desktop", default=str(Path.home() / "Desktop"))
    e.add_argument(
        "--horizon-max",
        type=int,
        default=60,
        help="highest minute-horizon in the accuracy curve",
    )
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
