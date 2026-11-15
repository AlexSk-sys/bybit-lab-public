"""Microstructure sensor + scorecard for an agent-made forecast.

Division of labour:
  * this script is the SENSOR  -- it collects raw market microstructure and
    reduces it to a compact table. It makes no predictions.
  * the agent is the ANALYST -- it reads that table (plus the raw order books
    of the most liquid symbols) and writes explicit calls *with rationale*.
  * `evaluate` scores the agent's frozen calls against reality at two
    horizons, using 1-minute klines so the measurement does not depend on a
    scheduler firing at an exact minute.

Usage:
    python microstructure.py collect [--top 200] [--raw-top 10] [--trade-top 30]
    python microstructure.py evaluate <collect.json> <calls.json> [--desktop DIR]

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

BASE_URL = "https://api.bybit.com"
TIMEOUT = 25
MAX_RETRIES = 6
WORKERS = 8

ORDERBOOK = "/v5/market/orderbook"
KLINE = "/v5/market/kline"
TICKERS = "/v5/market/tickers"
ACCOUNT_RATIO = "/v5/market/account-ratio"
OPEN_INTEREST = "/v5/market/open-interest"
RECENT_TRADE = "/v5/market/recent-trade"

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
            if code == 10006:  # rate limited
                time.sleep(0.8 * (attempt + 1))
                continue
            if code != 0:
                raise RuntimeError(f"{code}: {payload.get('retMsg')}")
            return payload["result"]
        except Exception as exc:  # noqa: BLE001 - retry, then surface
            last = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"failed {path} {params}: {last}")


def _band_volume(
    levels: list[tuple[float, float]], mid: float, pct: float, upper: bool
) -> float:
    if upper:
        return sum(q for p, q in levels if p <= mid * (1 + pct))
    return sum(q for p, q in levels if p >= mid * (1 - pct))


def _band_usd(
    levels: list[tuple[float, float]], mid: float, pct: float, upper: bool
) -> float:
    if upper:
        return sum(p * q for p, q in levels if p <= mid * (1 + pct))
    return sum(p * q for p, q in levels if p >= mid * (1 - pct))


def _f(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_features(
    sym: dict, ob: dict, kl: list, ar: list, oi_hist: list, trades: list
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
    spread_bps = (ba - bb) / mid * 1e4

    bid_v05 = _band_volume(bids, mid, 0.005, upper=False)
    ask_v05 = _band_volume(asks, mid, 0.005, upper=True)
    bid_v1 = _band_volume(bids, mid, 0.01, upper=False)
    ask_v1 = _band_volume(asks, mid, 0.01, upper=True)

    def imb(b: float, a: float) -> float | None:
        return (b - a) / (b + a) if (b + a) > 0 else None

    depth_bid = _band_usd(bids, mid, 0.01, upper=False)
    depth_ask = _band_usd(asks, mid, 0.01, upper=True)

    close = np.array([float(r[4]) for r in reversed(kl)], dtype=float)
    high = np.array([float(r[2]) for r in reversed(kl)], dtype=float)
    low = np.array([float(r[3]) for r in reversed(kl)], dtype=float)

    ret_15m = close[-1] / close[-4] - 1 if len(close) > 3 else None
    ret_1h = close[-1] / close[-13] - 1 if len(close) > 12 else None

    atr_pct = None
    if len(close) > 15:
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
        )
        atr_pct = float(tr[-14:].mean() / close[-1] * 100)

    rvol_1h = None
    if len(close) > 13:
        logret = np.diff(np.log(close))
        rvol_1h = float(logret[-12:].std() * math.sqrt(12) * 100)

    ls_ratio = None
    if ar:
        br, sr = _f(ar[0].get("buyRatio")), _f(ar[0].get("sellRatio"))
        if br and sr:
            ls_ratio = br / sr

    oi_chg = None
    if oi_hist and len(oi_hist) >= 2:
        a, b = _f(oi_hist[0].get("openInterest")), _f(oi_hist[1].get("openInterest"))
        if a and b:
            oi_chg = (a / b - 1) * 100

    taker_buy = None
    if trades:
        buy = sum(float(t["size"]) for t in trades if t.get("side") == "Buy")
        total = sum(float(t["size"]) for t in trades)
        if total > 0:
            taker_buy = buy / total

    return {
        "symbol": sym["symbol"],
        "last_price": last_price,
        "mid": mid,
        "spread_bps": spread_bps,
        "imb_05": imb(bid_v05, ask_v05),
        "imb_1": imb(bid_v1, ask_v1),
        "depth_bid_usd": depth_bid,
        "depth_ask_usd": depth_ask,
        "fund_bps": (_f(sym.get("fundingRate")) or 0) * 1e4
        if sym.get("fundingRate") is not None
        else None,
        "oi_usd": _f(sym.get("openInterestValue")),
        "oi_chg_pct": oi_chg,
        "ls_ratio": ls_ratio,
        "taker_buy": taker_buy,
        "atr_pct": atr_pct,
        "rvol_1h_pct": rvol_1h,
        "ret_15m": ret_15m,
        "ret_1h": ret_1h,
        "turnover24h": _f(sym.get("turnover24h")),
    }


def fetch_universe(top: int) -> list[dict]:
    rows = api(TICKERS, {"category": "linear"})["list"]
    usdt = [
        x
        for x in rows
        if x["symbol"].endswith("USDT") and (_f(x.get("turnover24h")) or 0) > 0
    ]
    usdt.sort(key=lambda x: _f(x.get("turnover24h")) or 0, reverse=True)
    return usdt[:top]


def _fmt(value, digits: int = 2, sign: bool = False) -> str:
    if value is None:
        return "na"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def render_table(aggregates: list[dict], raw: dict, meta: dict) -> str:
    lines = [
        "MICROSTRUCTURE SENSOR TABLE",
        f"created_utc: {meta['created_utc']}",
        f"universe: top {meta['top']} USDT perps of {meta['universe_total']} liquid symbols by 24h turnover",
        "",
        "columns:",
        "  imb05/imb1 : order-book imbalance in +/-0.5% / +/-1% band, (bid-ask)/(bid+ask), range -1..1",
        "  ddepM      : resting depth USD within +/-1% (millions)  bid/ask",
        "  sprd       : spread in bps",
        "  fund8h     : current funding rate in bps per 8h (+ = longs pay shorts)",
        "  oiM        : open interest in USD millions",
        "  dOI        : open interest change over last 5m, %",
        "  L/S        : long/short account ratio (>1 = more accounts long)",
        "  taker      : share of taker volume that was BUY (last ~500 trades)",
        "  atr/rvol   : 5m ATR% and annualised-ish 1h realised vol %",
        "  r15m/r1h   : return over last 15m / 60m, %",
        "",
        f"{'symbol':<14}{'mid':>11}{'sprd':>7}{'imb05':>8}{'imb1':>8}{'ddepM b/a':>15}"
        f"{'fund8h':>9}{'oiM':>8}{'dOI%':>8}{'L/S':>7}{'taker':>8}{'atr%':>7}{'rvol%':>7}{'r15m%':>8}{'r1h%':>8}",
    ]

    def key(a: dict):
        return -(a.get("turnover24h") or 0)

    for a in sorted(aggregates, key=key):
        dd = f"{_fmt((a['depth_bid_usd'] or 0) / 1e6, 0)}/{_fmt((a['depth_ask_usd'] or 0) / 1e6, 0)}"
        lines.append(
            f"{a['symbol']:<14}{_fmt(a['mid'], 6 if a['mid'] < 5 else 2):>11}{_fmt(a['spread_bps'], 2):>7}"
            f"{_fmt(a['imb_05'], 3, True):>8}{_fmt(a['imb_1'], 3, True):>8}{dd:>15}"
            f"{_fmt(a['fund_bps'], 2, True):>9}{_fmt((a['oi_usd'] or 0) / 1e6, 1):>8}"
            f"{_fmt(a['oi_chg_pct'], 2, True):>8}{_fmt(a['ls_ratio'], 2):>7}"
            f"{_fmt(a['taker_buy'], 3):>8}{_fmt(a['atr_pct'], 2):>7}{_fmt(a['rvol_1h_pct'], 2):>7}"
            f"{_fmt((a['ret_15m'] or 0) * 100, 2, True):>8}{_fmt((a['ret_1h'] or 0) * 100, 2, True):>8}"
        )

    if raw:
        lines += [
            "",
            "=" * 100,
            "RAW ORDER BOOKS (top liquid symbols, 15 levels each side)",
            "=" * 100,
        ]
        for symbol, book in raw.items():
            lines.append("")
            lines.append(f"--- {symbol} ---")
            lines.append(
                f"{'bid price':>14}{'bid size':>14}   |{'ask price':>14}{'ask size':>14}"
            )
            bids = book["bids"][:15]
            asks = book["asks"][:15]
            for i in range(max(len(bids), len(asks))):
                left = (
                    f"{bids[i][0]:>14.6g}{bids[i][1]:>14.4g}"
                    if i < len(bids)
                    else " " * 28
                )
                right = (
                    f"{asks[i][0]:>14.6g}{asks[i][1]:>14.4g}"
                    if i < len(asks)
                    else " " * 28
                )
                lines.append(f"{left}   |{right}")
    return "\n".join(lines)


def cmd_collect(args: argparse.Namespace) -> int:
    created = datetime.now(timezone.utc)
    print(
        f"collect at {created.isoformat()} | top={args.top} raw={args.raw_top} trades={args.trade_top}",
        flush=True,
    )
    all_rows = api(TICKERS, {"category": "linear"})["list"]
    liquid = [
        x
        for x in all_rows
        if x["symbol"].endswith("USDT") and (_f(x.get("turnover24h")) or 0) > 0
    ]
    universe = fetch_universe(args.top)
    print(f"universe: {len(universe)} of {len(liquid)} liquid USDT perps", flush=True)

    raw_set = {x["symbol"] for x in universe[: args.raw_top]}
    trade_set = {x["symbol"] for x in universe[: args.trade_top]}

    def work(sym: dict):
        symbol = sym["symbol"]
        ob = api(ORDERBOOK, {"category": "linear", "symbol": symbol, "limit": 50})
        kl = api(
            KLINE,
            {"category": "linear", "symbol": symbol, "interval": "5", "limit": 30},
        )["list"]
        ar = api(
            ACCOUNT_RATIO,
            {"category": "linear", "symbol": symbol, "period": "5min", "limit": 1},
        )["list"]
        oi = api(
            OPEN_INTEREST,
            {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": "5min",
                "limit": 2,
            },
        )["list"]
        trades = (
            api(RECENT_TRADE, {"category": "linear", "symbol": symbol, "limit": 500})[
                "list"
            ]
            if symbol in trade_set
            else []
        )
        agg = compute_features(sym, ob, kl, ar, oi, trades)
        raw = None
        if symbol in raw_set:
            raw = {
                "bids": [[float(p), float(q)] for p, q in ob.get("b", [])],
                "asks": [[float(p), float(q)] for p, q in ob.get("a", [])],
            }
        return agg, symbol, raw

    aggregates: list[dict] = []
    raw_books: dict = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(work, s): s["symbol"] for s in universe}
        done = 0
        for fut in as_completed(futures):
            done += 1
            symbol = futures[fut]
            try:
                agg, sym, raw = fut.result()
                if agg:
                    aggregates.append(agg)
                    if raw:
                        raw_books[sym] = raw
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{symbol}: {exc}")
            if done % 25 == 0:
                print(f"  {done}/{len(universe)}", flush=True)

    meta = {
        "created_utc": created.isoformat(),
        "top": args.top,
        "raw_top": args.raw_top,
        "trade_top": args.trade_top,
        "universe_total": len(liquid),
        "collected": len(aggregates),
        "failures": failures,
    }
    payload = {"meta": meta, "aggregates": aggregates, "raw": raw_books}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = created.strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"collect_{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    table = render_table(aggregates, raw_books, meta)
    table_path = out_dir / f"table_{stamp}.txt"
    table_path.write_text(table, encoding="utf-8")

    print(
        f"collected: {len(aggregates)} symbols, {len(raw_books)} raw books, {len(failures)} failures"
    )
    print(f"json : {json_path}")
    print(f"table: {table_path}")
    if failures:
        print("failures:", file=sys.stderr)
        for f in failures[:10]:
            print(f"  {f}", file=sys.stderr)
    return 0


def _price_after(rows: list[list], minutes: int) -> float | None:
    """Close of the 1-minute bar that starts `minutes` after the baseline bar."""
    if len(rows) <= minutes:
        return None
    return float(rows[minutes][4])


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
    created = datetime.fromisoformat(meta["created_utc"])
    aggregates = {a["symbol"]: a for a in data["aggregates"]}

    calls: dict[str, dict] = {}
    calls_meta: dict = {}
    if Path(args.calls).exists():
        cd = json.loads(Path(args.calls).read_text(encoding="utf-8"))
        calls = {c["symbol"]: c for c in cd.get("calls", [])}
        calls_meta = {k: v for k, v in cd.items() if k != "calls"}

    created_ms = int(created.timestamp() * 1000)
    now = datetime.now(timezone.utc)
    elapsed = (now - created).total_seconds() / 60

    def work(symbol: str):
        kl = api(
            KLINE,
            {
                "category": "linear",
                "symbol": symbol,
                "interval": "1",
                "start": created_ms,
                "limit": 120,
            },
        )["list"]
        return symbol, list(reversed(kl))

    series: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(work, s) for s in aggregates]
        for fut in as_completed(futures):
            try:
                symbol, rows = fut.result()
                series[symbol] = rows
            except Exception as exc:  # noqa: BLE001
                print(f"  series failed: {exc}", file=sys.stderr)

    rows_out: list[dict] = []
    for symbol, agg in aggregates.items():
        bars = series.get(symbol, [])
        base = agg.get("last_price")
        if not base or not bars:
            continue
        rec = {"symbol": symbol, "base_price": base, "ret_1h_signal": agg.get("ret_1h")}
        c = calls.get(symbol)
        for horizon, minutes in (("15m", 15), ("60m", 60)):
            px = _price_after(bars, minutes)
            chg = None if px is None else (px / base - 1) * 100
            actual = _direction(chg)
            call = (c or {}).get("call")
            decided = actual not in ("flat", "pend")
            rec[f"chg_{horizon}"] = chg
            rec[f"actual_{horizon}"] = actual
            rec[f"call_{horizon}"] = call
            rec[f"hit_{horizon}"] = bool(call and decided and call == actual)
            rec[f"call_{horizon}_defined"] = bool(call and decided)
            pu = (c or {}).get("p_up")
            if pu is not None and decided:
                rec[f"brier_{horizon}"] = (pu - (1.0 if actual == "up" else 0.0)) ** 2
            ret1h = agg.get("ret_1h")
            if ret1h is not None:
                mom = "up" if ret1h > 0 else "down"
                rec[f"momentum_{horizon}"] = mom
                rec[f"momentum_hit_{horizon}"] = decided and mom == actual
                rec[f"reversion_hit_{horizon}"] = decided and mom != actual
        rows_out.append(rec)

    def agg_metric(horizon: str, key: str) -> tuple[int, float | None]:
        vals = [r[key] for r in rows_out if r.get(f"call_{horizon}_defined")]
        if not vals:
            return 0, None
        return len(vals), sum(1 for v in vals if v) / len(vals)

    md: list[str] = [
        "# Эксперимент: мой прогноз по микроструктуре",
        "",
        f"- Данные собраны: {created.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Сверка сделана: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Прошло: **{elapsed:.1f} мин** (цель 15 и 60)",
        f"- Универсум: топ-{meta['top']} USDT-перпетуалов Bybit по обороту 24ч",
        f"- Сырые стаканы прочитаны мной: топ-{meta['raw_top']}",
        f"- Символов оценено: **{len(rows_out)}**",
    ]
    if calls_meta:
        md.append(f"- Мой файл прогнозов: `{Path(args.calls).name}`")
    md += [
        "",
        "> Это исследовательский эксперимент, а не инвестиционный совет.",
        "> На горизонте 15-60 минут ожидаемая точность около 50%.",
        "",
        "## Итог: умею ли я читать рынок",
        "",
        "| Горизонт | Символов | Моя точность | Гипотеза «моментум» | Гипотеза «откат» | Монетка |",
        "|---|---|---|---|---|---|",
    ]
    for horizon in ("15m", "60m"):
        n, acc = agg_metric(horizon, f"hit_{horizon}")
        _, mom = agg_metric(horizon, f"momentum_hit_{horizon}")
        _, rev = agg_metric(horizon, f"reversion_hit_{horizon}")
        mine = "—" if acc is None else f"**{acc:.1%}**"
        mom_s = "—" if mom is None else f"{mom:.1%}"
        rev_s = "—" if rev is None else f"{rev:.1%}"
        md.append(f"| {horizon} | {n} | {mine} | {mom_s} | {rev_s} | 50.0% |")

    for horizon in ("15m", "60m"):
        briers = [r[f"brier_{horizon}"] for r in rows_out if f"brier_{horizon}" in r]
        if briers:
            md.append("")
            md.append(
                f"Brier score ({horizon}) по моим p_up: **{sum(briers) / len(briers):.4f}** (0.25 = бесполезно)"
            )

    md += [
        "",
        "## Точность по моей уверенности",
        "",
        "| Уверенность | Символов | 15м | 60м |",
        "|---|---|---|---|",
    ]
    for level in ("high", "medium", "low"):
        syms = [s for s, c in calls.items() if c.get("confidence") == level]
        if not syms:
            continue
        cell = []
        for horizon in ("15m", "60m"):
            subset = [
                r
                for r in rows_out
                if r["symbol"] in syms and r.get(f"call_{horizon}_defined")
            ]
            ok = [r for r in subset if r.get(f"hit_{horizon}")]
            cell.append(
                "—"
                if not subset
                else f"{len(ok) / len(subset):.1%} ({len(ok)}/{len(subset)})"
            )
        md.append(f"| {level} | {len(syms)} | {cell[0]} | {cell[1]} |")

    md += [
        "",
        "## Ключевая проверка: бета против анализа",
        "",
        "Совпал ли мой вызов с тупым правилом «моментум продолжается»? Если да — это бета,",
        "ставка на общий дрейф рынка, а не мой навык. Ценность имеет только подмножество,",
        "где я **отклонился** от дрейфа (откаты, сквизы, стены в стакане).",
        "",
        "| Подмножество | Символов | Точность 15м | Точность 60м |",
        "|---|---|---|---|",
    ]

    def subset(same: bool) -> list[dict]:
        out = []
        for r in rows_out:
            m = r.get("momentum_15m")
            c = r.get("call_15m")
            if m is None or c is None:
                continue
            if (c == m) == same:
                out.append(r)
        return out

    for label, same in (
        ("Совпал с дрейфом (бета)", True),
        ("Отклонился от дрейфа (анализ)", False),
    ):
        items = subset(same)
        cells = []
        for horizon in ("15m", "60m"):
            defined = [r for r in items if r.get(f"call_{horizon}_defined")]
            ok = [r for r in defined if r.get(f"hit_{horizon}")]
            cells.append(
                "—"
                if not defined
                else f"{len(ok) / len(defined):.1%} ({len(ok)}/{len(defined)})"
            )
        md.append(f"| {label} | {len(items)} | {cells[0]} | {cells[1]} |")

    up15 = [r for r in rows_out if r.get("actual_15m") == "up"]
    up60 = [r for r in rows_out if r.get("actual_60m") == "up"]
    d15 = [r for r in rows_out if r.get("actual_15m") not in (None, "pend", "flat")]
    d60 = [r for r in rows_out if r.get("actual_60m") not in (None, "pend", "flat")]
    md += [
        "",
        f"Доля реально выросших символов: 15м — "
        f"{'—' if not d15 else f'{len(up15) / len(d15):.1%}'} ({len(up15)}/{len(d15)}), "
        f"60м — {'—' if not d60 else f'{len(up60) / len(d60):.1%}'} ({len(up60)}/{len(d60)}).",
        "Если эта доля сильно ниже 50%, то «угадать» движение вниз легко и без навыка.",
    ]

    def table_block(title: str, items: list[dict], horizon: str) -> list[str]:
        out = [
            "",
            f"## {title}",
            "",
            "| Символ | Мой вызов | Факт 15м | Факт 60м | Обоснование |",
            "|---|---|---|---|---|",
        ]
        for r in items:
            c = calls.get(r["symbol"], {})
            ch15 = r.get("chg_15m")
            ch60 = r.get("chg_60m")
            f15 = "—" if ch15 is None else f"{ch15:+.2f}%"
            f60 = "—" if ch60 is None else f"{ch60:+.2f}%"
            mark = "✅" if r.get(f"hit_{horizon}") else "❌"
            rat = (c.get("rationale") or "—").replace("|", "/")
            out.append(
                f"| {r['symbol']} | {mark} {c.get('call', '—')} | {f15} | {f60} | {rat} |"
            )
        if not items:
            out.append("| — | — | — | — | — |")
        return out

    hits = [r for r in rows_out if r.get("hit_60m")]
    misses = [
        r
        for r in rows_out
        if r.get("actual_60m") not in (None, "pend", "flat") and not r.get("hit_60m")
    ]
    md += table_block(
        "Где я был прав (горизонт 60 мин)",
        sorted(hits, key=lambda r: -(r.get("chg_60m") or 0))[:20],
        "60m",
    )
    md += table_block(
        "Где я ошибся (горизонт 60 мин)",
        sorted(misses, key=lambda r: abs(r.get("chg_60m") or 0), reverse=True)[:20],
        "60m",
    )

    md += [
        "",
        "## Самые сильные движения за час",
        "",
        "| Символ | Изменение 60м | Мой вызов | Итог |",
        "|---|---|---|---|",
    ]

    def mag(r: dict) -> float:
        return abs(r.get("chg_60m") or 0)

    for r in sorted(
        [x for x in rows_out if x.get("actual_60m") not in (None, "pend", "flat")],
        key=mag,
        reverse=True,
    )[:15]:
        c = calls.get(r["symbol"], {})
        mark = "✅" if r.get("hit_60m") else "❌"
        ch = r.get("chg_60m")
        md.append(
            f"| {r['symbol']} | {'—' if ch is None else f'{ch:+.2f}%'} | {c.get('call', '—')} | {mark} |"
        )

    md += [
        "",
        "## Как читать этот документ",
        "",
        "Скрипт-датчик собрал микроструктуру, но **не делал выводов** — выводы сделал я,",
        "прочитав таблицу и сырые стаканы топ-ликвидных пар. Обоснования были",
        "заморожены на диск до факта, поэтому подогнать их задним числом невозможно.",
        "",
        "Если моя точность в пределах ~50% — значит моё «чтение рынка» было иллюзией,",
        "и честный вывод: микроструктура на этих горизонтах мне ничего не даёт.",
        "Выше 50% на одной выборке — ещё не доказательство: на таком числе символов",
        "случайный разброс составляет несколько процентных пунктов.",
        "",
        "---",
        f"Сенсор: `microstructure.py` | данные: `{Path(args.collect).name}` | прогнозы: `{Path(args.calls).name}`",
    ]

    dest_dir = Path(args.desktop) if args.desktop else Path.cwd()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"bybit-micro-forecast-{now.strftime('%Y-%m-%d_%H-%M')}.md"
    dest.write_text("\n".join(md), encoding="utf-8")
    json_dest = dest.with_suffix(".json")
    json_dest.write_text(json.dumps(rows_out, indent=1), encoding="utf-8")

    print(f"elapsed_min={elapsed:.1f} symbols={len(rows_out)}")
    for horizon in ("15m", "60m"):
        n, acc = agg_metric(horizon, f"hit_{horizon}")
        print(f"  mine {horizon}: {'n/a' if acc is None else f'{acc:.1%}'} over {n}")
    print(f"written: {dest}")
    print(f"written: {json_dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Microstructure sensor + agent scorecard (read-only)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser(
        "collect", help="gather microstructure (script makes no predictions)"
    )
    c.add_argument("--top", type=int, default=200)
    c.add_argument("--raw-top", type=int, default=10)
    c.add_argument("--trade-top", type=int, default=30)
    c.add_argument("--out-dir", default="micro")
    c.set_defaults(func=cmd_collect)

    e = sub.add_parser("evaluate", help="score the agent's frozen calls")
    e.add_argument("collect")
    e.add_argument("calls")
    e.add_argument("--desktop", default=str(Path.home() / "Desktop"))
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
