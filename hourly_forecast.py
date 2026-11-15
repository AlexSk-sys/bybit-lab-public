"""Hourly forecast experiment: freeze predictions now, score them later.

This is a falsifiable experiment, NOT investment advice and NOT a trading
signal. On a one-hour horizon crypto price direction is close to a coin flip.
The point is to publish explicit, locked predictions and then measure them.

Two naive hypotheses are compared against each other:
  * momentum   -- the last hour's direction continues
  * reversion  -- the last hour's direction reverses

A crude blended probability is also recorded, with tiny tilts only, so the
scorecard can report a Brier score.

Usage:
    python hourly_forecast.py baseline [--top 200] [--out-dir DIR]
    python hourly_forecast.py evaluate <baseline.json> [--desktop DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import talib

BASE_URL = "https://api.bybit.com"
TICKERS = "/v5/market/tickers"
KLINE = "/v5/market/kline"

TIMEOUT = 25
MAX_RETRIES = 3

# Heuristic tilts. Deliberately tiny and documented so the experiment is
# reproducible. They encode "I know almost nothing", not a model.
TILT_MOMENTUM = 0.04
TILT_RSI = 0.04
TILT_RANGE = 0.03
P_MIN, P_MAX = 0.35, 0.65


@dataclass
class Prediction:
    symbol: str
    last_price: float
    turnover24h: float
    ret_1h: float
    rsi14: float
    pos_100: float
    atr_pct: float
    momentum_call: str
    reversion_call: str
    p_up: float
    call: str
    confidence: str


def _get(path: str, params: dict) -> dict:
    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(BASE_URL + path, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            if payload.get("retCode") != 0:
                raise RuntimeError(f"{payload.get('retCode')}: {payload.get('retMsg')}")
            return payload["result"]
        except Exception as exc:  # noqa: BLE001 - retry then report
            last = exc
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"request failed: {path} {params}: {last}")


def fetch_usdt_perps(top: int) -> list[dict]:
    rows = _get(TICKERS, {"category": "linear"})["list"]
    usdt = [x for x in rows if x["symbol"].endswith("USDT")]
    usdt = [x for x in usdt if float(x.get("turnover24h") or 0) > 0]
    usdt.sort(key=lambda x: float(x["turnover24h"]), reverse=True)
    return usdt[:top]


def _call_from_ret(ret: float) -> str:
    if ret > 0:
        return "up"
    if ret < 0:
        return "down"
    return "flat"


def make_prediction(sym: dict) -> Prediction | None:
    symbol = sym["symbol"]
    last_price = float(sym["lastPrice"])
    rows = _get(
        KLINE, {"category": "linear", "symbol": symbol, "interval": "5", "limit": 100}
    )["list"]
    if len(rows) < 30:
        return None
    rows = list(reversed(rows))  # oldest first
    close = np.array([float(r[4]) for r in rows])
    high = np.array([float(r[2]) for r in rows])
    low = np.array([float(r[3]) for r in rows])

    ret_1h = close[-1] / close[-13] - 1 if len(close) > 13 else 0.0
    rsi = talib.RSI(close, timeperiod=14)[-1]
    atr = talib.ATR(high, low, close, timeperiod=14)[-1]

    window = min(100, len(close))
    hi, lo = float(high[-window:].max()), float(low[-window:].min())
    pos_100 = (close[-1] - lo) / (hi - lo) if hi > lo else 0.5

    momentum_call = _call_from_ret(ret_1h)
    reversion_call = {"up": "down", "down": "up", "flat": "flat"}[momentum_call]

    p_up = 0.5
    if ret_1h > 0:
        p_up += TILT_MOMENTUM
    elif ret_1h < 0:
        p_up -= TILT_MOMENTUM
    if not np.isnan(rsi):
        if rsi > 70:
            p_up -= TILT_RSI
        elif rsi < 30:
            p_up += TILT_RSI
    if pos_100 > 0.9:
        p_up -= TILT_RANGE
    elif pos_100 < 0.1:
        p_up += TILT_RANGE
    p_up = float(min(P_MAX, max(P_MIN, p_up)))

    call = "up" if p_up > 0.5 else "down" if p_up < 0.5 else "flat"
    distance = abs(p_up - 0.5)
    confidence = "high" if distance >= 0.08 else "medium" if distance >= 0.05 else "low"

    return Prediction(
        symbol=symbol,
        last_price=last_price,
        turnover24h=float(sym["turnover24h"]),
        ret_1h=float(ret_1h),
        rsi14=float(rsi) if not np.isnan(rsi) else float("nan"),
        pos_100=float(pos_100),
        atr_pct=float(atr / close[-1] * 100) if not np.isnan(atr) else float("nan"),
        momentum_call=momentum_call,
        reversion_call=reversion_call,
        p_up=p_up,
        call=call,
        confidence=confidence,
    )


def cmd_baseline(args: argparse.Namespace) -> int:
    created = datetime.now(timezone.utc)
    print(
        f"baseline at {created.isoformat()} | top={args.top} by 24h turnover",
        flush=True,
    )
    perps = fetch_usdt_perps(args.top)
    print(f"universe: {len(perps)} USDT perpetuals", flush=True)

    preds: list[Prediction] = []
    for i, sym in enumerate(perps, 1):
        try:
            p = make_prediction(sym)
            if p:
                preds.append(p)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {sym['symbol']}: {exc}", file=sys.stderr)
        if i % 25 == 0:
            print(f"  {i}/{len(perps)}", flush=True)

    payload = {
        "created_utc": created.isoformat(),
        "horizon_minutes": 60,
        "top": args.top,
        "tilts": {
            "momentum": TILT_MOMENTUM,
            "rsi": TILT_RSI,
            "range": TILT_RANGE,
            "clamp": [P_MIN, P_MAX],
        },
        "predictions": [asdict(p) for p in preds],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"baseline_{created.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    ups = sum(1 for p in preds if p.call == "up")
    print(f"saved: {path}")
    print(f"calls: {ups} up / {len(preds) - ups} down")
    for p in preds[:10]:
        print(
            f"  {p.symbol:<12} {p.call:<4} p_up={p.p_up:.2f} conf={p.confidence:<6} ret1h={p.ret_1h:+.2%}"
        )
    print(f"baseline file: {path}")
    return 0


def _pct(a: float, b: float) -> float:
    return (b - a) / a if a else 0.0


def cmd_evaluate(args: argparse.Namespace) -> int:
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    preds = base["predictions"]
    created = datetime.fromisoformat(base["created_utc"])
    evaluated = datetime.now(timezone.utc)
    elapsed_min = (evaluated - created).total_seconds() / 60

    rows = _get(TICKERS, {"category": "linear"})["list"]
    now = {x["symbol"]: float(x["lastPrice"]) for x in rows}

    scored = []
    for p in preds:
        cur = now.get(p["symbol"])
        if cur is None or not p["last_price"]:
            continue
        chg = _pct(p["last_price"], cur)
        actual = "up" if chg > 0 else "down" if chg < 0 else "flat"
        scored.append(
            {
                **p,
                "price_now": cur,
                "change_pct": chg * 100,
                "actual": actual,
                "call_ok": actual != "flat" and p["call"] == actual,
                "momentum_ok": actual != "flat" and p["momentum_call"] == actual,
                "reversion_ok": actual != "flat" and p["reversion_call"] == actual,
                "brier": (p["p_up"] - (1.0 if actual == "up" else 0.0)) ** 2,
            }
        )

    n = len(scored)
    decided = [s for s in scored if s["actual"] != "flat"]

    def acc(key: str) -> float:
        return (
            sum(1 for s in decided if s[key]) / len(decided)
            if decided
            else float("nan")
        )

    brier = sum(s["brier"] for s in scored) / n if n else float("nan")
    moved = sorted(scored, key=lambda s: abs(s["change_pct"]), reverse=True)

    def bucket(name: str, items: list[dict]) -> str:
        d = [s for s in items if s["actual"] != "flat"]
        a = sum(1 for s in d if s["call_ok"]) / len(d) if d else float("nan")
        return f"| {name} | {len(items)} | {a:.1%} |"

    md = [
        "# Эксперимент: прогноз на час по криптовалютам",
        "",
        f"- Сделано: {created.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Проверено: {evaluated.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"- Прошло: **{elapsed_min:.1f} мин** (цель 60)",
        f"- Универсум: топ-{base['top']} USDT-перпетуалов Bybit по обороту 24ч",
        f"- Оценено символов: **{n}**",
        "",
        "> Это эксперимент, а не инвестиционный совет и не торговая стратегия.",
        "> На горизонте час ожидаемая точность около 50%.",
        "",
        "## Итог",
        "",
        "| Правило | Точность |",
        "|---|---|",
        f"| Мой сводный «прогноз» | **{acc('call_ok'):.1%}** |",
        f"| Гипотеза «моментум продолжается» | {acc('momentum_ok'):.1%} |",
        f"| Гипотеза «откат» | {acc('reversion_ok'):.1%} |",
        f"| Монетка (ориентир) | 50.0% |",
        "",
        f"Brier score моего p_up: **{brier:.4f}** (0.25 = бесполезно, меньше = лучше)",
        f"Символов выросло: {sum(1 for s in scored if s['change_pct'] > 0)} / упало: {sum(1 for s in scored if s['change_pct'] < 0)}",
        "",
        "## Точность по уверенности",
        "",
        "| Уверенность | Символов | Точность |",
        "|---|---|---|",
        bucket("high", [s for s in scored if s["confidence"] == "high"]),
        bucket("medium", [s for s in scored if s["confidence"] == "medium"]),
        bucket("low", [s for s in scored if s["confidence"] == "low"]),
        "",
        "## Где я был прав",
        "",
        "| Символ | Прогноз | Уверенность | Факт | Изменение | Итог |",
        "|---|---|---|---|---|---|",
    ]
    for s in [x for x in scored if x["call_ok"]][:15]:
        md.append(
            f"| {s['symbol']} | {s['call']} | {s['confidence']} | {s['actual']} | "
            f"{s['change_pct']:+.2f}% | ✅ |"
        )
    if not any(s["call_ok"] for s in scored):
        md.append("| — | — | — | — | — | нет попаданий |")

    md += [
        "",
        "## Где я ошибся",
        "",
        "| Символ | Прогноз | Уверенность | Факт | Изменение | Итог |",
        "|---|---|---|---|---|---|",
    ]
    for s in [x for x in scored if not x["call_ok"]][:15]:
        md.append(
            f"| {s['symbol']} | {s['call']} | {s['confidence']} | {s['actual']} | "
            f"{s['change_pct']:+.2f}% | ❌ |"
        )
    if not any(not s["call_ok"] for s in scored):
        md.append("| — | — | — | — | — | ошибок нет |")

    md += [
        "",
        "## Самые сильные движения за час",
        "",
        "| Символ | Изменение | Мой прогноз | Итог |",
        "|---|---|---|---|",
    ]
    for s in moved[:10]:
        mark = "✅" if s["call_ok"] else "❌"
        md.append(f"| {s['symbol']} | {s['change_pct']:+.2f}% | {s['call']} | {mark} |")

    md += [
        "",
        "## Вывод",
        "",
        "Если точность близка к 50% — подтверждается то, что я говорил сразу:",
        "на часовом горизонте предсказуемости нет, и любой «уверенный прогноз» был бы выдумкой.",
        "Если заметно выше 50% на одной выборке — это ещё не доказательство:",
        "на 200 символах разброс около ±3.5 п.п. Нужны десятки прогонов.",
        "",
        "---",
        f"Скрипт: `bybit-lab/hourly_forecast.py` | baseline: `{Path(args.baseline).name}`",
    ]

    out = Path(args.desktop) if args.desktop else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"bybit-forecast-{evaluated.strftime('%Y-%m-%d_%H-%M')}.md"
    dest.write_text("\n".join(md), encoding="utf-8")

    json_dest = dest.with_suffix(".json")
    json_dest.write_text(json.dumps(scored, indent=2), encoding="utf-8")

    print("\n".join(md[:16]))
    print(f"\nwritten: {dest}")
    print(f"written: {json_dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Hourly crypto forecast experiment (read-only)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="freeze predictions now")
    b.add_argument("--top", type=int, default=200)
    b.add_argument("--out-dir", default="forecasts")
    b.set_defaults(func=cmd_baseline)

    e = sub.add_parser("evaluate", help="score predictions against reality")
    e.add_argument("baseline")
    e.add_argument("--desktop", default=str(Path.home() / "Desktop"))
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
