"""Bybit market report: numbers for the machine, PNG for the human.

Data source: Bybit public V5 REST API. No API key, no auth, read-only.
The MCP server connected to OpenCode exposes the same data to the agent;
this script is the standalone numeric layer so results are reproducible
without going through a model.

Usage:
    python bybit_lab.py --symbol BTCUSDT --interval 60
    python bybit_lab.py --symbol BTCUSDT --interval 60 --json
    python bybit_lab.py --symbol BTCUSDT --interval 60 --png chart.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import requests
import talib

BASE_URL = "https://api.bybit.com"
KLINE_PATH = "/v5/market/kline"
TICKER_PATH = "/v5/market/tickers"
COLUMNS = ["start_ms", "open", "high", "low", "close", "volume", "turnover"]


def fetch_klines(
    symbol: str,
    category: str = "linear",
    interval: str = "60",
    limit: int = 500,
    timeout: int = 20,
) -> pd.DataFrame:
    """Fetch OHLCV klines. Bybit returns newest-first; we re-sort oldest-first."""
    limit = max(1, min(limit, 1000))
    resp = requests.get(
        BASE_URL + KLINE_PATH,
        params={
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}"
        )
    rows = payload["result"]["list"]
    if not rows:
        raise RuntimeError(f"No klines returned for {symbol} ({category}, {interval})")

    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col])
    df = df.sort_values("start_ms").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["start_ms"], unit="ms", utc=True)
    return df


def fetch_ticker(symbol: str, category: str = "linear", timeout: int = 20) -> dict:
    resp = requests.get(
        BASE_URL + TICKER_PATH,
        params={"category": category, "symbol": symbol},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}"
        )
    rows = payload["result"].get("list") or []
    return rows[0] if rows else {}


@dataclass
class Snapshot:
    """Derived numbers -- the actual input for a model. No pixels involved."""

    symbol: str
    category: str
    interval: str
    bars: int
    last_close: float
    # trend
    ema20: float
    ema50: float
    ema200: float
    price_vs_ema20_pct: float
    price_vs_ema50_pct: float
    price_vs_ema200_pct: float
    ema_stack: str  # bull / bear / mixed
    # momentum
    rsi14: float
    macd: float
    macd_signal: float
    macd_hist: float
    macd_hist_prev: float
    # volatility
    atr14: float
    atr14_pct: float
    bb_upper: float
    bb_lower: float
    bb_width_pct: float
    bb_percent_b: float
    # structure / candles
    range_position_100: float
    body_ratio: float
    upper_wick_ratio: float
    lower_wick_ratio: float
    vol_ratio_20: float
    # long-window context
    high_100: float
    low_100: float
    dist_from_high_100_pct: float
    dist_from_low_100_pct: float


def _safe(value) -> float:
    """TA-Lib yields NaN for the warm-up window; keep JSON valid."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    return float(value)


def compute_snapshot(
    df: pd.DataFrame, symbol: str, category: str, interval: str
) -> Snapshot:
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    ema20 = talib.EMA(close, timeperiod=20)
    ema50 = talib.EMA(close, timeperiod=50)
    ema200 = talib.EMA(close, timeperiod=200)
    rsi14 = talib.RSI(close, timeperiod=14)
    macd, macd_sig, macd_hist = talib.MACD(close, 12, 26, 9)
    atr14 = talib.ATR(high, low, close, timeperiod=14)
    bb_up, bb_mid, bb_low = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

    last = -1
    c = close[last]
    e20, e50, e200 = _safe(ema20[last]), _safe(ema50[last]), _safe(ema200[last])

    if e20 > e50 > e200:
        stack = "bull"
    elif e20 < e50 < e200:
        stack = "bear"
    else:
        stack = "mixed"

    window = min(100, len(df))
    hi100 = float(np.max(high[-window:]))
    lo100 = float(np.min(low[-window:]))
    span = hi100 - lo100
    range_pos = (c - lo100) / span if span else 0.0

    op, hi, lo = (
        float(df["open"].iloc[last]),
        float(df["high"].iloc[last]),
        float(df["low"].iloc[last]),
    )
    rng = hi - lo
    body = abs(c - op)
    upper_wick = hi - max(c, op)
    lower_wick = min(c, op) - lo

    bb_up_v, bb_low_v = _safe(bb_up[last]), _safe(bb_low[last])
    bb_mid_v = _safe(bb_mid[last])
    bb_width = (bb_up_v - bb_low_v) / bb_mid_v * 100 if bb_mid_v else float("nan")
    percent_b = (
        (c - bb_low_v) / (bb_up_v - bb_low_v) if (bb_up_v - bb_low_v) else float("nan")
    )

    vol_sma20 = talib.SMA(vol, timeperiod=20)[last]

    return Snapshot(
        symbol=symbol,
        category=category,
        interval=interval,
        bars=len(df),
        last_close=c,
        ema20=e20,
        ema50=e50,
        ema200=e200,
        price_vs_ema20_pct=(c - e20) / e20 * 100 if e20 else float("nan"),
        price_vs_ema50_pct=(c - e50) / e50 * 100 if e50 else float("nan"),
        price_vs_ema200_pct=(c - e200) / e200 * 100 if e200 else float("nan"),
        ema_stack=stack,
        rsi14=_safe(rsi14[last]),
        macd=_safe(macd[last]),
        macd_signal=_safe(macd_sig[last]),
        macd_hist=_safe(macd_hist[last]),
        macd_hist_prev=_safe(macd_hist[-2]) if len(df) > 1 else float("nan"),
        atr14=_safe(atr14[last]),
        atr14_pct=_safe(atr14[last]) / c * 100 if c else float("nan"),
        bb_upper=bb_up_v,
        bb_lower=bb_low_v,
        bb_width_pct=bb_width,
        bb_percent_b=percent_b,
        range_position_100=range_pos,
        body_ratio=body / rng if rng else 0.0,
        upper_wick_ratio=upper_wick / rng if rng else 0.0,
        lower_wick_ratio=lower_wick / rng if rng else 0.0,
        vol_ratio_20=float(vol[last] / vol_sma20) if vol_sma20 else float("nan"),
        high_100=hi100,
        low_100=lo100,
        dist_from_high_100_pct=(hi100 - c) / hi100 * 100 if hi100 else float("nan"),
        dist_from_low_100_pct=(c - lo100) / lo100 * 100 if lo100 else float("nan"),
    )


def render_png(df: pd.DataFrame, snap: Snapshot, path: str) -> None:
    """Chart for the human eye -- deliberately NOT the model's input."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    view = df.tail(min(200, len(df)))
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    x = np.arange(len(view))
    ax.plot(x, view["close"], lw=1.2, color="#1f77b4", label="close")
    for period, color in ((20, "#ff7f0e"), (50, "#2ca02c")):
        ema = talib.EMA(view["close"].to_numpy(dtype=float), timeperiod=period)
        ax.plot(x, ema, lw=1.0, color=color, label=f"EMA{period}")
    ax.set_title(
        f"{snap.symbol} {snap.category} {snap.interval}m | close {snap.last_close:,.2f} | "
        f"RSI {snap.rsi14:.1f} | MACD hist {snap.macd_hist:.2f} | stack {snap.ema_stack}"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    colors = [
        "#2ca02c" if c >= o else "#d62728" for c, o in zip(view["close"], view["open"])
    ]
    ax2.bar(x, view["volume"], color=colors, width=0.8)
    ax2.set_ylabel("volume")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Bybit numeric market report (read-only, no API key)"
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument(
        "--category", default="linear", choices=["spot", "linear", "inverse"]
    )
    ap.add_argument("--interval", default="60")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument(
        "--json", action="store_true", help="print machine-readable JSON only"
    )
    ap.add_argument(
        "--png", help="write a candlestick/indicator chart for human review"
    )
    args = ap.parse_args(argv)

    try:
        df = fetch_klines(args.symbol, args.category, args.interval, args.limit)
        ticker = fetch_ticker(args.symbol, args.category)
    except (requests.RequestException, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    snap = compute_snapshot(df, args.symbol, args.category, args.interval)

    if args.png:
        render_png(df, snap, args.png)

    if args.json:
        print(json.dumps(asdict(snap), indent=2, allow_nan=True))
        return 0

    t = f"{snap.last_close:,.2f}"
    print(
        f"== {snap.symbol} [{snap.category} {snap.interval}m] bars={snap.bars} close={t}"
    )
    if ticker:
        print(
            f"   24h: {ticker.get('price24hPcnt', '?')} | high {ticker.get('highPrice24h', '?')}"
            f" | low {ticker.get('lowPrice24h', '?')} | OI {ticker.get('openInterest', '?')}"
        )
    print(
        f"   trend    EMA20 {snap.ema20:,.1f} ({snap.price_vs_ema20_pct:+.2f}%)"
        f" EMA50 {snap.ema50:,.1f} ({snap.price_vs_ema50_pct:+.2f}%)"
        f" EMA200 {snap.ema200:,.1f} ({snap.price_vs_ema200_pct:+.2f}%) stack={snap.ema_stack}"
    )
    print(
        f"   momentum RSI14 {snap.rsi14:.1f} | MACD {snap.macd:+.2f} signal {snap.macd_signal:+.2f}"
        f" hist {snap.macd_hist:+.2f} (prev {snap.macd_hist_prev:+.2f})"
    )
    print(
        f"   volatil  ATR14 {snap.atr14:,.1f} ({snap.atr14_pct:.2f}%)"
        f" | BB width {snap.bb_width_pct:.2f}% %b {snap.bb_percent_b:.2f}"
    )
    print(
        f"   struct   pos-in-100 {snap.range_position_100:.2f}"
        f" | body {snap.body_ratio:.2f} wicks {snap.upper_wick_ratio:.2f}/{snap.lower_wick_ratio:.2f}"
        f" | vol x{snap.vol_ratio_20:.2f}"
    )
    print(
        f"   context  high100 {snap.high_100:,.1f} ({snap.dist_from_high_100_pct:.2f}% away)"
        f" | low100 {snap.low_100:,.1f} ({snap.dist_from_low_100_pct:.2f}% away)"
    )
    if args.png:
        print(f"   chart    {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
