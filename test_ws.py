"""Check whether the Bybit public WebSocket is reachable from this environment."""

from __future__ import annotations

import json
import time

from websockets.sync.client import connect

URL = "wss://stream.bybit.com/v5/public/linear"

print(f"connecting {URL} ...")
t0 = time.time()
try:
    with connect(URL, open_timeout=20, close_timeout=5) as ws:
        print(f"  connected in {time.time() - t0:.2f}s")
        ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": ["publicTrade.BTCUSDT", "orderbook.1.ETHUSDT"],
                }
            )
        )
        got = 0
        deadline = time.time() + 25
        while got < 4 and time.time() < deadline:
            raw = ws.recv(timeout=10)
            msg = json.loads(raw)
            topic = msg.get("topic", msg.get("op", "?"))
            if "data" in msg and isinstance(msg["data"], list) and msg["data"]:
                first = msg["data"][0]
                if topic.startswith("publicTrade"):
                    print(
                        f"  [{topic}] n={len(msg['data'])} sample: side={first.get('S')} size={first.get('v')} price={first.get('p')}"
                    )
                else:
                    print(
                        f"  [{topic}] type={msg.get('type')} sample keys={sorted(first.keys())}"
                    )
            else:
                print(f"  [{topic}] {json.dumps(msg)[:120]}")
            got += 1
    print("OK: websocket works")
except Exception as exc:  # noqa: BLE001
    print(f"FAIL: {type(exc).__name__}: {exc}")
