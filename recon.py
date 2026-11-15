"""Recon: how many USDT perpetuals are there and how is turnover distributed?"""

import requests

r = requests.get(
    "https://api.bybit.com/v5/market/tickers",
    params={"category": "linear"},
    timeout=30,
)
d = r.json()
rows = d["result"]["list"]
usdt = [x for x in rows if x["symbol"].endswith("USDT")]
print(f"http={r.status_code} retCode={d['retCode']}")
print(f"all linear tickers: {len(rows)}")
print(f"USDT perps:         {len(usdt)}")

tok = [float(x["turnover24h"]) for x in usdt]
tok.sort(reverse=True)
for cut in (1e6, 1e7, 1e8):
    print(f"  turnover24h > {cut:,.0f}: {sum(1 for t in tok if t > cut)}")

usdt.sort(key=lambda x: float(x["turnover24h"]), reverse=True)
print("top 5 by turnover:")
for x in usdt[:5]:
    print(f"  {x['symbol']:<14} {float(x['turnover24h']):>18,.0f}")
