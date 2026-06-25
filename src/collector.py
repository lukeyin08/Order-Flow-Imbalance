"""Live limit-order-book collector for Binance (spot).

Run this on a machine with internet access to record real top-of-book data in
the SAME schema the simulator produces, so the entire analysis pipeline runs
unchanged on real data:

    python scripts/run_collect.py --symbol btcusdt --minutes 30

It subscribes to two public Binance websocket streams:
  * <symbol>@depth20@100ms : top-20 book snapshot every 100 ms
  * <symbol>@trade         : individual trades (for aggressor side)

Each depth snapshot becomes one row; the most recent trade side/size since the
previous snapshot is attached. Output is a CSV with columns:
    ts, bid_px, ask_px, mid, spread, bid_sz, ask_sz,
    bid_sz_1..N, ask_sz_1..N, trade_sign, trade_size

Requires:  pip install websocket-client
Note: this does not run inside the project's analysis sandbox (no network); it
is meant to be run by you locally. The simulator is the reproducible default.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

from . import config as C

# Binance global is the default. Its public API/websocket is geo-restricted in
# some regions (e.g. the US returns HTTP 451); pass venue="us" to use Binance.US,
# which serves the identical stream schema (@depth20@100ms / @trade).
STREAM_HOSTS = {
    "com": "wss://stream.binance.com:9443/stream",
    "us": "wss://stream.binance.us:9443/stream",
}


def _columns(n_levels: int):
    cols = ["ts", "bid_px", "ask_px", "mid", "spread", "bid_sz", "ask_sz"]
    for i in range(1, n_levels + 1):
        cols += [f"bid_sz_{i}", f"ask_sz_{i}"]
    cols += ["trade_sign", "trade_size"]
    return cols


def collect(symbol: str = "btcusdt",
            minutes: float = 30.0,
            n_levels: int = C.N_LEVELS,
            out_path: str | None = None,
            venue: str = "com") -> str:
    try:
        import websocket  # websocket-client
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install websocket-client first:  pip install websocket-client") from exc

    if venue not in STREAM_HOSTS:
        raise SystemExit(f"Unknown venue {venue!r}; choose one of {sorted(STREAM_HOSTS)}")
    symbol = symbol.lower()
    out_path = out_path or os.path.join(C.DATA_DIR, f"book_{symbol}.csv")
    streams = f"{symbol}@depth20@100ms/{symbol}@trade"
    url = f"{STREAM_HOSTS[venue]}?streams={streams}"
    cols = _columns(n_levels)

    f = open(out_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(cols)

    state = {"trade_sign": 0, "trade_size": 0.0, "rows": 0,
             "deadline": time.time() + minutes * 60.0}

    def on_message(ws, message):
        msg = json.loads(message)
        stream = msg.get("stream", "")
        data = msg.get("data", {})
        if stream.endswith("@trade"):
            # m == True means buyer is the market maker -> aggressor is a seller
            state["trade_sign"] = -1 if data.get("m") else 1
            state["trade_size"] = float(data.get("q", 0.0))
            return
        # depth snapshot
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if len(bids) < n_levels or len(asks) < n_levels:
            return
        bid_px = float(bids[0][0]); ask_px = float(asks[0][0])
        row = [time.time(), bid_px, ask_px, 0.5 * (bid_px + ask_px),
               ask_px - bid_px, float(bids[0][1]), float(asks[0][1])]
        for i in range(n_levels):
            row += [float(bids[i][1]), float(asks[i][1])]
        row += [state["trade_sign"], state["trade_size"]]
        writer.writerow(row)
        state["rows"] += 1
        state["trade_sign"] = 0          # reset until next trade arrives
        state["trade_size"] = 0.0
        if state["rows"] % 200 == 0:
            print(f"  {state['rows']} snapshots  mid={row[3]:.2f}", end="\r")
        if time.time() >= state["deadline"]:
            ws.close()

    def on_error(ws, error):  # pragma: no cover
        print("websocket error:", error)

    print(f"Collecting {symbol} (binance.{venue}) for {minutes:g} min -> {out_path}")
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error)
    ws.run_forever(ping_interval=180, ping_timeout=10)
    f.close()
    print(f"\nSaved {state['rows']} snapshots to {out_path}")
    return out_path
