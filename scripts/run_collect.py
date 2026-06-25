#!/usr/bin/env python3
"""Collect real Binance order-book data into the project's data/ folder.

Run this on a machine with internet access (it needs the websocket-client
package). The output CSV matches the simulator's schema, so you can point the
rest of the pipeline at real data:

    pip install websocket-client
    python scripts/run_collect.py --symbol btcusdt --minutes 30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collector import collect


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--venue", default="com", choices=["com", "us"],
                    help="binance.com (global, default) or binance.us "
                         "(use this if .com is geo-restricted, e.g. in the US)")
    args = ap.parse_args()
    collect(symbol=args.symbol, minutes=args.minutes, out_path=args.out,
            venue=args.venue)
