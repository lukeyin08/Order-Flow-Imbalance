#!/usr/bin/env python3
"""End-to-end pipeline: simulate -> OFI -> regressions -> costs -> backtest ->
market maker -> figures + metrics.json. Deterministic given the seed.

    python scripts/run_analysis.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config as C
from src import plotting as P
from src.simulator import simulate
from src.features import build_buckets
from src.analysis import price_impact, multilevel_impact, forecast_decay, out_of_sample
from src.backtest import fit_signal, backtest_taker
from src.costs import edge_vs_cost
from src.market_maker import compare_market_makers


def _clean(obj):
    """Make numpy/pandas objects JSON-serialisable; drop big arrays."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main(data_path=None):
    if data_path:
        import pandas as pd
        print(f"1/7  loading order book from {data_path} ...")
        df = pd.read_csv(data_path)
    else:
        print("1/7  simulating limit order book ...")
        df = simulate()
    dur_h = (df["ts"].iloc[-1] - df["ts"].iloc[0]) / 3600.0

    print("2/7  building buckets and OFI features ...")
    b = build_buckets(df)

    print("3/7  predictive regressions ...")
    impact = price_impact(b)
    ml = multilevel_impact(b)
    decay = forecast_decay(b)
    oos = out_of_sample(b)

    print("4/7  signal fit + cost analysis ...")
    _, _, alpha, beta = fit_signal(b, train_frac=0.7)
    costs = edge_vs_cost(b, beta=beta, alpha=alpha)

    print("5/7  taker backtest ...")
    bt_naive = backtest_taker(b, policy="naive")
    bt_sel = backtest_taker(b, policy="selective")

    print("6/7  market-maker simulation ...")
    mm = compare_market_makers(b, beta=beta)

    print("7/7  figures ...")
    # Pick a price/OFI window that fits the dataset: the simulator (~30k
    # buckets) keeps the original start=3000 slice; a shorter real-data sample
    # falls back to a centred window so the figure still renders.
    win = min(900, len(b))
    start = 3000 if len(b) >= 3000 + win else max(0, (len(b) - win) // 2)
    figs = [
        P.fig_orderbook(df),
        P.fig_price_and_ofi(b, start=start, n=win),
        P.fig_impact_scatter(b, impact),
        P.fig_decay(decay),
        P.fig_edge_vs_cost(costs),
        P.fig_backtest(bt_naive, bt_sel),
        P.fig_market_maker(mm),
    ]

    metrics = {
        "data": {
            "n_events": int(len(df)),
            "duration_hours": round(dur_h, 2),
            "n_buckets": int(len(b)),
            "bucket_seconds": C.BUCKET_SECONDS,
            "median_spread_ticks": round(float(df["spread"].median() / C.TICK), 2),
            "mean_best_depth": round(float(df["bid_sz"].mean()), 1),
            "trade_fraction": round(float((df["trade_sign"] != 0).mean()), 3),
        },
        "price_impact": _clean(impact),
        "multilevel": _clean(ml),
        "forecast_decay": [ _clean(r) for r in decay.to_dict("records") ],
        "out_of_sample": _clean(oos),
        "costs": _clean(costs),
        "taker_backtest": {
            "naive": _clean(bt_naive),
            "selective": _clean(bt_sel),
        },
        "market_maker": {k: _clean(v) for k, v in mm.items()},
    }
    out = os.path.join(C.RESULTS_DIR, "metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)

    # -------------------------------------------------------------- summary
    print("\n================  SUMMARY  ================")
    print(f"data            : {len(df):,} events, {dur_h:.1f}h, {len(b):,} buckets, "
          f"spread {metrics['data']['median_spread_ticks']:.0f} tick")
    print(f"price impact    : R^2={impact['r2']:.2f}  (t={impact['tstat']:.0f})  "
          f"-> OFI explains {impact['r2']*100:.0f}% of contemporaneous moves")
    print(f"multi-level     : single R^2={ml['r2_single']:.2f} -> "
          f"5-level R^2={ml['r2_multi']:.2f}")
    print(f"forecast (1s)   : IC={decay.iloc[0]['ic']:.3f}  R^2={decay.iloc[0]['r2']:.3f}")
    print(f"out-of-sample   : IS IC={oos['ic_in_sample']:.3f}  OOS IC={oos['ic_out_sample']:.3f}")
    print(f"edge vs cost    : edge={costs['mean_abs_edge_ticks']:.3f}t  "
          f"round-trip cost={costs['round_trip_cost_ticks']:.2f}t  "
          f"({costs['round_trip_cost']/costs['mean_abs_edge']:.0f}x)")
    print(f"taker (naive)   : gross +{bt_naive['gross_per_trade_ticks']:.3f}t/trade -> "
          f"net {bt_naive['net_per_trade_ticks']:.3f}t/trade  "
          f"(${bt_naive['total_gross']:.1f} -> ${bt_naive['total_net']:.1f})")
    print(f"taker (select.) : {bt_sel['n_trades']} trades — edge never clears the spread")
    for pol in ("naive", "as", "as_ofi"):
        r = mm[pol]
        print(f"MM {pol:7s}   : PnL ${r['final_pnl']:7.1f}  |inv|max {r['inv_abs_max']:4.0f}  "
              f"markout/fill {r['markout_per_fill']:.4f}  Sharpe {r['pnl_sharpe']:.0f}")
    print("==========================================")
    print(f"\nfigures -> {C.FIG_DIR}")
    print(f"metrics -> {out}")
    return metrics


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="OFI analysis pipeline")
    ap.add_argument("--data", default=None,
                    help="CSV of collected book data (run_collect.py output); "
                         "if omitted, the calibrated simulator is used")
    args = ap.parse_args()
    main(data_path=args.data)
