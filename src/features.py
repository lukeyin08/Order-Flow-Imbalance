"""Aggregate the event stream into fixed-length time buckets and build the
OFI features and forward-return labels used by the analysis.

Alignment (important for avoiding look-ahead bias):
  * ofi[k]          : sum of OFI contributions during bucket k
  * ret[k]          : mid change *during* bucket k  (contemporaneous)
  * fwd_ret_h[k]    : mid change over the h buckets *after* bucket k (forecast)

The contemporaneous regression ret ~ ofi measures price impact. The forecasting
regression fwd_ret_h ~ ofi uses only information available at the end of bucket
k to predict the strictly-future move -- no leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .ofi import compute_ofi, compute_multilevel_ofi


def build_buckets(df: pd.DataFrame,
                  n_levels: int = C.N_LEVELS,
                  bucket_seconds: float = C.BUCKET_SECONDS) -> pd.DataFrame:
    e = compute_ofi(df)
    ml = compute_multilevel_ofi(df, n_levels)

    work = pd.DataFrame({
        "ts": df["ts"].values,
        "mid": df["mid"].values,
        "spread": df["spread"].values,
        "ofi": e,
        "is_trade": (df["trade_sign"].values != 0).astype(int),
    })
    for m in range(1, n_levels + 1):
        work[f"ofi_{m}"] = ml[f"ofi_{m}"].values
    work["bucket"] = np.floor(work["ts"].values / bucket_seconds).astype(int)

    g = work.groupby("bucket")
    agg = g.agg(
        t_start=("ts", "first"),
        mid_first=("mid", "first"),
        mid_last=("mid", "last"),
        spread=("spread", "mean"),
        ofi=("ofi", "sum"),
        n_events=("ofi", "size"),
        n_trades=("is_trade", "sum"),
    )
    for m in range(1, n_levels + 1):
        agg[f"ofi_{m}"] = g[f"ofi_{m}"].sum()

    # Reindex to a contiguous bucket grid so empty seconds are explicit
    full = pd.RangeIndex(agg.index.min(), agg.index.max() + 1, name="bucket")
    agg = agg.reindex(full)
    agg["mid_last"] = agg["mid_last"].ffill()
    agg["mid_first"] = agg["mid_first"].fillna(agg["mid_last"])
    agg["spread"] = agg["spread"].ffill()
    ofi_cols = ["ofi"] + [f"ofi_{m}" for m in range(1, n_levels + 1)]
    agg[ofi_cols] = agg[ofi_cols].fillna(0.0)
    agg[["n_events", "n_trades"]] = agg[["n_events", "n_trades"]].fillna(0.0)
    agg["t_start"] = agg["t_start"].ffill()

    # Returns (in dollars). Use end-of-bucket mid for a contiguous price path.
    agg["ret"] = agg["mid_last"].diff()
    for h in C.FORECAST_HORIZONS:
        agg[f"fwd_ret_{h}"] = agg["mid_last"].shift(-h) - agg["mid_last"]

    return agg.reset_index()


if __name__ == "__main__":
    from .simulator import simulate
    df = simulate()
    b = build_buckets(df)
    print(f"buckets          : {len(b):,}")
    print(f"mean events/bkt  : {b['n_events'].mean():.1f}")
    print(f"OFI std          : {b['ofi'].std():.1f}")
    print(f"ret std (ticks)  : {b['ret'].std()/C.TICK:.2f}")
    print(b[["ofi", "ret", "fwd_ret_1"]].describe())
