"""The central economic question: is the forecasting edge big enough to trade?

A liquidity *taker* pays, per round trip, roughly the full spread (cross on the
way in, cross on the way out) plus fees. A liquidity *maker* does the opposite:
it earns the spread, but risks being adversely selected -- filled right before
the price moves against it. This module quantifies the edge against both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def taker_cost_per_unit(price: float, half_spread: float) -> float:
    """Cost of one unit of trading for a taker: half the spread + the fee."""
    fee = (C.TAKER_FEE_BPS / 1e4) * price
    return half_spread + fee


def edge_vs_cost(b: pd.DataFrame, beta: float, alpha: float = 0.0) -> dict:
    """Compare the model's predicted next-bucket move with trading costs.

    `beta`/`alpha` come from the forecasting regression fwd_ret_1 ~ ofi.
    Everything is expressed in dollars and in ticks for readability.
    """
    d = b.dropna(subset=["ofi", "fwd_ret_1"]).copy()
    pred = alpha + beta * d["ofi"].values            # predicted next move ($)

    price = float(d["mid_last"].mean())
    half_spread = float(d["spread"].mean()) / 2.0
    fee = (C.TAKER_FEE_BPS / 1e4) * price

    # Round-trip taker cost = cross spread twice (= full spread) + fee twice
    rt_cost = 2.0 * half_spread + 2.0 * fee

    mean_abs_edge = float(np.mean(np.abs(pred)))
    frac_tradeable = float(np.mean(np.abs(pred) > rt_cost))

    return {
        "price": price,
        "tick": C.TICK,
        "half_spread": half_spread,
        "half_spread_ticks": half_spread / C.TICK,
        "taker_fee": fee,
        "round_trip_cost": rt_cost,
        "round_trip_cost_ticks": rt_cost / C.TICK,
        "mean_abs_edge": mean_abs_edge,
        "mean_abs_edge_ticks": mean_abs_edge / C.TICK,
        "edge_to_cost_ratio": mean_abs_edge / rt_cost,
        "frac_buckets_tradeable_as_taker": frac_tradeable,
        # Maker view: capture ~half the spread per round trip if not run over
        "maker_capture_per_rt": 2.0 * half_spread,
        "maker_capture_per_rt_ticks": (2.0 * half_spread) / C.TICK,
    }
