"""Avellaneda-Stoikov market making, with and without an OFI tilt.

A taker cannot profit from OFI after costs, but a market maker earns the spread.
Its problem is inventory risk and being run over by informed flow. Because OFI
forecasts the next move, the maker can skew its quotes to position inventory ahead
of it, improving risk-adjusted PnL at controlled inventory.

Three quoting policies are compared on the same mid-price path:
  * "naive"  : fixed symmetric spread around the mid, no inventory control;
  * "as"     : Avellaneda-Stoikov reservation price + optimal spread (inventory
               aware), no signal;
  * "as_ofi" : same, plus a reservation-price skew proportional to the OFI
               forecast.

Fills follow the Avellaneda-Stoikov intensity lambda(delta) = A * exp(-kappa*delta),
modulated by the *realised* next move so that quotes resting on the wrong side
get hit more often right before an adverse move -- i.e. adverse selection is
built into the fill model, not assumed away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class MMParams:
    gamma: float = 14.0      # inventory risk aversion
    kappa: float = 100.0     # fill-intensity decay (1/dollar); sets base spread
    A: float = 0.9           # base fill intensity (per second)
    tau: float = 1.0         # A-S horizon parameter
    skew: float = 8.0        # strength of the OFI reservation-price tilt
    adverse: float = 0.9     # strength of adverse selection in the fill model
    vol_window: int = 300    # rolling window (buckets) for volatility estimate
    dt: float = C.BUCKET_SECONDS


def simulate_mm(b: pd.DataFrame, beta: float, policy: str = "as_ofi",
                params: MMParams | None = None, seed: int = C.SEED) -> dict:
    p = params or MMParams()
    d = b.dropna(subset=["ofi", "fwd_ret_1", "ret"]).reset_index(drop=True)
    mid = d["mid_last"].values
    ofi = d["ofi"].values
    true_drift = d["fwd_ret_1"].values                       # m_{t+1} - m_t
    sigma = (d["ret"].rolling(p.vol_window).std()
             .bfill().fillna(d["ret"].std()).values)
    t_hours = d["t_start"].values / 3600.0
    rng = np.random.default_rng(seed)
    n = len(d)

    q = 0.0
    cash = 0.0
    pnl = np.empty(n)
    inv = np.empty(n)
    fills = 0
    adverse_markout = 0.0       # sum of (fill PnL vs next mid) -- negative = bad

    # Avellaneda-Stoikov parameters (kappa, gamma) are calibrated at the ~$100
    # reference price. Scale them to the data's price level so the maker quotes and
    # fills sensibly at any price (e.g. BTC near $65k), not only in the simulator.
    scale = C.INIT_PRICE / float(np.median(mid))
    kappa = p.kappa * scale
    gamma = p.gamma * scale

    floor = -3 * C.TICK         # cap how aggressive a quote can get
    for t in range(n):
        s2 = sigma[t] ** 2
        delta_total = gamma * s2 * p.tau + (2.0 / gamma) * np.log(1 + gamma / kappa)
        half = delta_total / 2.0

        r = mid[t]
        if policy in ("as", "as_ofi"):
            r -= q * gamma * s2 * p.tau                      # inventory control
        if policy == "as_ofi":
            r += p.skew * beta * ofi[t]                      # OFI tilt

        ask = r + half
        bid = r - half
        d_ask = max(ask - mid[t], floor)
        d_bid = max(mid[t] - bid, floor)

        mod = true_drift[t] / (sigma[t] + 1e-12)             # ~N(0,1)
        lam_ask = p.A * np.exp(-kappa * d_ask) * np.exp(p.adverse * mod)
        lam_bid = p.A * np.exp(-kappa * d_bid) * np.exp(-p.adverse * mod)
        p_ask = min(0.95, 1.0 - np.exp(-lam_ask * p.dt))
        p_bid = min(0.95, 1.0 - np.exp(-lam_bid * p.dt))

        nxt = mid[t + 1] if t + 1 < n else mid[t]
        if rng.random() < p_ask:        # our ask lifted: we SELL at ask
            cash += ask
            q -= 1.0
            fills += 1
            adverse_markout += (ask - nxt)      # +ve if we sold above next mid
        if rng.random() < p_bid:        # our bid hit: we BUY at bid
            cash -= bid
            q += 1.0
            fills += 1
            adverse_markout += (nxt - bid)      # +ve if we bought below next mid

        pnl[t] = cash + q * mid[t]
        inv[t] = q

    incr = np.diff(pnl, prepend=0.0)
    sd = incr.std()
    sharpe = float(incr.mean() / sd * np.sqrt(C.BUCKETS_PER_YEAR)) if sd > 0 else 0.0
    return {
        "policy": policy,
        "pnl": pnl,
        "inventory": inv,
        "t_hours": t_hours,
        "final_pnl": float(pnl[-1]),
        "pnl_sharpe": sharpe,
        "inv_abs_mean": float(np.mean(np.abs(inv))),
        "inv_abs_max": float(np.max(np.abs(inv))),
        "n_fills": int(fills),
        "markout_per_fill": float(adverse_markout / fills) if fills else 0.0,
    }


def compare_market_makers(b: pd.DataFrame, beta: float,
                          params: MMParams | None = None) -> dict:
    return {pol: simulate_mm(b, beta, policy=pol, params=params)
            for pol in ("naive", "as", "as_ofi")}
