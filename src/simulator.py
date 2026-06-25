"""Event-driven limit-order-book simulator.

The goal is NOT to hand the analysis its answer. It is to generate a realistic
top-of-book stream in which Order Flow Imbalance (OFI) relates to price changes
the same way it does in real markets -- as an *emergent* property of the order
flow, not a hard-coded correlation.

Mechanism
---------
* A latent "alpha" state z_t follows a slow Ornstein-Uhlenbeck process. It is
  never observed by the analysis.
* z_t tilts the arrival intensity of market buys vs market sells. When z_t > 0
  there is persistent net buying.
* Market orders consume the resting queue at the best quote. When a queue is
  exhausted the price moves one tick. Limit adds/cancels rebuild and erode it.

Two things therefore emerge naturally:
  1. Contemporaneous link: within a bucket, net market buying both pushes the
     price up AND registers as positive OFI -> high contemporaneous R^2.
  2. Predictability: because z_t is persistent, a positive-OFI bucket tends to
     be followed by further buying -> a small, decaying forecasting edge.

The persistence of z_t and the queue depth jointly control how large that
forecasting edge is relative to the spread -- which is the whole point of the
project's cost analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Parameters governing the micro-dynamics (kept here, not in config, because
# they are simulator-internal rather than analysis choices).
# ---------------------------------------------------------------------------
@dataclass
class SimParams:
    n_events: int = C.N_EVENTS
    n_levels: int = C.N_LEVELS
    band: int = C.N_LEVELS + 3        # levels tracked per side (>= n_levels)

    base_rate: float = 10.0           # events per second (sets the clock)

    # event-type mix (per event, before the buy/sell split); sums to 1
    p_limit_add: float = 0.48
    p_cancel: float = 0.20
    p_market: float = 0.32

    p_improve: float = 0.45           # chance a limit add improves the price

    depth_mean: float = 30.0          # mean resting size at a fresh level
    add_mean: float = 2.0             # mean size of a limit add
    cancel_mean: float = 2.0          # mean size of a cancel
    trade_mean: float = 4.0           # mean size of a market order

    # latent alpha (Ornstein-Uhlenbeck), per-second parameters
    alpha_tau: float = 30.0           # mean-reversion timescale (seconds)
    alpha_std: float = 1.0            # stationary std of z
    alpha_gain: float = 1.0           # how strongly z tilts market-order side


def _new_depth(rng: np.random.Generator, mean: float) -> int:
    return int(max(1, rng.poisson(mean)))


def simulate(params: SimParams | None = None, seed: int = C.SEED) -> pd.DataFrame:
    """Run the simulation and return a per-event top-of-book DataFrame."""
    p = params or SimParams()
    rng = np.random.default_rng(seed)
    N = p.n_events
    L = p.band
    nl = p.n_levels

    # ------------------------------------------------------------------ book
    # Levels are integer tick indices; price = INIT_PRICE + level * TICK.
    best_bid_lvl = -1
    best_ask_lvl = 1
    # size bands: index 0 is the best quote, increasing index = deeper
    bid_sz = [_new_depth(rng, p.depth_mean) for _ in range(L)]
    ask_sz = [_new_depth(rng, p.depth_mean) for _ in range(L)]

    # --------------------------------------------------------------- latent z
    dt_mean = 1.0 / p.base_rate
    phi = np.exp(-dt_mean / p.alpha_tau)                  # OU persistence / event
    eps_std = p.alpha_std * np.sqrt(1.0 - phi * phi)      # OU innovation std
    z = 0.0

    # --------------------------------------------------------------- storage
    ts = np.empty(N)
    bid_px = np.empty(N)
    ask_px = np.empty(N)
    bid_levels = np.empty((N, nl))
    ask_levels = np.empty((N, nl))
    z_arr = np.empty(N)
    trade_sign = np.zeros(N, dtype=np.int8)
    trade_size = np.zeros(N)

    t = 0.0
    p_la, p_ca, p_mk = p.p_limit_add, p.p_cancel, p.p_market

    for n in range(N):
        # advance clock and latent alpha
        t += rng.exponential(dt_mean)
        z = phi * z + rng.normal(0.0, eps_std)

        # choose event type
        u = rng.random()
        if u < p_la:
            etype = 0          # limit add
        elif u < p_la + p_ca:
            etype = 1          # cancel
        else:
            etype = 2          # market order

        if etype == 0:         # ---------------------------------- limit add
            # side: symmetric
            if rng.random() < 0.5:   # add to bid
                if (best_ask_lvl - best_bid_lvl) >= 2 and rng.random() < p.p_improve:
                    best_bid_lvl += 1
                    bid_sz.insert(0, _new_depth(rng, p.depth_mean * 0.3))
                    bid_sz.pop()
                else:
                    bid_sz[0] += _new_depth(rng, p.add_mean)
            else:                    # add to ask
                if (best_ask_lvl - best_bid_lvl) >= 2 and rng.random() < p.p_improve:
                    best_ask_lvl -= 1
                    ask_sz.insert(0, _new_depth(rng, p.depth_mean * 0.3))
                    ask_sz.pop()
                else:
                    ask_sz[0] += _new_depth(rng, p.add_mean)

        elif etype == 1:       # ------------------------------------- cancel
            if rng.random() < 0.5:   # cancel from bid
                bid_sz[0] -= _new_depth(rng, p.cancel_mean)
                if bid_sz[0] <= 0:
                    bid_sz.pop(0)
                    best_bid_lvl -= 1
                    bid_sz.append(_new_depth(rng, p.depth_mean))
            else:                    # cancel from ask
                ask_sz[0] -= _new_depth(rng, p.cancel_mean)
                if ask_sz[0] <= 0:
                    ask_sz.pop(0)
                    best_ask_lvl += 1
                    ask_sz.append(_new_depth(rng, p.depth_mean))

        else:                  # ------------------------------- market order
            # side tilted by latent alpha z
            w_buy = np.exp(p.alpha_gain * z)
            w_sell = np.exp(-p.alpha_gain * z)
            buy = rng.random() < (w_buy / (w_buy + w_sell))
            s = _new_depth(rng, p.trade_mean)
            if buy:              # consume the ask queue, price ticks up on depletion
                trade_sign[n] = 1
                trade_size[n] = s
                while s > 0:
                    if ask_sz[0] > s:
                        ask_sz[0] -= s
                        s = 0
                    else:
                        s -= ask_sz[0]
                        ask_sz.pop(0)
                        best_ask_lvl += 1
                        ask_sz.append(_new_depth(rng, p.depth_mean))
            else:                # consume the bid queue, price ticks down
                trade_sign[n] = -1
                trade_size[n] = s
                while s > 0:
                    if bid_sz[0] > s:
                        bid_sz[0] -= s
                        s = 0
                    else:
                        s -= bid_sz[0]
                        bid_sz.pop(0)
                        best_bid_lvl -= 1
                        bid_sz.append(_new_depth(rng, p.depth_mean))

        # keep a one-tick minimum spread (guard against improvement crossing)
        if best_ask_lvl <= best_bid_lvl:
            best_ask_lvl = best_bid_lvl + 1

        # ------------------------------------------------------------- record
        ts[n] = t
        bid_px[n] = C.INIT_PRICE + best_bid_lvl * C.TICK
        ask_px[n] = C.INIT_PRICE + best_ask_lvl * C.TICK
        bid_levels[n] = bid_sz[:nl]
        ask_levels[n] = ask_sz[:nl]
        z_arr[n] = z
        # trade_sign / trade_size already set (0 for non-trade events)

    # --------------------------------------------------------------- assemble
    data = {
        "ts": ts,
        "bid_px": bid_px,
        "ask_px": ask_px,
        "mid": 0.5 * (bid_px + ask_px),
        "spread": ask_px - bid_px,
        "bid_sz": bid_levels[:, 0],
        "ask_sz": ask_levels[:, 0],
        "trade_sign": trade_sign,
        "trade_size": trade_size,
        "z": z_arr,
    }
    for i in range(nl):
        data[f"bid_sz_{i+1}"] = bid_levels[:, i]
        data[f"ask_sz_{i+1}"] = ask_levels[:, i]
    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = simulate()
    print(f"events           : {len(df):,}")
    print(f"duration (hours) : {df['ts'].iloc[-1] / 3600:.2f}")
    print(f"mid range        : {df['mid'].min():.2f} -> {df['mid'].max():.2f}")
    print(f"mean spread (ticks): {(df['spread'].mean() / C.TICK):.2f}")
    print(f"mean best depth  : {df['bid_sz'].mean():.1f} / {df['ask_sz'].mean():.1f}")
    print(f"trades           : {(df['trade_sign'] != 0).sum():,} "
          f"({100*(df['trade_sign']!=0).mean():.1f}% of events)")
    print(f"z std            : {df['z'].std():.3f}")
