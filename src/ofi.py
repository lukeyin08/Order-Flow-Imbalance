"""Order Flow Imbalance (OFI).

Single-level OFI follows Cont, Kukanov & Stoikov (2014), "The Price Impact of
Order Book Events". For each book update n, with best bid price/size (P^b, q^b)
and best ask price/size (P^a, q^a):

    dW_n = q^b_n              if P^b_n  > P^b_{n-1}
           q^b_n - q^b_{n-1}  if P^b_n == P^b_{n-1}
          -q^b_{n-1}          if P^b_n  < P^b_{n-1}

    dV_n =-q^a_{n-1}          if P^a_n  > P^a_{n-1}
           q^a_n - q^a_{n-1}  if P^a_n == P^a_{n-1}
           q^a_n              if P^a_n  < P^a_{n-1}

    OFI contribution  e_n = dW_n - dV_n

Intuition: e_n is positive when buyers add/lift (bid built up or ask pulled
away) and negative when sellers add/hit. Summed over a time bucket it measures
net directional pressure on the book.

Multi-level OFI (Xu, Cont et al., 2019) applies the same rule independently at
each of the top M price levels, giving a richer description of pressure deeper
in the book.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _ofi_from_arrays(pbid, qbid, pask, qask) -> np.ndarray:
    """Vectorised single-level OFI contributions e_n (e[0] = 0)."""
    pbid = np.asarray(pbid, float)
    qbid = np.asarray(qbid, float)
    pask = np.asarray(pask, float)
    qask = np.asarray(qask, float)

    pbid_p = np.roll(pbid, 1)
    qbid_p = np.roll(qbid, 1)
    pask_p = np.roll(pask, 1)
    qask_p = np.roll(qask, 1)

    dW = np.where(pbid > pbid_p, qbid,
                  np.where(pbid == pbid_p, qbid - qbid_p, -qbid_p))
    dV = np.where(pask > pask_p, -qask_p,
                  np.where(pask == pask_p, qask - qask_p, qask))
    e = dW - dV
    e[0] = 0.0
    return e


def compute_ofi(df: pd.DataFrame) -> np.ndarray:
    """Single-level (best quote) OFI contribution per event."""
    return _ofi_from_arrays(df["bid_px"].values, df["bid_sz"].values,
                            df["ask_px"].values, df["ask_sz"].values)


def compute_multilevel_ofi(df: pd.DataFrame, n_levels: int = C.N_LEVELS) -> pd.DataFrame:
    """Per-event OFI contribution at each of the top ``n_levels`` levels.

    Level m sits m-1 ticks behind the best quote on each side, so its price is
    derived from the best price and the tick size.
    """
    out = {}
    for m in range(1, n_levels + 1):
        pbid = df["bid_px"].values - (m - 1) * C.TICK
        pask = df["ask_px"].values + (m - 1) * C.TICK
        qbid = df[f"bid_sz_{m}"].values
        qask = df[f"ask_sz_{m}"].values
        out[f"ofi_{m}"] = _ofi_from_arrays(pbid, qbid, pask, qask)
    return pd.DataFrame(out, index=df.index)
