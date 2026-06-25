"""Event-driven backtest of the OFI signal traded as a liquidity TAKER.

Workflow (strictly causal):
  * fit the forecasting model fwd_ret_1 ~ ofi on the training period only;
  * on the out-of-sample period, take a long/short/flat position each bucket
    from the sign of the predicted move;
  * realise the next bucket's return; charge spread + fees on every change in
    position.

Two policies are compared:
  * "naive"     : act on the sign of every prediction (high turnover);
  * "selective" : only trade when the predicted move clears the round-trip cost.

The expected result is the honest one: gross PnL is positive (the signal is
real) but net PnL is not (a taker pays more in spread than the edge is worth).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def fit_signal(b: pd.DataFrame, train_frac: float = 0.7):
    d = b.dropna(subset=["fwd_ret_1", "ofi"]).reset_index(drop=True)
    cut = int(len(d) * train_frac)
    tr = d.iloc[:cut]
    # Clean OLS slope/intercept (avoids any ddof mismatch between cov and var).
    slope, intercept = np.polyfit(tr["ofi"].values, tr["fwd_ret_1"].values, 1)
    return d, cut, float(intercept), float(slope)


def _metrics(gross, net, pos, dpos):
    def sharpe(x):
        sd = x.std()
        return float(x.mean() / sd * np.sqrt(C.BUCKETS_PER_YEAR)) if sd > 0 else 0.0

    eq_net = np.cumsum(net)
    peak = np.maximum.accumulate(eq_net)
    max_dd = float(np.max(peak - eq_net)) if len(eq_net) else 0.0
    in_mkt = pos != 0
    hit = float(np.mean(gross[in_mkt] > 0)) if in_mkt.any() else 0.0
    n_tr = int(np.sum(dpos > 0))
    return {
        "total_gross": float(np.sum(gross)),
        "total_net": float(np.sum(net)),
        "gross_per_trade_ticks": float(np.sum(gross) / n_tr / C.TICK) if n_tr else 0.0,
        "net_per_trade_ticks": float(np.sum(net) / n_tr / C.TICK) if n_tr else 0.0,
        "sharpe_gross": sharpe(gross),
        "sharpe_net": sharpe(net),
        "max_drawdown_net": max_dd,
        "n_trades": n_tr,
        "turnover_per_bucket": float(np.mean(dpos)),
        "pct_time_in_market": float(np.mean(in_mkt)),
        "hit_rate": hit,
    }


def backtest_taker(b: pd.DataFrame, train_frac: float = 0.7,
                   policy: str = "naive") -> dict:
    d, cut, alpha, beta = fit_signal(b, train_frac)
    te = d.iloc[cut:].reset_index(drop=True)

    pred = alpha + beta * te["ofi"].values
    ret = te["fwd_ret_1"].values
    half = te["spread"].values / 2.0
    price = float(te["mid_last"].mean())
    fee = (C.TAKER_FEE_BPS / 1e4) * price
    per_unit_cost = half + fee                      # one spread-cross + fee

    if policy == "selective":
        thr = 2.0 * half.mean() + 2.0 * fee          # round-trip hurdle
    else:
        thr = 0.0

    pos = np.where(pred > thr, 1.0, np.where(pred < -thr, -1.0, 0.0))
    gross = pos * ret
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    cost = dpos * per_unit_cost
    net = gross - cost

    out = _metrics(gross, net, pos, dpos)
    out.update({
        "policy": policy,
        "alpha": alpha, "beta": beta,
        "equity_gross": np.cumsum(gross),
        "equity_net": np.cumsum(net),
        "t_hours": te["t_start"].values / 3600.0,
    })
    return out
