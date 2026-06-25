"""Predictive analysis of OFI.

Three questions, in increasing order of difficulty / honesty:
  1. Price impact (contemporaneous): does OFI explain price moves *now*?
  2. Forecasting (decay): does OFI predict *future* moves, and how fast does
     that edge decay with horizon?
  3. Out-of-sample: does a model fit on the past still work on unseen data?

All regressions use Newey-West (HAC) standard errors because high-frequency
returns are serially correlated; ordinary OLS errors would overstate
significance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C

HAC_LAGS = 10


def _ols(y, X, add_const=True, lags=HAC_LAGS):
    X = sm.add_constant(X) if add_const else X
    model = sm.OLS(y, X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    return model


def price_impact(b: pd.DataFrame) -> dict:
    """Contemporaneous regression: ret ~ ofi (single level)."""
    d = b.dropna(subset=["ret", "ofi"])
    m = _ols(d["ret"].values, d["ofi"].values)
    return {
        "beta": float(m.params[1]),
        "tstat": float(m.tvalues[1]),
        "r2": float(m.rsquared),
        "n": int(m.nobs),
    }


def multilevel_impact(b: pd.DataFrame, n_levels: int = C.N_LEVELS) -> dict:
    """Compare single-level vs multi-level OFI on contemporaneous returns."""
    cols = [f"ofi_{m}" for m in range(1, n_levels + 1)]
    d = b.dropna(subset=["ret"] + cols)
    single = _ols(d["ret"].values, d["ofi_1"].values)
    multi = _ols(d["ret"].values, d[cols].values)
    return {
        "r2_single": float(single.rsquared),
        "r2_multi": float(multi.rsquared),
        "betas_multi": [float(x) for x in multi.params[1:]],
        "n": int(multi.nobs),
    }


def forecast_decay(b: pd.DataFrame, horizons=C.FORECAST_HORIZONS) -> pd.DataFrame:
    """In-sample forecasting power of OFI at several future horizons."""
    rows = []
    for h in horizons:
        col = f"fwd_ret_{h}"
        d = b.dropna(subset=[col, "ofi"])
        # Overlapping h-bucket returns induce MA(h-1) residuals; widen HAC lags.
        m = _ols(d[col].values, d["ofi"].values, lags=max(HAC_LAGS, h))
        ic = float(np.corrcoef(d["ofi"].values, d[col].values)[0, 1])
        rows.append({"horizon": h, "beta": float(m.params[1]),
                     "tstat": float(m.tvalues[1]), "r2": float(m.rsquared),
                     "ic": ic, "n": int(m.nobs)})
    return pd.DataFrame(rows)


def out_of_sample(b: pd.DataFrame, train_frac: float = 0.7) -> dict:
    """Chronological train/test split, forecasting the next bucket (h=1)."""
    d = b.dropna(subset=["fwd_ret_1", "ofi"]).reset_index(drop=True)
    n = len(d)
    cut = int(n * train_frac)
    tr, te = d.iloc[:cut], d.iloc[cut:]

    # Fit on train only
    m = _ols(tr["fwd_ret_1"].values, tr["ofi"].values)
    a, bcoef = float(m.params[0]), float(m.params[1])

    def r2(y, yhat):
        y = np.asarray(y); yhat = np.asarray(yhat)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

    pred_tr = a + bcoef * tr["ofi"].values
    pred_te = a + bcoef * te["ofi"].values
    return {
        "beta_train": bcoef,
        "r2_in_sample": float(r2(tr["fwd_ret_1"].values, pred_tr)),
        "r2_out_sample": float(r2(te["fwd_ret_1"].values, pred_te)),
        "ic_in_sample": float(np.corrcoef(tr["ofi"], tr["fwd_ret_1"])[0, 1]),
        "ic_out_sample": float(np.corrcoef(te["ofi"], te["fwd_ret_1"])[0, 1]),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "cut_time_h": float(te["t_start"].iloc[0] / 3600.0),
    }
