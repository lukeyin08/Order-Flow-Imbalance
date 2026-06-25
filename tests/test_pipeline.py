"""Pipeline-level checks, most importantly that the forecasting label contains
no look-ahead: fwd_ret_1[k] must equal the *next* bucket's return ret[k+1]."""
import numpy as np

from src.simulator import simulate, SimParams
from src.features import build_buckets


def _small_buckets():
    return build_buckets(simulate(SimParams(n_events=30_000)))


def test_forward_label_is_strictly_next_bucket():
    b = _small_buckets()
    lhs = b["fwd_ret_1"].values
    rhs = b["ret"].shift(-1).values
    mask = ~np.isnan(lhs) & ~np.isnan(rhs)
    # fwd_ret_1[k] = mid[k+1]-mid[k] must equal ret[k+1] exactly (no overlap)
    assert np.allclose(lhs[mask], rhs[mask])


def test_core_columns_have_no_nans():
    b = _small_buckets()
    assert b["ofi"].notna().all()
    assert b["mid_last"].notna().all()


def test_contemporaneous_relationship_is_positive():
    # Sanity: OFI and same-bucket return should be positively correlated.
    b = _small_buckets().dropna(subset=["ret", "ofi"])
    assert np.corrcoef(b["ofi"], b["ret"])[0, 1] > 0.3
